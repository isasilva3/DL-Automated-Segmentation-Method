"""## Setup imports"""

from monai.handlers.tensorboard_handlers import SummaryWriter
from monai.transforms import Rand3DElasticd, RandGaussianNoised, RandScaleIntensityd, RandGaussianSmoothd, \
    RandAdjustContrastd, RandFlipd

"""## Setup imports"""

import glob
import os
import shutil
import tempfile
import nibabel as nib
import numpy as np

import matplotlib.pyplot as plt
import torch

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.utils import set_determinism, GridSampleMode, GridSamplePadMode
from monai.networks.nets import SegResNet
from monai.data.nifti_saver import NiftiSaver
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import compute_meandice, DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    KeepLargestConnectedComponent,
    LabelToContour,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    ToTensord,
)
from monai.utils import first, set_determinism
from numpy import math

print_config()
print("INFERENCE")

md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

root_dir = "/home/imoreira"
#root_dir = "C:\\Users\\isasi\\Downloads"
data_dir = os.path.join(root_dir, "Data")
out_dir = os.path.join(data_dir, "Kidneys_Best_Model")
tensorboard_dir= "//home//imoreira//Data//Tensorboard_Kidneys"


writer = SummaryWriter(log_dir=tensorboard_dir)

"""## Set dataset path"""

train_images = sorted(glob.glob(os.path.join(data_dir, "Images", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

#data_dicts = [
#     {"image": image_name}
#     for image_name in train_images
# ]

#n = len(data_dicts)
#train_files, val_files = data_dicts[:-3], data_dicts[-3:]
#train_files, val_files = data_dicts[:int(n*0.8)], data_dicts[int(n*0.2):]

val_files, train_files, test_files = data_dicts[0:8], data_dicts[8:40], data_dicts[40:50]


"""## Set deterministic training for reproducibility"""

set_determinism(seed=0)

'''
Label 1: Bladder
Label 2: Liver
Label 3: Lungs
Label 4: Heart
Label 5: Pancreas
'''

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-300, a_max=300, b_min=0.0, b_max=1.0, clip=True,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
            pos=1,
            neg=1,
            num_samples=4,
            image_key="image",
            image_threshold=0,
        ),
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(5, 30),
            magnitude_range=(70, 90),
            spatial_size=None,
            prob=0.5,
            rotate_range=(0, -math.pi / 36, math.pi / 36, 0),  # -15, 15 / -5, 5
            shear_range=None,
            translate_range=None,
            scale_range=(0.15, 0.15, 0.15),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            # as_tensor_output=False
        ),
        RandGaussianNoised(
            keys=["image"],
            prob=0.5,
            mean=0.0,
            std=0.03
            # allow_missing_keys=False
        ),
       #RandScaleIntensityd(
       #    keys=["image"],
       #    factors=0.05,  # this is 10%, try 5%
       #    prob=0.5
       #),
       #RandGaussianSmoothd(
       #    keys=["image"],
       #    sigma_x=(0.25, 1.5),
       #    sigma_y=(0.25, 1.5),
       #    sigma_z=(0.25, 1.5),
       #   prob=0.5,
       #   approx='erf'
            # allow_missing_keys=False
       #),
       RandAdjustContrastd(
          keys=["image"],
          prob=0.5,
          gamma=(0.9, 1.1)
          #allow_missing_keys=False
       ),
        # user can also add other random transforms
        # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=(96, 96, 96),
        #             rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),

       ToTensord(keys=["image", "label"]),
    ]
)
train_inf_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-300, a_max=300, b_min=0.0, b_max=1.0, clip=True,
        ),
        ToTensord(keys=["image", "label"]),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-300, a_max=300, b_min=0.0, b_max=1.0, clip=True,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)
test_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-300, a_max=300, b_min=0.0, b_max=1.0, clip=True,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

"""## Check transforms in DataLoader"""

check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
#image, label = (check_data["image"][0][0], check_data["label"][0][0])
#print(f"image shape: {image.shape}, label shape: {label.shape}")
# plot the slice [:, :, 80]
#fig = plt.figure("check", (12, 6)) #figure size
#plt.subplot(1, 2, 1)
#plt.title("image")
#plt.imshow(image[:, :, 80], cmap="gray")
#plt.subplot(1, 2, 2)
#plt.title("label")
#plt.imshow(label[:, :, 80])
#plt.show()
#fig.savefig('my_figure.png')


"""## Define CacheDataset and DataLoader for training and validation

Here we use CacheDataset to accelerate training and validation process, it's 10x faster than the regular Dataset.  
To achieve best performance, set `cache_rate=1.0` to cache all the data, if memory is not enough, set lower value.  
Users can also set `cache_num` instead of `cache_rate`, will use the minimum value of the 2 settings.  
And set `num_workers` to enable multi-threads during caching.  
If want to to try the regular Dataset, just change to use the commented code below.
"""

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0)
# train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)


train_inf_ds = CacheDataset(data=train_files, transform=train_inf_transforms, cache_rate=1.0, num_workers=0)
train_inf_loader = DataLoader(train_inf_ds, batch_size=1, num_workers=0)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=0)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=0)

test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=0)
#test_ds = Dataset(data=test_files)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)


"""## Create Model, Loss, Optimizer"""

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
#loss_function = DiceLoss(to_onehot_y=True, softmax=True)
#optimizer = torch.optim.Adam(model.parameters(), 1e-4)

loss_function = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, lambda_dice=0.5, lambda_ce=0.5)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
dice_metric = DiceMetric(include_background=False, reduction="mean")
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5) ##

#largest = KeepLargestConnectedComponent(applied_labels=[0])

"""## Makes the Inferences """

model.load_state_dict(torch.load(os.path.join(out_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    #saver = NiftiSaver(output_dir='C:\\Users\\isasi\\Downloads\\Bladder_Segs_Out')
    saver = NiftiSaver(output_dir='//home//imoreira//Kidneys_Segs_Out',
                       output_postfix="seg_kidneys",
                       output_ext=".nii.gz",
                       mode="nearest",
                       padding_mode="zeros"
                       )
    for val_data in val_loader:
        val_images = val_data["image"].to(device)
        roi_size = (96, 96, 96)
        sw_batch_size = 4

        val_outputs = sliding_window_inference(
            val_images, roi_size, sw_batch_size, model, overlap=0.8
        )

        # val_outputs = torch.squeeze(val_outputs, dim=1)

        val_outputs = val_outputs.argmax(dim=1, keepdim=True)

        #val_outputs = largest(val_outputs)

        val_outputs = val_outputs.cpu().clone().numpy()
        val_outputs = val_outputs.astype(np.bool)

        saver.save_batch(val_outputs, val_data["image_meta_dict"])

