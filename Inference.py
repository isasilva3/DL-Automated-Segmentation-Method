# """## Check best model output with the input image and label"""
# import glob
# import os
# import torch
# import nibabel as nib
#
# from MONAI.monai.data import NiftiSaver, CacheDataset, DataLoader
# from MONAI.monai.inferers import sliding_window_inference
# from MONAI.monai.networks.layers import Norm
# from MONAI.monai.networks.nets import UNet
# from MONAI.monai.transforms import LoadImage, AddChannel, Spacing, Orientation, ScaleIntensityRange, ToTensord, Compose, \
#     LoadImaged, AddChanneld, Spacingd, Orientationd, ScaleIntensityRanged
#
# root_dir = "//home//imoreira"
# data_dir = os.path.join(root_dir, "Data")
# out_dir= os.path.join(data_dir, "Best_Model")
#
# test_images = sorted(glob.glob(os.path.join(data_dir, "Test_Images", "*.nii.gz")))
#
# data_dicts = [
#     {"image": image_name}
#     for image_name in test_images
# ]
#
# test_transforms = Compose(
#     [
#         LoadImaged(keys=["image"]),
#         AddChanneld(keys=["image"]),
#         Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
#         Orientationd(keys=["image"], axcodes="RAS"),
#         ScaleIntensityRanged(
#             keys=["image"], a_min=-1000, a_max=300, b_min=0.0, b_max=1.0, clip=True,
#         ),
#         #CropForegroundd(keys=["image", "label"], source_key="image"),
#         ToTensord(keys=["image"]),
#     ]
# )
#
# test_ds = CacheDataset(data=test_images, transform=test_transforms, cache_rate=1.0, num_workers=1)
# #test_ds = Dataset(data=test_files)
# test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)
#
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# out_dir = "//home//imoreira//Data//Best_Model"
# #out_dir = "C:\\Users\\isasi\\Downloads\\Bladder_Best_Model"
#
# model = UNet(
#     dimensions=3,
#     in_channels=1,
#     out_channels=6, #6 channels, 1 for each organ more background
#     channels=(16, 32, 64, 128, 256),
#     strides=(2, 2, 2, 2),
#     num_res_units=2,
#     norm=Norm.BATCH,
# ).to(device)
#
#
# """## Makes the Inferences """
#
# model.load_state_dict(torch.load(os.path.join(out_dir, "best_metric_model.pth")))
# model.eval()
# with torch.no_grad():
#     #saver = NiftiSaver(output_dir='C:\\Users\\isasi\\Downloads\\Bladder_Segs_Out')
#     saver = NiftiSaver(output_dir='//home//imoreira//Segs_Out',
#                        output_postfix="seg",
#                        output_ext=".nii.gz",
#                        mode="nearest",
#                        padding_mode="zeros"
#                        )
#     for test_data in test_loader:
#         test_images = test_data["image"].to(device)
#         roi_size = (160, 160, 160)
#         sw_batch_size = 1
#         val_outputs = sliding_window_inference(
#             test_images, roi_size, sw_batch_size, model
#         )
#         val_outputs = val_outputs.argmax(dim=1, keepdim=True)
#         val_outputs = val_outputs.squeeze(dim=0).cpu().clone().numpy()
#         #val_outputs = largest(val_outputs)
#
#         #val_outputs = val_outputs.cpu().clone().numpy()
#         #val_outputs = val_outputs.astype(np.bool)
#
#
#         saver.save_batch(val_outputs, test_data["image_meta_dict"])

# -*- coding: utf-8 -*-

from MONAI.monai.transforms import Rand3DElasticd, RandGaussianNoised, RandScaleIntensityd, RandGaussianSmoothd, \
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
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import set_determinism, GridSampleMode, GridSamplePadMode
from monai.networks.nets import SegResNet
from monai.data.nifti_saver import NiftiSaver
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import compute_meandice
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

root_dir = "//home//imoreira"
#root_dir = "C:\\Users\\isasi\\Downloads"
data_dir = os.path.join(root_dir, "Data")
out_dir = os.path.join(data_dir, "Best_Model")

"""## Set dataset path"""

train_images = sorted(glob.glob(os.path.join(data_dir, "Images", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "Labels", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]
#n = len(data_dicts)
#train_files, val_files = data_dicts[:-3], data_dicts[-3:]
#train_files, val_files = data_dicts[:int(n*0.8)], data_dicts[int(n*0.2):]

val_files, train_files, test_files = data_dicts[0:8], data_dicts[8:40], data_dicts[40:50]


"""## Set deterministic training for reproducibility"""

set_determinism(seed=0)

'''
Label 1: Bladder
Label 2: Heart
Label 3: Liver
Label 4: Lungs
Label 5: Pancreas
'''

train_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=300, b_min=0.0, b_max=1.0, clip=True,
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
        #Rand3DElasticd(
        #    keys=["image", "label"],
        #    sigma_range=(0, 1),
        #    magnitude_range=(0, 1),
        #    spatial_size=None,
        #    prob=0.5,
        #    rotate_range=(-math.pi / 36, math.pi / 36),  # -15, 15 / -5, 5
        #    shear_range=None,
        #    translate_range=None,
        #    scale_range=None,
        #    mode=("bilinear", "nearest"),
        #    padding_mode="zeros",
        #   as_tensor_output=False
        #),
        #RandGaussianNoised(
        #    keys=["image"],
        #    prob=0.5,
        #   mean=0.0,
        #    std=0.1
         #allow_missing_keys=False
        #),
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
       #RandAdjustContrastd(
       #    keys=["image"],
       #    prob=0.5,
       #    gamma=(0.9, 1.1)
           #allow_missing_keys=False
       #),
        # user can also add other random transforms
        # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=(96, 96, 96),
        #             rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),

       ToTensord(keys=["image", "label"]),
    ]
)
#train_inf_transforms = Compose(
#    [
#        LoadImaged(keys=["image", "label"]),
#        AddChanneld(keys=["image", "label"]),
#        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
#        Orientationd(keys=["image", "label"], axcodes="RAS"),
#        ScaleIntensityRanged(
#            keys=["image"], a_min=-350, a_max=50, b_min=0.0, b_max=1.0, clip=True,
#        ),
#        ToTensord(keys=["image", "label"]),
#    ]
#)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=300, b_min=0.0, b_max=1.0, clip=True,
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
            keys=["image"], a_min=-1000, a_max=300, b_min=0.0, b_max=1.0, clip=True,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)

"""## Check transforms in DataLoader"""

check_ds = Dataset(data=val_files, transform=val_transforms)
check_loader = DataLoader(check_ds, batch_size=1)
check_data = first(check_loader)
image, label = (check_data["image"][0][0], check_data["label"][0][0])
print(f"image shape: {image.shape}, label shape: {label.shape}")
# plot the slice [:, :, 80]
fig = plt.figure("check", (12, 6)) #figure size
plt.subplot(1, 2, 1)
plt.title("image")
plt.imshow(image[:, :, 80], cmap="gray")
plt.subplot(1, 2, 2)
plt.title("label")
plt.imshow(label[:, :, 80])
#plt.show()
#fig.savefig('my_figure.png')


train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=2)
# train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)


#train_inf_ds = CacheDataset(data=train_files, transform=train_inf_transforms, cache_rate=1.0, num_workers=2)
#train_inf_loader = DataLoader(train_inf_ds, batch_size=1, num_workers=2)

val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=2)
# val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_rate=1.0, num_workers=2)
#test_ds = Dataset(data=test_files)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=2)


"""## Create Model, Loss, Optimizer"""

# standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=6, #6 channels, 1 for each organ more background
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=False, sigmoid=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

"""## Makes the Inferences """

model.load_state_dict(torch.load(os.path.join(out_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    #saver = NiftiSaver(output_dir='C:\\Users\\isasi\\Downloads\\Bladder_Segs_Out')
    saver = NiftiSaver(output_dir='//home//imoreira//Segs_Out',
                       output_postfix="seg",
                       output_ext=".nii.gz",
                       mode="nearest",
                       padding_mode="zeros"
                       )
    for test_data in test_loader:
        test_images = test_data["image"].to(device)
        roi_size = (160, 160, 160)
        sw_batch_size = 1
        val_outputs = sliding_window_inference(
            test_images, roi_size, sw_batch_size, model
        )
       # val_outputs = val_outputs.argmax(dim=1, keepdim=True)
        val_outputs = val_outputs.squeeze(dim=0).cpu().clone().numpy()
        val_outputs = val_outputs.astype(np.bool)
        #val_outputs = largest(val_outputs)

        #val_outputs = val_outputs.cpu().clone().numpy()
        #val_outputs = val_outputs.astype(np.bool)


        saver.save_batch(val_outputs, test_data["image_meta_dict"])

