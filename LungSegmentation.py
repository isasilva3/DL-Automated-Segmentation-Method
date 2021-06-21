# -*- coding: utf-8 -*-
"""
# Lungs 3D segmentation with MONAI

This tutorial shows how to integrate MONAI into an existing PyTorch medical DL program.

And easily use below features:
1. Transforms for dictionary format data.
1. Load Nifti image with metadata.
1. Add channel dim to the data if no channel dimension.
1. Scale medical image intensity with expected range.
1. Crop out a batch of balanced images based on positive / negative label ratio.
1. Cache IO and transforms to accelerate training and validation.
1. 3D UNet model, Dice loss function, Mean Dice metric for 3D segmentation task.
1. Sliding window inference method.
1. Deterministic training for reproducibility.

Target: Lungs
Modality: CT
Size: 10 3D volumes (8 Training + 2 Testing)
Source: Catarina

"""

import ants as ants
import numpy
import skimage
from scipy import ndimage
from skimage.viewer.plugins import measure

from MONAI.monai.transforms import Rand3DElastic, RandGaussianNoise, RandScaleIntensity, RandGaussianSmooth, \
    RandAdjustContrast, RandGaussianSmoothd, RandGaussianNoised, RandAdjustContrastd, RandScaleIntensityd, \
    Rand3DElasticd

"""## Setup imports"""

import glob
import os
import shutil
import tempfile
import nibabel as nib
import numpy as np
import ants

import matplotlib.pyplot as plt
import torch

import cc3d

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import CacheDataset, DataLoader, Dataset, write_nifti
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
    AsDiscreteD,
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
from numpy import clip, asarray, math
from skimage import measure
from skimage.measure import label

print_config()

"""## Setup data directory

You can specify a directory with the `MONAI_DATA_DIRECTORY` environment variable.  
This allows you to save results and reuse downloads.  
If not specified a temporary directory will be used.
"""

"""## Download dataset

Downloads and extracts the dataset.  
The dataset comes from http://medicaldecathlon.com/.
"""

md5 = "410d4a301da4e5b2f6f86ec3ddba524e"

root_dir = "//home//imoreira//Data"
#root_dir = "C:\\Users\\isasi\\Downloads"
data_dir = os.path.join(root_dir, "Lungs")
out_dir = os.path.join(root_dir, "Output")

"""## Set MSD Spleen dataset path"""


#test_images = sorted(glob.glob(os.path.join(data_dir, "imagesTs", "*.nii.gz")))
train_images = sorted(glob.glob(os.path.join(data_dir, "imagesTr", "*.nii.gz")))
train_labels = sorted(glob.glob(os.path.join(data_dir, "labelsTr", "*.nii.gz")))
data_dicts = [
    {"image": image_name, "label": label_name}
    for image_name, label_name in zip(train_images, train_labels)
]

#n = len(data_dicts)
#train_files, val_files = data_dicts[:-3], data_dicts[-3:]
#train_files, val_files = data_dicts[:int(n*0.8)], data_dicts[:int(n*0.2)]

val_files, train_files, test_files = data_dicts[0:8], data_dicts[8:40], data_dicts[40:50]

print("validation files", val_files)
print("train files", train_files)
print("test files", test_files)

#5 fold cross validation
#0:32 training and 33:40 validation split the data diferently and compare the results

#test_files = test_dicts


"""## Set deterministic training for reproducibility"""

set_determinism(seed=0)

"""## Setup transforms for training and validation

Here we use several transforms to augment the dataset:
1. `LoadImaged` loads the spleen CT images and labels from NIfTI format files.
1. `AddChanneld` as the original data doesn't have channel dim, add 1 dim to construct "channel first" shape.
1. `Spacingd` adjusts the spacing by `pixdim=(1.5, 1.5, 2.)` based on the affine matrix.
1. `Orientationd` unifies the data orientation based on the affine matrix.
1. `ScaleIntensityRanged` extracts intensity range [-57, 164] and scales to [0, 1].
1. `CropForegroundd` removes all zero borders to focus on the valid body area of the images and labels.
1. `RandCropByPosNegLabeld` randomly crop patch samples from big image based on pos / neg ratio.  
The image centers of negative samples must be in valid body area.
1. `RandAffined` efficiently performs `rotate`, `scale`, `shear`, `translate`, etc. together based on PyTorch affine transform.
1. `ToTensord` converts the numpy array to PyTorch Tensor for further steps.
"""

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
        Rand3DElasticd(
            keys=["image", "label"],
            sigma_range=(0, 1),
            magnitude_range=(0, 1),
            spatial_size=None,
            prob=0.5,
            rotate_range=(-math.pi/36, math.pi/36), #-15, 15 / -5, 5
            shear_range=None,
            translate_range=None,
            scale_range=(0.15, 0.15, 0.15),
            mode=("bilinear", "nearest"),
            padding_mode="zeros",
            as_tensor_output=False
        ),
        #RandGaussianNoised(
        #    keys=["image"],
        #    prob=0.5,
        #    mean=0.0,
        #    std=0.1
            #allow_missing_keys=False
        #),
        #RandScaleIntensityd(
        #    keys=["image"],
        #    factors=0.05, #this is 10%, try 5%
        #    prob=0.5
        #),
        #RandGaussianSmoothd(
         #   keys=["image"],
         #   sigma_x=(0.25, 1.5),
         #   sigma_y=(0.25, 1.5),
         #   sigma_z=(0.25, 1.5),
         #   prob=0.5,
         #   approx='erf'
            #allow_missing_keys=False
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
train_inf_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"]),
        AddChanneld(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=300, b_min=0.0, b_max=1.0, clip=True,
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
            keys=["image"], a_min=-1000, a_max=300, b_min=0.0, b_max=1.0, clip=True,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image", "label"]),
    ]
)
#try without the scale intensity

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
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0) #Shuffle false


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
#Droupout layer, see how the model is define in the library
model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
    #dropout=0.2,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-4)

"""## Execute a typical PyTorch training process"""

epoch_num = 200
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=2)
post_label = AsDiscrete(to_onehot=True, n_classes=2)


for epoch in range(epoch_num):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{epoch_num}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
        )

        #test_im = inputs'`,`,`,',0«.cpu().tonumpy.()
        #matplotlib

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.0
            metric_count = 0
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (160, 160, 160)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = post_pred(val_outputs)

                largest = KeepLargestConnectedComponent(applied_labels=[1])

                val_labels = post_label(val_labels)
                value = compute_meandice(
                    y_pred=val_outputs,
                    y=val_labels,
                    include_background=False,
                )
                metric_count += len(value)
                metric_sum += value.sum().item()
            metric = metric_sum / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(out_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"
            )

print(f"train completed, best_metric: {best_metric:.4f}  at epoch: {best_metric_epoch}")

"""## Plot the loss and metric"""

fig2=plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
y = epoch_loss_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
plt.xlabel("epoch")
plt.plot(x, y)
plt.show()
fig2.savefig('Lungs_Plot.png')


"""## Check best model output with the input image and label"""
"""## Makes the Inferences """


out_dir = "//home//imoreira//Data//Output"
#out_dir = "C:\\Users\\isasi\\Downloads\\Output"
model.load_state_dict(torch.load(os.path.join(out_dir, "best_metric_model.pth")))
model.eval()

with torch.no_grad():
    #saver = NiftiSaver(output_dir='C:\\Users\\isasi\\Downloads\\Segmentations')
    saver = NiftiSaver(output_dir='//home//imoreira//Segmentations',
                       output_postfix="seg_lungs",
                       output_ext=".nii.gz",
                       mode="nearest",
                       padding_mode = "zeros"
                      )

    for i, train_inf_data in enumerate(train_inf_loader):
        train_images = train_inf_data["image"].to(device)
        roi_size = (160, 160, 160)
        sw_batch_size = 4
        val_outputs_1 = sliding_window_inference(
            train_images, roi_size, sw_batch_size, model
        )

        val_outputs_2 = sliding_window_inference(
            train_images, roi_size, sw_batch_size, model
        )

        val_outputs_1 = val_outputs_1.argmax(dim=1, keepdim=True)
        val_outputs_2 = val_outputs_2.argmax(dim=1, keepdim=True)

        #a = val_outputs_1.cpu().detach().numpy()
        #b = val_outputs_2.cpu().detach().numpy()

        first_lung = largest(val_outputs_1)
        second_lung = largest(val_outputs_2 - first_lung)
        #second_largest = largest(second_lung)
        #val_outputs = both_lungs
        #else:
            #both_lungs = largest(val_outputs)



        g = ndimage.sum(first_lung) * 0.10

        if ndimage.sum(second_lung) >= g:
            both_lungs = first_lung + second_lung
            both_lungs = both_lungs.cpu().clone().numpy()
            both_lungs = both_lungs.astype(np.bool)
        else:
            both_lungs = largest(val_outputs_1)
            both_lungs = both_lungs.cpu().clone().numpy()
            both_lungs = both_lungs.astype(np.bool)

        saver.save_batch(both_lungs, train_inf_data["image_meta_dict"])

print("FINISH!!")





