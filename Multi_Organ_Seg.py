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
from scipy.interpolate import make_interp_spline, BSpline

print_config()
print("MULTI-ORGAN")

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


train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=0)
# train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

# use batch_size=2 to load images and use RandCropByPosNegLabeld
# to generate 2 x 4 images for network training
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)


#train_inf_ds = CacheDataset(data=train_files, transform=train_inf_transforms, cache_rate=1.0, num_workers=2)
#train_inf_loader = DataLoader(train_inf_ds, batch_size=1, num_workers=2)

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
    out_channels=7, #6 channels, 1 for each organ more background
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), 1e-5)

"""## Execute a typical PyTorch training process"""

epoch_num = 3
val_interval = 2
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()
#metric_values_class = list()
post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=7)
post_label = AsDiscrete(to_onehot=True, n_classes=7)

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
            #metric_sum_class = 0.0
            #metric_count_class = 0
            for val_data in val_loader:
                val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                )
                roi_size = (96, 96, 96)
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                val_outputs = post_pred(val_outputs)
                val_labels = post_label(val_labels)
                #largest = KeepLargestConnectedComponent(applied_labels=[1])
                value = compute_meandice(
                    y_pred=val_outputs,
                    y=val_labels,
                    include_background=False,
                )

                metric_count += len(value[0])
                #metric_count_class += len(value[1])
                metric_sum += value[0].sum().item()
                #metric_sum_class += value[1].sum().item()
            metric = metric_sum / metric_count
            #metric_class = metric_sum_class / metric_count_class
            metric_values.append(metric)
            #metric_values_class.append(metric_class)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(out_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                #f"current epoch: {epoch + 1} current class mean dice: {metric_class:.4f}"
                f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}"

            )

print(f"train completed, best_metric: {best_metric:.4f}  at epoch: {best_metric_epoch}")

"""## Plot the loss and metric"""

fig2=plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Epoch Average Loss")
x = [i + 1 for i in range(len(epoch_loss_values))]
x_new = np.linspace(0, 3, 1) #
y = epoch_loss_values
spl = make_interp_spline(x, y, k=10) #
y_smooth = spl(x_new) #
plt.xlabel("epoch")
plt.plot(x_new, y_smooth)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [val_interval * (i + 1) for i in range(len(metric_values))]
y = metric_values
x_new = np.linspace(0, 3, 1) #
spl = make_interp_spline(x, y, k=10) #
y_smooth = spl(x_new) #
plt.xlabel("epoch")
plt.plot(x_new, y_smooth) #
plt.show()
fig2.savefig('Training_Plot.png')


"""## Check best model output with the input image and label"""
"""## Makes the Inferences """
###
out_dir = "//home//imoreira//Data//Best_Model"
#out_dir = "C:\\Users\\isasi\\Downloads\\Bladder_Best_Model"
model.load_state_dict(torch.load(os.path.join(out_dir, "best_metric_model.pth")))
model.eval()
with torch.no_grad():
    #saver = NiftiSaver(output_dir='C:\\Users\\isasi\\Downloads\\Bladder_Segs_Out')
    saver = NiftiSaver(output_dir='//home//imoreira//Segs_Out',
                    #output_dir='C:\\Users\\isasi\\Downloads\\Segs_Out',
                       output_postfix="seg",
                       output_ext=".nii.gz",
                       mode="nearest",
                       padding_mode="zeros"
                       )
    for i, test_data in enumerate(test_loader):
        test_images = test_data["image"].to(device)
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            test_images, roi_size, sw_batch_size, model, overlap=0.8
        )
        val_outputs = val_outputs.argmax(dim=1, keepdim=True)
        #val_outputs = val_outputs.squeeze(dim=0).cpu().clone().numpy()
        #val_outputs = largest(val_outputs)

        val_outputs = val_outputs.cpu().clone().numpy()
        val_outputs = val_outputs.astype(np.int)

        #val_outputs = torch.argmax(val_outputs, dim=1)
        #val_outputs = val_outputs.squeeze(dim=0).cpu().data.numpy()

        saver.save_batch(val_outputs, test_data["image_meta_dict"])
