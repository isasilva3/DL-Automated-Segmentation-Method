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
# train_labels = sorted(glob.glob(os.path.join(data_dir, "Labels", "*.nii.gz")))
# data_dicts = [
#     {"image": image_name, "label": label_name}
#     for image_name, label_name in zip(train_images, train_labels)
# ]

data_dicts = [
     {"image": image_name}
     for image_name in train_images
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

test_transforms = Compose(
    [
        LoadImaged(keys="image"),
        AddChanneld(keys="image"),
        Spacingd(keys="image", pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
        Orientationd(keys="image", axcodes="RAS"),
        ScaleIntensityRanged(
            keys="image", a_min=-1000, a_max=300, b_min=0.0, b_max=1.0, clip=True,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys="image"),
    ]
)

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
    out_channels=7, #6 channels, 1 for each organ more background
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
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        val_outputs = sliding_window_inference(
            test_images, roi_size, sw_batch_size, model, overlap=0.8
        )
        val_outputs = val_outputs.argmax(dim=1, keepdim=True)
        val_outputs = val_outputs.cpu().clone().numpy()
        val_outputs = val_outputs.astype(np.int)

        saver.save_batch(val_outputs, test_data["image_meta_dict"])

