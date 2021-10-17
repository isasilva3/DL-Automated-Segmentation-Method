"""## Check best model output with the input image and label"""
import glob
import os
import torch
import nibabel as nib

from MONAI.monai.data import NiftiSaver, CacheDataset, DataLoader
from MONAI.monai.inferers import sliding_window_inference
from MONAI.monai.networks.layers import Norm
from MONAI.monai.networks.nets import UNet
from MONAI.monai.transforms import LoadImage, AddChannel, Spacing, Orientation, ScaleIntensityRange, ToTensord, Compose, \
    LoadImaged, AddChanneld, Spacingd, Orientationd, ScaleIntensityRanged

root_dir = "//home//imoreira"
data_dir = os.path.join(root_dir, "Data")
out_dir= os.path.join(data_dir, "Best_Model")

test_images = sorted(glob.glob(os.path.join(data_dir, "Test_Images", "*.nii.gz")))
data_dicts = [
    {"image": image}
    for image in zip(test_images)]

test_transforms = Compose(
    [
        LoadImaged(keys="image"),
        AddChanneld(keys="image"),
        Spacingd(keys="image", pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
        Orientationd(keys="image", axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"], a_min=-1000, a_max=300, b_min=0.0, b_max=1.0, clip=True,
        ),
        #CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image"]),
    ]
)

test_ds = CacheDataset(data=test_images, transform=test_transforms, cache_rate=1.0, num_workers=1)
#test_ds = Dataset(data=test_files)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

out_dir = "//home//imoreira//Data//Best_Model"
#out_dir = "C:\\Users\\isasi\\Downloads\\Bladder_Best_Model"

model = UNet(
    dimensions=3,
    in_channels=1,
    out_channels=6, #6 channels, 1 for each organ more background
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
).to(device)


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
    for i, test_data in enumerate(test_loader):
        test_images = test_data["image"].to(device)
        roi_size = (160, 160, 160)
        sw_batch_size = 1
        val_outputs = sliding_window_inference(
            test_images, roi_size, sw_batch_size, model
        )
        val_outputs = val_outputs.argmax(dim=1, keepdim=True)
        val_outputs = val_outputs.squeeze(dim=0).cpu().clone().numpy()
        #val_outputs = largest(val_outputs)

        #val_outputs = val_outputs.cpu().clone().numpy()
        #val_outputs = val_outputs.astype(np.bool)


        saver.save_batch(val_outputs, test_data["image_meta_dict"])