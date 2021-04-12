import torch

from MONAI.monai.data import DataLoader
from MONAI.monai.metrics import compute_hausdorff_distance
from MONAI.monai.transforms import ToTensord, Compose, LoadImaged, AddChanneld

import metrics as metrics
import numpy as np
import nibabel as nib
from scipy.spatial.distance import directed_hausdorff, euclidean, cdist
import seg_metrics.seg_metrics as sg

from monai.utils import optional_import


binary_dilation, _ = optional_import("scipy.ndimage.morphology", name="binary_dilation")
directed_hausdorff, _ = optional_import("scipy.spatial.distance", name="directed_hausdorff")

#gdth_path=nib.load('C:\\Users\\isasi\\Downloads\\Metrics\\Labels\\CNS054_liver.nii.gz')
#pred_path=nib.load('C:\\Users\\isasi\\Downloads\\Metrics\\Pred\\CNS054_ct_seg_liver.nii.gz')

gdth_path=nib.load('/home/imoreira/Metrics/Labels/CNS044_lungs.nii.gz')
pred_path=nib.load('/home/imoreira/Metrics/Pred/CNS044_ct_seg_lungs.nii.gz')

gdth_data = gdth_path.get_fdata()
pred_data = pred_path.get_fdata()

gdth_data_arr = np.asarray(gdth_data).astype(np.bool)
pred_data_arr = np.asarray(pred_data).astype(np.bool)

'''
Compute the dice score between two input images or volumes. Note that we use a smoothing factor of 1.
:param im1: Image 1
:param im2: Image 2
:return: Dice score
'''

#pred_data_arr = np.rollaxis(pred_data_arr,3,1).reshape(1,-1,1)

if gdth_data_arr.shape != pred_data_arr.shape:

    pred_data_arr.shape = gdth_data_arr.shape

    print("pred_shape is:", pred_data_arr.shape)
    print("gdth_shape is:", gdth_data_arr.shape)

'''
# Compute Dice coefficient
intersection = np.logical_and(gdth_data_arr, pred_data_arr)

dice_score = (2 * intersection.sum() + 1) / (gdth_data_arr.sum() + pred_data_arr.sum() + 1)


#JACCARD INDEX

intersection = np.logical_and(gdth_data_arr, pred_data_arr)

union = np.logical_or(gdth_data_arr, pred_data_arr)

jc = intersection.sum() / float(union.sum())

#HAUSDORFF DISTANCE
'''
if not (np.any(gdth_data_arr) and np.any(pred_data_arr)):
    raise ValueError(f"Labelfields should have at least 1 voxel containing the desired labelfield")

# Do binary dilation and use XOR to get edges
edges_1 = binary_dilation(gdth_data_arr) ^ gdth_data_arr
edges_2 = binary_dilation(pred_data_arr) ^ pred_data_arr

# Extract coordinates of these edge points
coords_1 = np.argwhere(edges_1)
coords_2 = np.argwhere(edges_2)

# Get (potentially directed) Hausdorff distance
#if directed:
#hd = float(directed_hausdorff(coords_1, coords_2)[0])

#else:
#    hd = float(max(directed_hausdorff(coords_1, coords_2)[0], directed_hausdorff(coords_2, coords_1)[0]))

min_euc_list_u = []
min_euc_list_v = []

for i in range(coords_1.shape[0]):
    euc_list = []
    for j in range(coords_2.shape[0]):
        euc = euclidean(coords_1[i, :], coords_2[j, :])
        euc_list.append(euc)
    min_euc_list_u.append(min(euc_list))

for i in range(coords_2.shape[0]):
    euc_list = []
    for j in range(coords_1.shape[0]):
        euc = euclidean(coords_2[i, :], coords_1[j, :])
        euc_list.append(euc)
    min_euc_list_v.append(min(euc_list))

hd = max(np.array(min_euc_list_u).mean(), np.array(min_euc_list_v).mean())

#AVARAGE DISTANCE

#ad = cdist(gdth_data_arr, pred_data_arr)


#print("The dice score is:", dice_score)
#print("The jaccard index is:", jc)
print("The hausdorff distance is:", hd)
#print("The average distance is:", ad)


