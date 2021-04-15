import glob

import torch
import os
import numpy as np
import nibabel as nib

from surface_distance import metrics, compute_surface_distances, compute_robust_hausdorff, \
    compute_average_surface_distance



'''
ITERATING FORM

data_path = 'path_to_your_data_folder'
_, _, filenames = next(os.walk(data_path))
ct_masks_list = []
[ct_masks_list.append(f) for f in filenames if (('CNS0' in f)]

for mask_file in ct_masks_list:
 (here you put your code to read in the mask file and corresponding ground truth mask file, and then do your calculations)
 
'''



#gdth_path="C:\\Users\\isasi\\Downloads\\Metrics\\Labels\\"
#pred_path='C:\\Users\\isasi\\Downloads\\Metrics\\Pred\\'
#mas_path='C:\\Users\\isasi\\Downloads\\Metrics\\MAS'

gdth_path="/home/imoreira/Metrics/Labels/"
pred_path='/home/imoreira/Metrics/Labels/'

_, _, filenames_gd = next(os.walk(gdth_path))
_, _, filenames_pred = next(os.walk(pred_path))
#_, _, filenames_mas = next(os.walk(mas_path))

gdth_list = []
pred_list = []
#mas_list = []

for f in filenames_gd:
    gdth_list.append(f)

for i in filenames_pred:
    pred_list.append(i)

#for j in filenames_mas:
#    mas_list.append(j)

for f in gdth_list:
    #print(f)
    gdth_data = nib.load(f)
    gdth_data = gdth_data.get_fdata()
    gdth_arr = np.asarray(gdth_data).astype(np.bool)


#mas_path=nib.load('C:\\Users\\isasi\\Downloads\\Metrics\\MAS\\CNS044_bladder_MAS_majority.nii.gz')

#gdth_path=nib.load('/home/imoreira/Metrics/Labels/CNS044_lungs.nii.gz')
#pred_path=nib.load('/home/imoreira/Metrics/Pred/CNS044_ct_seg_lungs.nii.gz')
#mas_path=nib.load('/home/imoreira/Metrics/MAS/CNS044_bladder_MAS_majority.nii.gz')



    #gdth_data = gdth_path.get_fdata()
    #pred_data = pred_path.get_fdata()
    #mas_data = mas_path.get_fdata()

    #gdth_data_arr = np.asarray(gdth_name).astype(np.bool)
    #pred_data_arr = np.asarray(pred_name.astype(np.bool)
    #mas_data_arr = np.asarray(mas_data).astype(np.bool)

    #pred_data_arr = np.squeeze(pred_data_arr)

    #print("pred_shape is:", pred_data_arr.shape)
    print("gdth_shape is:", gdth_arr.shape)

    #Makes sure both arrays have the same size

    if gdth_data_arr.shape != pred_data_arr.shape:

        print("The arrays have different shapes.")

    else:
        # Compute Dice coefficient
        intersection = np.logical_and(gdth_data_arr, pred_data_arr)

        dice_score = (2 * intersection.sum()) / (gdth_data_arr.sum() + pred_data_arr.sum() + 0.001)

        #Nan Condition

        if dice_score == 0:
            dice_score = "Nan"

        else:
            dice_score = dice_score


        #JACCARD INDEX

        intersection = np.logical_and(gdth_data_arr, pred_data_arr)

        union = np.logical_or(gdth_data_arr, pred_data_arr)

        jc = intersection.sum() / float(union.sum())

        #HAUSDORFF DISTANCE

        sx, sy, sz = gdth_path.header.get_zooms()

        surface_distances = compute_surface_distances(gdth_data_arr, pred_data_arr, (sx, sy, sz))

        hd = compute_robust_hausdorff(surface_distances, percent=100)

        #AVERAGE DISTANCE

        ad = compute_average_surface_distance(surface_distances)

        print("The dice score is:", dice_score)
        print("The jaccard index is:", jc)
        print("The surface distance are:", surface_distances)
        print("The hausdorff distance is:", hd, "mm")
        print("The gdth_surface-pred_surface average distance is:", ad[0], "mm", "and the pred_surface-gdth_surface average distance is:", ad[1], "mm")


