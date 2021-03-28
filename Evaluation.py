import metrics as metrics
import numpy as np
import os
import SimpleITK as sitk
# import nibabel as nib
import pandas as pd
import copy
import PySimpleGUI as gui
import matplotlib.pyplot as plt
import glob
import sys
from myutil.myutil import load_itk, get_gdth_pred_names, one_hot_encode_3d
import seg_metrics.seg_metrics as sg
import nibabel as nib
import spicy
from scipy.spatial.distance import directed_hausdorff

#metrics = sg.write_metrics(labels=labels[1:],
#                           gdth_path=gdth_path,
#                           pred_path=pred_path,
#                           csv_file='Metrics.csv')

#print(metrics)

gdth_path=nib.load('/home/imoreira/Metrics/Labels/CNS044_lungs.nii')
pred_path=nib.load('/home/imoreira/Metrics/Pred/CNS044_lungs.nii')

gdth_data = gdth_path.get_fdata()
pred_data = pred_path.get_fdata()

dgth_data_arr = np.asarray(gdth_data)
pred_data_arr = np.asarray(pred_data)

hd = directed_hausdorff(dgth_data_arr, pred_data_arr)

print(hd)




