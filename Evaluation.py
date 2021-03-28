'''

import seg_metrics.seg_metrics as sg
import SimpleITK as sitk

#labels_dir = '/home/imoreira/Segmentations/Pred'

#labels_dicts = [{"image": image_name} for image_name in zip(labels_dir)]
'''

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


gdth_path='/home/imoreira/Metrics/Labels'
pred_path='/home/imoreira/Metrics/Pred'

data_dicts = [
    {"label": label_name, "seg": seg_name}
    for label_name, seg_name in zip(gdth_path, pred_path)
]


metrics = sg.write_metrics(labels=data_dicts[1:],
                           gdth_path=gdth_path,
                           pred_path=pred_path,
                           csv_file='Metrics.csv',
                           metrics=['dice', 'hd'])

dice = metrics['dice']
hd95 = metrics['hd95']