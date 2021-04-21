import glob
import os

import numpy as np
import nibabel as nib
import xlsxwriter

from surface_distance import metrics, compute_surface_distances, compute_robust_hausdorff, \
    compute_average_surface_distance


#gdth_path="C:\\Users\\isasi\\Downloads\\Metrics\\Labels\\Test\\"
#gdth_path_val="C:\\Users\\isasi\\Downloads\\Metrics\\Labels\\Val\\"
#gdth_path_train="C:\\Users\\isasi\\Downloads\\Metrics\\Labels\\Train\\"
#pred_path='C:\\Users\\isasi\\Downloads\\Metrics\\Pred\\Test\\'
#mas_path='C:\\Users\\isasi\\Downloads\\Metrics\\MAS\\'
#val_path ='C:\\Users\\isasi\\Downloads\\Metrics\\Pred\\Val\\'
#train_path ='C:\\Users\\isasi\\Downloads\\Metrics\\Pred\\Train\\'

gdth_path="/home/imoreira/Metrics/Labels/Test/"
gdth_path_val="/home/imoreira/Metrics/Labels/Val/"
gdth_path_train="/home/imoreira/Metrics/Labels/Train/"
pred_path='/home/imoreira/Metrics/Pred/Test/'
mas_path='/home/imoreira/Metrics/MAS/'
val_path='/home/imoreira/Metrics/Pred/Val/'
train_path='/home/imoreira/Metrics/Pred/Train/'

_, _, filenames_gd = next(os.walk(gdth_path))
_, _, filenames_gd_v = next(os.walk(gdth_path_val))
_, _, filenames_gd_t = next(os.walk(gdth_path_train))
_, _, filenames_pred = next(os.walk(pred_path))
_, _, filenames_mas = next(os.walk(mas_path))
_, _, filenames_val = next(os.walk(val_path))
_, _, filenames_train = next(os.walk(train_path))



gdth_list = []
pred_list = []
mas_list = []
gdth_val_list = []
gdth_train_list = []

gdth_arr_list = []
pred_arr_list = []
mas_arr_list =[]
gdth_images = []
gdth_images_val=[]
gdth_images_train=[]
gdth_val_arr_list = []
gdth_train_arr_list = []

dice = []
jc_i = []
hausdorff =[]
average = []

dice_lungs = []
jc_i_lungs = []
hausdorff_lungs =[]

dice_liver = []
jc_i_liver = []
hausdorff_liver =[]

dice_bladder = []
jc_i_bladder = []
hausdorff_bladder =[]

dice_mas = []
jc_i_mas = []
hausdorff_mas =[]
average_mas = []

dice_lungs_mas = []
jc_i_lungs_mas = []
hausdorff_lungs_mas =[]

dice_liver_mas = []
jc_i_liver_mas = []
hausdorff_liver_mas =[]

dice_bladder_mas = []
jc_i_bladder_mas = []
hausdorff_bladder_mas =[]

val_list = []
train_list = []

val_arr_l = []
train_arr_l = []

dice_val = []
jc_i_val = []
hausdorff_val =[]

dice_train = []
jc_i_train = []
hausdorff_train =[]

dice_val_lungs =[]
jc_i_val_lungs=[]
hausdorff_val_lungs=[]

dice_val_liver =[]
jc_i_val_liver=[]
hausdorff_val_liver=[]

dice_val_bladder=[]
jc_i_val_bladder=[]
hausdorff_val_bladder=[]

dice_train_lungs =[]
jc_i_train_lungs=[]
hausdorff_train_lungs=[]

dice_train_liver =[]
jc_i_train_liver =[]
hausdorff_train_liver =[]

dice_train_bladder=[]
jc_i_train_bladder=[]
hausdorff_train_bladder=[]

for f in filenames_gd:
    gdth_list.append(gdth_path + f)

for f in filenames_gd_v:
    gdth_val_list.append(gdth_path_val + f)

for f in filenames_gd_t:
    gdth_train_list.append(gdth_path_train + f)

for i in filenames_pred:
    pred_list.append(pred_path + i)

for j in filenames_mas:
    mas_list.append(mas_path + j)

for i in filenames_val:
    val_list.append(val_path + i)

for i in filenames_train:
    train_list.append(train_path + i)

print(gdth_list)
print(pred_list)
'''
for f in gdth_list:
    gdth_data = nib.load(f)
    gdth_data_np = gdth_data.get_fdata()
    #gdth_data = gdth_data.header
    gdth_images.append(gdth_data)

    gdth_arr = np.asarray(gdth_data_np).astype(np.bool)
    gdth_arr_list.append(gdth_arr)

for f in gdth_val_list:
    gdth_val_data = nib.load(f)
    gdth_val_data_np = gdth_val_data.get_fdata()
    #gdth_data = gdth_data.header
    gdth_images_val.append(gdth_val_data)

    gdth_val_arr = np.asarray(gdth_val_data_np).astype(np.bool)
    gdth_val_arr_list.append(gdth_val_arr)
'''
for f in gdth_train_list:
    gdth_train_data = nib.load(f)
    gdth_train_data_np = gdth_train_data.get_fdata()
    #gdth_data = gdth_data.header
    gdth_images_train.append(gdth_train_data)

    gdth_train_arr = np.asarray(gdth_train_data_np).astype(np.bool)
    gdth_train_arr_list.append(gdth_train_arr)

#print(gdth_images)
'''
for i in pred_list:
    pred_data = nib.load(i)
    pred_data = pred_data.get_fdata()
    pred_arr = np.asarray(pred_data).astype(np.bool)
    pred_arr = np.squeeze(pred_arr)
    pred_arr_list.append(pred_arr)

for i in mas_list:
    mas_data = nib.load(i)
    mas_data = mas_data.get_fdata()
    mas_arr = np.asarray(mas_data).astype(np.bool)
    mas_arr = np.squeeze(mas_arr)
    mas_arr_list.append(mas_arr)

for i in val_list:
    val_data = nib.load(i)
    val_data = val_data.get_fdata()
    val_arr = np.asarray(val_data).astype(np.bool)
    val_arr = np.squeeze(val_arr)
    val_arr_l.append(val_arr)
'''
for i in train_list:
    train_data = nib.load(i)
    train_data = train_data.get_fdata()
    train_arr = np.asarray(train_data).astype(np.bool)
    train_arr = np.squeeze(train_arr)
    train_arr_l.append(train_arr)

#print(gdth_arr_list)
#print(pred_arr_list)

g = 0
m = 0
n = 0

'''

for m in range(len(gdth_arr_list)):

    for n in range(len(pred_arr_list)):

        # Makes sure both arrays have the same size
        if gdth_arr_list[m].shape != pred_arr_list[n].shape:

            print("The arrays have different shapes.")

        else:
            # Compute Dice coefficient
            intersection = np.logical_and(gdth_arr_list[m], pred_arr_list[n])

            dice_score = (2 * intersection.sum()) / (gdth_arr_list[m].sum() + pred_arr_list[n].sum())

            dice_score = round(dice_score, 3)

            #Nan Condition

            if dice_score == 0:
                dice_score = "Nan"

            else:
                dice_score = dice_score
                dice.append(dice_score)


            #JACCARD INDEX

            intersection = np.logical_and(gdth_arr_list[m], pred_arr_list[n])

            union = np.logical_or(gdth_arr_list[m], pred_arr_list[n])

            jc = intersection.sum() / float(union.sum())

            jc = round(jc, 3)

            jc_i.append(jc)

            #HAUSDORFF DISTANCE

            #gdth_files = (gdth_path + f)

            for g in gdth_images:
                sx, sy, sz = gdth_data.header.get_zooms()

            #CONFIRMAR E FAZER COM QUE ESTEJA A IR BUSCAR O HEADER INFO DAS MESMAS IMAGENS QUE ESTAMOS A PROCESSAR DE CADA VEZ


            surface_distances = compute_surface_distances(gdth_arr_list[m], pred_arr_list[n], (sx, sy, sz))

            hd = compute_robust_hausdorff(surface_distances, percent=100)

            hd = round(hd, 3)

            hausdorff.append(hd)

            #AVERAGE DISTANCE

            ad = compute_average_surface_distance(surface_distances)
            average.append(ad)

            #print("For the arr_gdth", m, gdth_arr_list[m].shape, "and the arr_pred", n, pred_arr_list[n].shape)
            #print("The dice score is:", dice_score)
            #print("The jaccard index is:", jc)
            # print("The surface distance are:", surface_distances)
            #print("The hausdorff distance is:", hd, "mm")
            #print("The gdth_surface-pred_surface average distance is:", ad[0], "mm",
            #      "and the pred_surface-gdth_surface average distance is:", ad[1], "mm")

            m = m + 1
            n = n + 1

            if m == len(gdth_arr_list) + 1:
                break
    break

for m in range(len(gdth_arr_list)):

    for n in range(len(mas_arr_list)):

        # Makes sure both arrays have the same size
        if gdth_arr_list[m].shape != mas_arr_list[n].shape:

            print("The arrays have different shapes.")

        else:
            # Compute Dice coefficient
            intersection = np.logical_and(gdth_arr_list[m], mas_arr_list[n])

            dice_score = (2 * intersection.sum()) / (gdth_arr_list[m].sum() + mas_arr_list[n].sum())

            dice_score = round(dice_score, 3)

            #Nan Condition

            if dice_score == 0:
                dice_score = "Nan"

            else:
                dice_score = dice_score
                dice_mas.append(dice_score)


            #JACCARD INDEX

            intersection = np.logical_and(gdth_arr_list[m], mas_arr_list[n])

            union = np.logical_or(gdth_arr_list[m], mas_arr_list[n])

            jc = intersection.sum() / float(union.sum())

            jc = round(jc, 3)

            jc_i_mas.append(jc)

            #HAUSDORFF DISTANCE

            #gdth_files = (gdth_path + f)

            for g in gdth_images:
                sx, sy, sz = gdth_data.header.get_zooms()

            #CONFIRMAR E FAZER COM QUE ESTEJA A IR BUSCAR O HEADER INFO DAS MESMAS IMAGENS QUE ESTAMOS A PROCESSAR DE CADA VEZ


            surface_distances = compute_surface_distances(gdth_arr_list[m], mas_arr_list[n], (sx, sy, sz))

            hd = compute_robust_hausdorff(surface_distances, percent=100)

            hd = round(hd, 3)

            hausdorff_mas.append(hd)

            #AVERAGE DISTANCE

            ad = compute_average_surface_distance(surface_distances)
            average_mas.append(ad)

            #print("For the arr_gdth", m, gdth_arr_list[m].shape, "and the arr_pred", n, pred_arr_list[n].shape)
            #print("The dice score is:", dice_score)
            #print("The jaccard index is:", jc)
            # print("The surface distance are:", surface_distances)
            #print("The hausdorff distance is:", hd, "mm")
            #print("The gdth_surface-pred_surface average distance is:", ad[0], "mm",
            #      "and the pred_surface-gdth_surface average distance is:", ad[1], "mm")

            m = m + 1
            n = n + 1

            if m == len(gdth_arr_list) + 1:
                break

    break

for m in range(len(gdth_val_arr_list)):

    for n in range(len(val_arr_l)):

        # Makes sure both arrays have the same size
        if gdth_val_arr_list[m].shape != val_arr_l[n].shape:

            print("The arrays have different shapes.")

        else:
            # Compute Dice coefficient
            intersection = np.logical_and(gdth_val_arr_list[m], val_arr_l[n])

            dice_score = (2 * intersection.sum()) / (gdth_val_arr_list[m].sum() + val_arr_l[n].sum())

            dice_score = round(dice_score, 3)

            #Nan Condition

            if dice_score == 0:
                dice_score = "Nan"

            else:
                dice_score = dice_score
                dice_val.append(dice_score)


            #JACCARD INDEX

            intersection = np.logical_and(gdth_val_arr_list[m], val_arr_l[n])

            union = np.logical_or(gdth_val_arr_list[m], val_arr_l[n])

            jc = intersection.sum() / float(union.sum())

            jc = round(jc, 3)

            jc_i_val.append(jc)

            #HAUSDORFF DISTANCE

            #gdth_files = (gdth_path + f)

            for g in gdth_images_val:
                sx, sy, sz = gdth_val_data.header.get_zooms()

            #CONFIRMAR E FAZER COM QUE ESTEJA A IR BUSCAR O HEADER INFO DAS MESMAS IMAGENS QUE ESTAMOS A PROCESSAR DE CADA VEZ


            surface_distances = compute_surface_distances(gdth_val_arr_list[m], val_arr_l[n], (sx, sy, sz))

            hd = compute_robust_hausdorff(surface_distances, percent=100)

            hd = round(hd, 3)

            hausdorff_val.append(hd)

            #AVERAGE DISTANCE

            #ad = compute_average_surface_distance(surface_distances)
            #average_mas.append(ad)

            #print("For the arr_gdth", m, gdth_arr_list[m].shape, "and the arr_pred", n, pred_arr_list[n].shape)
            #print("The dice score is:", dice_score)
            #print("The jaccard index is:", jc)
            # print("The surface distance are:", surface_distances)
            #print("The hausdorff distance is:", hd, "mm")
            #print("The gdth_surface-pred_surface average distance is:", ad[0], "mm",
            #      "and the pred_surface-gdth_surface average distance is:", ad[1], "mm")

            m = m + 1
            n = n + 1

            if m == len(gdth_val_arr_list) + 1:
                break

    break
'''
for m in range(len(gdth_train_arr_list)):

    for n in range(len(train_arr_l)):

        # Makes sure both arrays have the same size
        if train_arr_l[m].shape != train_arr_l[n].shape:

            print("The arrays have different shapes.")

        else:
            # Compute Dice coefficient
            intersection = np.logical_and(gdth_train_arr_list[m], train_arr_l[n])

            dice_score = (2 * intersection.sum()) / (gdth_train_arr_list[m].sum() + train_arr_l[n].sum())

            dice_score = round(dice_score, 3)

            #Nan Condition

            if dice_score == 0:
                dice_score = "Nan"

            else:
                dice_score = dice_score
                dice_train.append(dice_score)


            #JACCARD INDEX

            intersection = np.logical_and(gdth_train_arr_list[m], train_arr_l[n])

            union = np.logical_or(gdth_train_arr_list[m], train_arr_l[n])

            jc = intersection.sum() / float(union.sum())

            jc = round(jc, 3)

            jc_i_train.append(jc)

            #HAUSDORFF DISTANCE

            #gdth_files = (gdth_path + f)

            for g in gdth_images_train:
                sx, sy, sz = gdth_train_data.header.get_zooms()

            #CONFIRMAR E FAZER COM QUE ESTEJA A IR BUSCAR O HEADER INFO DAS MESMAS IMAGENS QUE ESTAMOS A PROCESSAR DE CADA VEZ


            surface_distances = compute_surface_distances(gdth_train_arr_list[m], train_arr_l[n], (sx, sy, sz))

            hd = compute_robust_hausdorff(surface_distances, percent=100)

            hd = round(hd, 3)

            hausdorff_train.append(hd)

            #AVERAGE DISTANCE

            #ad = compute_average_surface_distance(surface_distances)
            #average_mas.append(ad)

            #print("For the arr_gdth", m, gdth_arr_list[m].shape, "and the arr_pred", n, pred_arr_list[n].shape)
            #print("The dice score is:", dice_score)
            #print("The jaccard index is:", jc)
            # print("The surface distance are:", surface_distances)
            #print("The hausdorff distance is:", hd, "mm")
            #print("The gdth_surface-pred_surface average distance is:", ad[0], "mm",
            #      "and the pred_surface-gdth_surface average distance is:", ad[1], "mm")

            m = m + 1
            n = n + 1

            if m == len(gdth_train_arr_list) + 1:
                break

    break

i=0

for i in range(len(dice)):
    if (i % 3 == 0):
        dice_bladder.append(dice[i])

dice_liver.append(dice[1])
dice_liver.append(dice[4])
dice_liver.append(dice[7])
dice_liver.append(dice[10])
dice_liver.append(dice[13])
dice_liver.append(dice[16])
dice_liver.append(dice[19])
dice_liver.append(dice[22])
dice_liver.append(dice[25])
dice_liver.append(dice[28])

dice_lungs.append(dice[2])
dice_lungs.append(dice[5])
dice_lungs.append(dice[8])
dice_lungs.append(dice[11])
dice_lungs.append(dice[14])
dice_lungs.append(dice[17])
dice_lungs.append(dice[20])
dice_lungs.append(dice[23])
dice_lungs.append(dice[26])
dice_lungs.append(dice[29])

for i in range(len(jc_i)):
    if (i % 3 == 0):
        jc_i_bladder.append(jc_i[i])

jc_i_liver.append(jc_i[1])
jc_i_liver.append(jc_i[4])
jc_i_liver.append(jc_i[7])
jc_i_liver.append(jc_i[10])
jc_i_liver.append(jc_i[13])
jc_i_liver.append(jc_i[16])
jc_i_liver.append(jc_i[19])
jc_i_liver.append(jc_i[22])
jc_i_liver.append(jc_i[25])
jc_i_liver.append(jc_i[28])

jc_i_lungs.append(jc_i[2])
jc_i_lungs.append(jc_i[5])
jc_i_lungs.append(jc_i[8])
jc_i_lungs.append(jc_i[11])
jc_i_lungs.append(jc_i[14])
jc_i_lungs.append(jc_i[17])
jc_i_lungs.append(jc_i[20])
jc_i_lungs.append(jc_i[23])
jc_i_lungs.append(jc_i[26])
jc_i_lungs.append(jc_i[29])

for i in range(len(hausdorff)):
    if (i % 3 == 0):
        hausdorff_bladder.append(hausdorff[i])

hausdorff_liver.append(hausdorff[1])
hausdorff_liver.append(hausdorff[4])
hausdorff_liver.append(hausdorff[7])
hausdorff_liver.append(hausdorff[10])
hausdorff_liver.append(hausdorff[13])
hausdorff_liver.append(hausdorff[16])
hausdorff_liver.append(hausdorff[19])
hausdorff_liver.append(hausdorff[22])
hausdorff_liver.append(hausdorff[25])
hausdorff_liver.append(hausdorff[28])

hausdorff_lungs.append(hausdorff[2])
hausdorff_lungs.append(hausdorff[5])
hausdorff_lungs.append(hausdorff[8])
hausdorff_lungs.append(hausdorff[11])
hausdorff_lungs.append(hausdorff[14])
hausdorff_lungs.append(hausdorff[17])
hausdorff_lungs.append(hausdorff[20])
hausdorff_lungs.append(hausdorff[23])
hausdorff_lungs.append(hausdorff[26])
hausdorff_lungs.append(hausdorff[29])

for i in range(len(dice_mas)):
    if (i % 3 == 0):
        dice_bladder_mas.append(dice_mas[i])

dice_liver_mas.append(dice_mas[1])
dice_liver_mas.append(dice_mas[4])
dice_liver_mas.append(dice_mas[7])
dice_liver_mas.append(dice_mas[10])
dice_liver_mas.append(dice_mas[13])
dice_liver_mas.append(dice_mas[16])
dice_liver_mas.append(dice_mas[19])
dice_liver_mas.append(dice_mas[22])
dice_liver_mas.append(dice_mas[25])
dice_liver_mas.append(dice_mas[28])

dice_lungs_mas.append(dice_mas[2])
dice_lungs_mas.append(dice_mas[5])
dice_lungs_mas.append(dice_mas[8])
dice_lungs_mas.append(dice_mas[11])
dice_lungs_mas.append(dice_mas[14])
dice_lungs_mas.append(dice_mas[17])
dice_lungs_mas.append(dice_mas[20])
dice_lungs_mas.append(dice_mas[23])
dice_lungs_mas.append(dice_mas[26])
dice_lungs_mas.append(dice_mas[29])

for i in range(len(jc_i_mas)):
    if (i % 3 == 0):
        jc_i_bladder_mas.append(jc_i_mas[i])

jc_i_liver_mas.append(jc_i_mas[1])
jc_i_liver_mas.append(jc_i_mas[4])
jc_i_liver_mas.append(jc_i_mas[7])
jc_i_liver_mas.append(jc_i_mas[10])
jc_i_liver_mas.append(jc_i_mas[13])
jc_i_liver_mas.append(jc_i_mas[16])
jc_i_liver_mas.append(jc_i_mas[19])
jc_i_liver_mas.append(jc_i_mas[22])
jc_i_liver_mas.append(jc_i_mas[25])
jc_i_liver_mas.append(jc_i_mas[28])

jc_i_lungs_mas.append(jc_i_mas[2])
jc_i_lungs_mas.append(jc_i_mas[5])
jc_i_lungs_mas.append(jc_i_mas[8])
jc_i_lungs_mas.append(jc_i_mas[11])
jc_i_lungs_mas.append(jc_i_mas[14])
jc_i_lungs_mas.append(jc_i_mas[17])
jc_i_lungs_mas.append(jc_i_mas[20])
jc_i_lungs_mas.append(jc_i_mas[23])
jc_i_lungs_mas.append(jc_i_mas[26])
jc_i_lungs_mas.append(jc_i_mas[29])

for i in range(len(hausdorff_mas)):
    if (i % 3 == 0):
        hausdorff_bladder_mas.append(hausdorff_mas[i])

hausdorff_liver_mas.append(hausdorff_mas[1])
hausdorff_liver_mas.append(hausdorff_mas[4])
hausdorff_liver_mas.append(hausdorff_mas[7])
hausdorff_liver_mas.append(hausdorff_mas[10])
hausdorff_liver_mas.append(hausdorff_mas[13])
hausdorff_liver_mas.append(hausdorff_mas[16])
hausdorff_liver_mas.append(hausdorff_mas[19])
hausdorff_liver_mas.append(hausdorff_mas[22])
hausdorff_liver_mas.append(hausdorff_mas[25])
hausdorff_liver_mas.append(hausdorff_mas[28])

hausdorff_lungs_mas.append(hausdorff_mas[2])
hausdorff_lungs_mas.append(hausdorff_mas[5])
hausdorff_lungs_mas.append(hausdorff_mas[8])
hausdorff_lungs_mas.append(hausdorff_mas[11])
hausdorff_lungs_mas.append(hausdorff_mas[14])
hausdorff_lungs_mas.append(hausdorff_mas[17])
hausdorff_lungs_mas.append(hausdorff_mas[20])
hausdorff_lungs_mas.append(hausdorff_mas[23])
hausdorff_lungs_mas.append(hausdorff_mas[26])
hausdorff_lungs_mas.append(hausdorff_mas[29])

for i in range(len(dice_val)):
    if (i % 3 == 0):
        dice_val_bladder.append(dice_val[i])

dice_val_liver.append(dice_val[1])
dice_val_liver.append(dice_val[4])
dice_val_liver.append(dice_val[7])
dice_val_liver.append(dice_val[10])
dice_val_liver.append(dice_val[13])
dice_val_liver.append(dice_val[16])
dice_val_liver.append(dice_val[19])
dice_val_liver.append(dice_val[22])

dice_val_lungs.append(dice_val[2])
dice_val_lungs.append(dice_val[5])
dice_val_lungs.append(dice_val[8])
dice_val_lungs.append(dice_val[11])
dice_val_lungs.append(dice_val[14])
dice_val_lungs.append(dice_val[17])
dice_val_lungs.append(dice_val[20])
dice_val_lungs.append(dice_val[23])

for i in range(len(dice_train)):
    if (i % 3 == 0):
        dice_train_bladder.append(dice_train[i])

dice_train_liver.append(dice_train[1])
dice_train_liver.append(dice_train[4])
dice_train_liver.append(dice_train[7])
dice_train_liver.append(dice_train[10])
dice_train_liver.append(dice_train[13])
dice_train_liver.append(dice_train[16])
dice_train_liver.append(dice_train[19])
dice_train_liver.append(dice_train[22])
dice_train_liver.append(dice_train[25])
dice_train_liver.append(dice_train[28])
dice_train_liver.append(dice_train[31])
dice_train_liver.append(dice_train[34])
dice_train_liver.append(dice_train[37])
dice_train_liver.append(dice_train[40])
dice_train_liver.append(dice_train[43])
dice_train_liver.append(dice_train[46])
dice_train_liver.append(dice_train[49])
dice_train_liver.append(dice_train[52])
dice_train_liver.append(dice_train[55])
dice_train_liver.append(dice_train[58])
dice_train_liver.append(dice_train[61])
dice_train_liver.append(dice_train[64])
dice_train_liver.append(dice_train[67])
dice_train_liver.append(dice_train[70])
dice_train_liver.append(dice_train[73])
dice_train_liver.append(dice_train[76])
dice_train_liver.append(dice_train[79])
dice_train_liver.append(dice_train[82])
dice_train_liver.append(dice_train[85])
dice_train_liver.append(dice_train[88])
dice_train_liver.append(dice_train[91])
dice_train_liver.append(dice_train[94])
dice_train_liver.append(dice_train[97])


dice_train_lungs.append(dice_train[2])
dice_train_lungs.append(dice_train[5])
dice_train_lungs.append(dice_train[8])
dice_train_lungs.append(dice_train[11])
dice_train_lungs.append(dice_train[14])
dice_train_lungs.append(dice_train[17])
dice_train_lungs.append(dice_train[20])
dice_train_lungs.append(dice_train[23])
dice_train_lungs.append(dice_train[26])
dice_train_lungs.append(dice_train[29])
dice_train_lungs.append(dice_train[32])
dice_train_lungs.append(dice_train[35])
dice_train_lungs.append(dice_train[38])
dice_train_lungs.append(dice_train[41])
dice_train_lungs.append(dice_train[44])
dice_train_lungs.append(dice_train[47])
dice_train_lungs.append(dice_train[50])
dice_train_lungs.append(dice_train[53])
dice_train_lungs.append(dice_train[56])
dice_train_lungs.append(dice_train[59])
dice_train_lungs.append(dice_train[62])
dice_train_lungs.append(dice_train[65])
dice_train_lungs.append(dice_train[68])
dice_train_lungs.append(dice_train[71])
dice_train_lungs.append(dice_train[74])
dice_train_lungs.append(dice_train[77])
dice_train_lungs.append(dice_train[80])
dice_train_lungs.append(dice_train[83])
dice_train_lungs.append(dice_train[86])
dice_train_lungs.append(dice_train[89])
dice_train_lungs.append(dice_train[92])
dice_train_lungs.append(dice_train[95])
dice_train_lungs.append(dice_train[98])

for i in range(len(jc_i_val)):
    if (i % 3 == 0):
        jc_i_val_bladder.append(jc_i_val[i])

jc_i_val_liver.append(jc_i_val[1])
jc_i_val_liver.append(jc_i_val[4])
jc_i_val_liver.append(jc_i_val[7])
jc_i_val_liver.append(jc_i_val[10])
jc_i_val_liver.append(jc_i_val[13])
jc_i_val_liver.append(jc_i_val[16])
jc_i_val_liver.append(jc_i_val[19])
jc_i_val_liver.append(jc_i_val[22])

jc_i_val_lungs.append(jc_i_val[2])
jc_i_val_lungs.append(jc_i_val[5])
jc_i_val_lungs.append(jc_i_val[8])
jc_i_val_lungs.append(jc_i_val[11])
jc_i_val_lungs.append(jc_i_val[14])
jc_i_val_lungs.append(jc_i_val[17])
jc_i_val_lungs.append(jc_i_val[20])
jc_i_val_lungs.append(jc_i_val[23])

for i in range(len(jc_i_train)):
    if (i % 3 == 0):
        jc_i_train_bladder.append(jc_i_train[i])

jc_i_train_liver.append(jc_i_train[1])
jc_i_train_liver.append(jc_i_train[4])
jc_i_train_liver.append(jc_i_train[7])
jc_i_train_liver.append(jc_i_train[10])
jc_i_train_liver.append(jc_i_train[13])
jc_i_train_liver.append(jc_i_train[16])
jc_i_train_liver.append(jc_i_train[19])
jc_i_train_liver.append(jc_i_train[22])
jc_i_train_liver.append(jc_i_train[25])
jc_i_train_liver.append(jc_i_train[28])
jc_i_train_liver.append(jc_i_train[31])
jc_i_train_liver.append(jc_i_train[34])
jc_i_train_liver.append(jc_i_train[37])
jc_i_train_liver.append(jc_i_train[40])
jc_i_train_liver.append(jc_i_train[43])
jc_i_train_liver.append(jc_i_train[46])
jc_i_train_liver.append(jc_i_train[49])
jc_i_train_liver.append(jc_i_train[52])
jc_i_train_liver.append(jc_i_train[55])
jc_i_train_liver.append(jc_i_train[58])
jc_i_train_liver.append(jc_i_train[61])
jc_i_train_liver.append(jc_i_train[64])
jc_i_train_liver.append(jc_i_train[67])
jc_i_train_liver.append(jc_i_train[70])
jc_i_train_liver.append(jc_i_train[73])
jc_i_train_liver.append(jc_i_train[76])
jc_i_train_liver.append(jc_i_train[79])
jc_i_train_liver.append(jc_i_train[82])
jc_i_train_liver.append(jc_i_train[85])
jc_i_train_liver.append(jc_i_train[88])
jc_i_train_liver.append(jc_i_train[91])
jc_i_train_liver.append(jc_i_train[94])
jc_i_train_liver.append(jc_i_train[97])



jc_i_train_lungs.append(jc_i_train[2])
jc_i_train_lungs.append(jc_i_train[5])
jc_i_train_lungs.append(jc_i_train[8])
jc_i_train_lungs.append(jc_i_train[11])
jc_i_train_lungs.append(jc_i_train[14])
jc_i_train_lungs.append(jc_i_train[17])
jc_i_train_lungs.append(jc_i_train[20])
jc_i_train_lungs.append(jc_i_train[23])
jc_i_train_lungs.append(jc_i_train[26])
jc_i_train_lungs.append(jc_i_train[29])
jc_i_train_lungs.append(jc_i_train[32])
jc_i_train_lungs.append(jc_i_train[35])
jc_i_train_lungs.append(jc_i_train[38])
jc_i_train_lungs.append(jc_i_train[41])
jc_i_train_lungs.append(jc_i_train[44])
jc_i_train_lungs.append(jc_i_train[47])
jc_i_train_lungs.append(jc_i_train[50])
jc_i_train_lungs.append(jc_i_train[53])
jc_i_train_lungs.append(jc_i_train[56])
jc_i_train_lungs.append(jc_i_train[59])
jc_i_train_lungs.append(jc_i_train[62])
jc_i_train_lungs.append(jc_i_train[65])
jc_i_train_lungs.append(jc_i_train[68])
jc_i_train_lungs.append(jc_i_train[71])
jc_i_train_lungs.append(jc_i_train[74])
jc_i_train_lungs.append(jc_i_train[77])
jc_i_train_lungs.append(jc_i_train[80])
jc_i_train_lungs.append(jc_i_train[83])
jc_i_train_lungs.append(jc_i_train[86])
jc_i_train_lungs.append(jc_i_train[89])
jc_i_train_lungs.append(jc_i_train[92])
jc_i_train_lungs.append(jc_i_train[95])
jc_i_train_lungs.append(jc_i_train[98])

for i in range(len(hausdorff_train)):
    if (i % 3 == 0):
        hausdorff_train_bladder.append(hausdorff_train[i])

hausdorff_train_liver.append(hausdorff_train[1])
hausdorff_train_liver.append(hausdorff_train[4])
hausdorff_train_liver.append(hausdorff_train[7])
hausdorff_train_liver.append(hausdorff_train[10])
hausdorff_train_liver.append(hausdorff_train[13])
hausdorff_train_liver.append(hausdorff_train[16])
hausdorff_train_liver.append(hausdorff_train[19])
hausdorff_train_liver.append(hausdorff_train[22])
hausdorff_train_liver.append(hausdorff_train[25])
hausdorff_train_liver.append(hausdorff_train[28])
hausdorff_train_liver.append(hausdorff_train[31])
hausdorff_train_liver.append(hausdorff_train[34])
hausdorff_train_liver.append(hausdorff_train[37])
hausdorff_train_liver.append(hausdorff_train[40])
hausdorff_train_liver.append(hausdorff_train[43])
hausdorff_train_liver.append(hausdorff_train[46])
hausdorff_train_liver.append(hausdorff_train[49])
hausdorff_train_liver.append(hausdorff_train[52])
hausdorff_train_liver.append(hausdorff_train[55])
hausdorff_train_liver.append(hausdorff_train[58])
hausdorff_train_liver.append(hausdorff_train[61])
hausdorff_train_liver.append(hausdorff_train[64])
hausdorff_train_liver.append(hausdorff_train[67])
hausdorff_train_liver.append(hausdorff_train[70])
hausdorff_train_liver.append(hausdorff_train[73])
hausdorff_train_liver.append(hausdorff_train[76])
hausdorff_train_liver.append(hausdorff_train[79])
hausdorff_train_liver.append(hausdorff_train[82])
hausdorff_train_liver.append(hausdorff_train[85])
hausdorff_train_liver.append(hausdorff_train[88])
hausdorff_train_liver.append(hausdorff_train[91])
hausdorff_train_liver.append(hausdorff_train[94])
hausdorff_train_liver.append(hausdorff_train[97])


hausdorff_train_lungs.append(hausdorff_train[2])
hausdorff_train_lungs.append(hausdorff_train[5])
hausdorff_train_lungs.append(hausdorff_train[8])
hausdorff_train_lungs.append(hausdorff_train[11])
hausdorff_train_lungs.append(hausdorff_train[14])
hausdorff_train_lungs.append(hausdorff_train[17])
hausdorff_train_lungs.append(hausdorff_train[20])
hausdorff_train_lungs.append(hausdorff_train[23])
hausdorff_train_lungs.append(hausdorff_train[26])
hausdorff_train_lungs.append(hausdorff_train[29])
hausdorff_train_lungs.append(hausdorff_train[32])
hausdorff_train_lungs.append(hausdorff_train[35])
hausdorff_train_lungs.append(hausdorff_train[38])
hausdorff_train_lungs.append(hausdorff_train[41])
hausdorff_train_lungs.append(hausdorff_train[44])
hausdorff_train_lungs.append(hausdorff_train[47])
hausdorff_train_lungs.append(hausdorff_train[50])
hausdorff_train_lungs.append(hausdorff_train[53])
hausdorff_train_lungs.append(hausdorff_train[56])
hausdorff_train_lungs.append(hausdorff_train[59])
hausdorff_train_lungs.append(hausdorff_train[62])
hausdorff_train_lungs.append(hausdorff_train[65])
hausdorff_train_lungs.append(hausdorff_train[68])
hausdorff_train_lungs.append(hausdorff_train[71])
hausdorff_train_lungs.append(hausdorff_train[74])
hausdorff_train_lungs.append(hausdorff_train[77])
hausdorff_train_lungs.append(hausdorff_train[80])
hausdorff_train_lungs.append(hausdorff_train[83])
hausdorff_train_lungs.append(hausdorff_train[86])
hausdorff_train_lungs.append(hausdorff_train[89])
hausdorff_train_lungs.append(hausdorff_train[92])
hausdorff_train_lungs.append(hausdorff_train[95])
hausdorff_train_lungs.append(hausdorff_train[98])


#CREATE EXCEL FILE

workbook = xlsxwriter.Workbook('/home/imoreira/Metrics/Labels/Test/EvaluationMetrics.xlsx')
#workbook = xlsxwriter.Workbook('C:\\Users\\isasi\\Downloads\\Metrics\\EvaluationMetrics.xlsx')

worksheet = workbook.add_worksheet()

bold = workbook.add_format({'bold': True})

#DECLARE DATA
organs = ["Bladder", "Liver", "Lungs"]

metrics = ["Dice", "Jaccard Index", "Hausdorff Distance"]

values = ["CNS044", "CNS045", "CNS046", "CNS047", "CNS048", "CNS049", "CNS051", "CNS052", "CNS054", "CNS058"]



#WRITE HEADERS
worksheet.write("A1", "DL_Test", bold)
worksheet.write("A2", "CT", bold)
worksheet.write("C1", "Bladder", bold)
worksheet.write("F1", "Liver", bold)
worksheet.write("I1", "Lungs", bold)
worksheet.write("B2", "Dice", bold)
worksheet.write("C2", "Jaccard Index", bold)
worksheet.write("D2", "Hausdorff Distance (mm)", bold)
worksheet.write("E2", "Dice", bold)
worksheet.write("F2", "Jaccard Index", bold)
worksheet.write("G2", "Hausdorff Distance (mm)", bold)
worksheet.write("H2", "Dice", bold)
worksheet.write("I2", "Jaccard Index", bold)
worksheet.write("J2", "Hausdorff Distance (mm)", bold)

worksheet.write("A14", "MAS", bold)
worksheet.write("A15", "CT", bold)
worksheet.write("C14", "Bladder", bold)
worksheet.write("F14", "Liver", bold)
worksheet.write("I14", "Lungs", bold)
worksheet.write("B15", "Dice", bold)
worksheet.write("C15", "Jaccard Index", bold)
worksheet.write("D15", "Hausdorff Distance (mm)", bold)
worksheet.write("E15", "Dice", bold)
worksheet.write("F15", "Jaccard Index", bold)
worksheet.write("G15", "Hausdorff Distance (mm)", bold)
worksheet.write("H15", "Dice", bold)
worksheet.write("I15", "Jaccard Index", bold)
worksheet.write("J15", "Hausdorff Distance (mm)", bold)

worksheet.write("A27", "DL_VAL", bold)
worksheet.write("A28", "CT", bold)
worksheet.write("C27", "Bladder", bold)
worksheet.write("F27", "Liver", bold)
worksheet.write("I27", "Lungs", bold)
worksheet.write("B28", "Dice", bold)
worksheet.write("C28", "Jaccard Index", bold)
worksheet.write("D28", "Hausdorff Distance (mm)", bold)
worksheet.write("E28", "Dice", bold)
worksheet.write("F28", "Jaccard Index", bold)
worksheet.write("G28", "Hausdorff Distance (mm)", bold)
worksheet.write("H28", "Dice", bold)
worksheet.write("I28", "Jaccard Index", bold)
worksheet.write("J28", "Hausdorff Distance (mm)", bold)

worksheet.write("A40", "DL_TRAIN", bold)
worksheet.write("A41", "CT", bold)
worksheet.write("C40", "Bladder", bold)
worksheet.write("F40", "Liver", bold)
worksheet.write("I40", "Lungs", bold)
worksheet.write("B41", "Dice", bold)
worksheet.write("C41", "Jaccard Index", bold)
worksheet.write("D41", "Hausdorff Distance (mm)", bold)
worksheet.write("E41", "Dice", bold)
worksheet.write("F41", "Jaccard Index", bold)
worksheet.write("G41", "Hausdorff Distance (mm)", bold)
worksheet.write("H41", "Dice", bold)
worksheet.write("I41", "Jaccard Index", bold)
worksheet.write("J41", "Hausdorff Distance (mm)", bold)

for item in range(len(values)):
    worksheet.write(item+2, 0, values[item], bold)

for item in range(len(values)):
    worksheet.write(item+15, 0, values[item], bold)

#DICE PARA BLADDER
for item in range(len(dice_bladder)):
    worksheet.write(item+2, 1, dice_bladder[item])

#DICE PARA LIVER
for item in range(len(dice_liver)):
    worksheet.write(item+2, 4, dice_liver[item])

#DICE PARA LUNGS
for item in range(len(dice_lungs)):
    worksheet.write(item+2, 7, dice_lungs[item])

#JACCARD PARA BLADDER
for item in range(len(jc_i_bladder)):
    worksheet.write(item+2, 2, jc_i_bladder[item])

#JACCARD PARA LIVER
for item in range(len(jc_i_liver)):
    worksheet.write(item+2, 5, jc_i_liver[item])

#JACARD PARA LUNGS
for item in range(len(jc_i_lungs)):
    worksheet.write(item+2, 8, jc_i_lungs[item])

#HD PARA BLADDER
for item in range(len(hausdorff_bladder)):
    worksheet.write(item+2, 3, hausdorff_bladder[item])

#HD PARA LIVER
for item in range(len(hausdorff_liver)):
    worksheet.write(item+2, 6, hausdorff_liver[item])

#HD PARA LUNGS
for item in range(len(hausdorff_lungs)):
    worksheet.write(item+2, 9, hausdorff_lungs[item])


###MAS

#DICE PARA BLADDER
for item in range(len(dice_bladder_mas)):
    worksheet.write(item+15, 1, dice_bladder_mas[item])

#DICE PARA LIVER
for item in range(len(dice_liver_mas)):
    worksheet.write(item+15, 4, dice_liver_mas[item])

#DICE PARA LUNGS
for item in range(len(dice_lungs_mas)):
    worksheet.write(item+15, 7, dice_lungs_mas[item])

#JACCARD PARA BLADDER
for item in range(len(jc_i_bladder_mas)):
    worksheet.write(item+15, 2, jc_i_bladder_mas[item])

#JACCARD PARA LIVER
for item in range(len(jc_i_liver_mas)):
    worksheet.write(item+15, 5, jc_i_liver_mas[item])

#JACARD PARA LUNGS
for item in range(len(jc_i_lungs_mas)):
    worksheet.write(item+15, 8, jc_i_lungs_mas[item])

#HD PARA BLADDER
for item in range(len(hausdorff_bladder_mas)):
    worksheet.write(item+15, 3, hausdorff_bladder_mas[item])

#HD PARA LIVER
for item in range(len(hausdorff_liver_mas)):
    worksheet.write(item+15, 6, hausdorff_liver_mas[item])

#HD PARA LUNGS
for item in range(len(hausdorff_lungs_mas)):
    worksheet.write(item+15, 9, hausdorff_lungs_mas[item])


###VAL

#DICE PARA BLADDER
for item in range(len(dice_val_bladder)):
    worksheet.write(item+28, 1, dice_val_bladder[item])

#DICE PARA LIVER
for item in range(len(dice_val_liver)):
    worksheet.write(item+28, 4, dice_val_liver[item])

#DICE PARA LUNGS
for item in range(len(dice_val_lungs)):
    worksheet.write(item+28, 7, dice_val_lungs[item])

#JACCARD PARA BLADDER
for item in range(len(jc_i_val_bladder)):
    worksheet.write(item+28, 2, jc_i_val_bladder[item])

#JACCARD PARA LIVER
for item in range(len(jc_i_val_liver)):
    worksheet.write(item+28, 5, jc_i_val_liver[item])

#JACARD PARA LUNGS
for item in range(len(jc_i_val_lungs)):
    worksheet.write(item+28, 8, jc_i_val_lungs[item])

#HD PARA BLADDER
for item in range(len(hausdorff_val_bladder)):
    worksheet.write(item+28, 3, hausdorff_val_bladder[item])

#HD PARA LIVER
for item in range(len(hausdorff_val_liver)):
    worksheet.write(item+28, 6, hausdorff_val_liver[item])

#HD PARA LUNGS
for item in range(len(hausdorff_val_lungs)):
    worksheet.write(item+28, 9, hausdorff_val_lungs[item])

workbook.close()


print("Process Finish!")
