import os
import nibabel as nib
import numpy as np


'''
Label 1: Bladder
Label 2: Heart
Label 3: Liver
Label 4: Lungs
Label 5: Pancreas
'''

root_dir = "C:\\Users\\isasi\\Downloads"
segs_dir = "C:\\Users\\isasi\\Downloads\\SEGS_INFERENCES_FINAL\\MULTI_ORGAN\\SEGS\\TEST\\Final3\\"
out_dir = "C:\\Users\\isasi\\Downloads\\SEGS_INFERENCES_FINAL\\MULTI_ORGAN\\SEPERATE_SEGS\\"

_, _, filenames = next(os.walk(segs_dir))
seg_list = []
organ_list = ['bladder', 'brain', 'liver', 'lungs', 'heart', 'pancreas', 'kidneys']
organ_number = [1, 2, 3, 4, 5, 6, 7]

for f in filenames:
    if ('CNS' in f):
        seg_list.append(f.split('_')[0])
        # organ_list.append(f.split('_')[1])
case_names = list(set(seg_list))
print(case_names)

for case_name in case_names:
    for index in range(len(organ_list)):

        seg = case_name + '_ct_seg' + '.nii.gz'
        seg_path = segs_dir + seg
        img_nii = nib.load(seg_path)
        img_arr = np.asarray(img_nii.get_data())
        single_seg = np.zeros_like(img_arr)
        single_seg[img_arr == organ_number[index]] = 1


        ni_img = nib.Nifti1Image(single_seg, img_nii.affine)
        name = 'Seg_MO_' + case_name + '_' + organ_list[index] + '.nii.gz'
        nib.save(ni_img, out_dir + name)

print('FINISH')






