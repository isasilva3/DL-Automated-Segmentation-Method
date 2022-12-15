import os
import nibabel as nib
import numpy as np
from skimage.measure import label

root_dir = "C:\\Users\\isasi\\Downloads"
segs_dir = "C:\\Users\\isasi\\Downloads\\SEGS_INFERENCES_FINAL\\SINGLE_ORGAN\\KIDNEYS\\TRAIN\\Final2\\"
out_dir = "C:\\Users\\isasi\\Downloads\\SEGS_INFERENCES_FINAL\\SINGLE_ORGAN\\KIDNEYS\\WITH_LARGEST_CC\\TRAIN\\Final2\\"
_, _, filenames = next(os.walk(segs_dir))
seg_list = []
organ_list = ['bladder', 'brain', 'liver', 'lungs', 'heart', 'pancreas', 'kidneys']
organ_number = [1, 2, 3, 4, 5, 6, 7]

for f in filenames:
    if ('CNS' in f):
        #seg_list.append(f.split('_')[2])
        seg_list.append(f.split('_')[0])
case_names = list(set(seg_list))
print(case_names)

for case_name in case_names:

    #seg = 'Seg' + '_MO_' + case_name + '_pancreas.nii.gz'
    seg= case_name + '_ct_seg' + '_kidneys.nii.gz'
    seg_path = segs_dir + seg
    img_nii = nib.load(seg_path)
    img_arr = np.asarray(img_nii.get_data())

    #print('img_arr data type', img_arr.dtype)

    labels = label(img_arr)
    assert(labels.max() != 0)  # assume at least 1 CC
    largestCC_1 = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    largestCC_1 = np.float32(largestCC_1)
    #print('largestCC data type', largestCC.dtype)

    labels = label(img_arr - largestCC_1)
    if (labels.max() != 0): #if there is at least 1 CC
        largestCC_2 = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        largestCC_2 = np.float32(largestCC_2)
        largestCC = largestCC_1 + largestCC_2
    else: #if there isn't any CC
        largestCC = largestCC_1

    ni_img = nib.Nifti1Image(largestCC, img_nii.affine)
    name = 'Seg_' + case_name + '_' + '_kidneys.nii.gz'
    nib.save(ni_img, out_dir + name)

print('FINISH')






