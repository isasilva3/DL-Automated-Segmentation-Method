
import seg_metrics.seg_metrics as sg
import SimpleITK as sitk

'''
labels_dir = '/home/imoreira/Segmentations/Pred'


labels_dicts = [{"image": image_name} for image_name in zip(labels_dir)]

gdth_path = '/home/imoreira/Data/Lungs/labelsTr/Labels'

pred_path = '/home/imoreira/Segmentations/Pred'

csv_file = '/home/imoreira/Metrics.csv'

metrics = sg.write_metrics(labels=labels_dicts[1:],  # exclude background
                  gdth_path=gdth_path,
                  pred_path=pred_path,
                  csv_file=csv_file)

print(metrics)
'''

labels = [0, 4, 5 ,6 ,7 , 8]

gdth_file = '/home/imoreira/Data/Lungs/labelsTr/Labels/CNS044_lungs.nii.gz'

pred_file = '/home/imoreira/Segmentations/Pred/CNS044_ct_seg.nii.gz'

csv_file = '/home/imoreira/Metrics.csv'

metrics = sg.write_metrics(labels=labels[1:],  # exclude background if needed
                  gdth_path=gdth_file,
                  pred_path=pred_file,
                  csv_file=csv_file,)