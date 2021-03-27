
import seg_metrics.seg_metrics as sg
import SimpleITK as sitk


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
