import seg_metrics.seg_metrics as sg
import SimpleITK as sitk

#labels_dir = '/home/imoreira/Segmentations/Pred'

#labels_dicts = [{"image": image_name} for image_name in zip(labels_dir)]



gdth_path = '/home/imoreira/Metrics/Labels'

pred_path = '/home/imoreira/Metrics/Pred'

csv_file = 'Metrics.csv'

#labels = [0, 4, 5 ,6 ,7 , 8]

metrics = sg.write_metrics(gdth_path=gdth_path,
                  pred_path=pred_path,
                  csv_file=csv_file)

print(metrics)
