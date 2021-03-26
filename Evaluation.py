
import seg_metrics.seg_metrics as sg
import SimpleITK as sitk


labels_dir = "//home//imoreira//Segmentations"


labels_dicts = [{"image": image_name} for image_name in zip(labels_dir)]

gdth_path = '//home//imoreira//Data//Lungs//labelsTr'

pred_path = '//home//imoreira'

csv_file = 'Metrics.csv'

metrics = sg.write_metrics(labels=labels_dicts[1:],  # exclude background
                  gdth_path=gdth_path,
                  pred_path=pred_path,
                  csv_file=csv_file)

print(metrics)
