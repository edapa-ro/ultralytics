import torch
import torchvision
import fiftyone as fo
import os 

name = "downtest"
dataset_dir = "./datasets/downtest/"
labels_path = "./datasets/downtest/labels/train"
splits = ['train', 'val', 'test']
dataset = fo.Dataset(name)
for split in splits:
    dataset.add_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split=split,
        tags=split,
    )

print(dataset)
print(dataset.head())



# samples = []
# for img_filename in os.listdir(img_path):
#     sample = fo.Sample(filepath=img_path + "/" + img_filename)
#     detections = []
#     labels_file = labels_path + "/" + img_filename
#     if (not os.path.isfile(labels_file)):
#         print("NU A FOST GASIT FISIERUL DE CE ???")
#         continue
#     with open(labels_file) as f:
#         for line in f:
#           parts = line.strip().split()
#           class_id = int(parts[0])
#           x_center = float(parts[1])
#           y_center = float(parts[2])
#           x_width = float(parts[3])
#           y_width = float(parts[4])


