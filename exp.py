from ultralytics.nn.modules.block import AAM
import torch
from ultralytics import YOLO
path = 'yamls/yolov5nATTENTION.yaml'
model = YOLO(path)
print(model.info())