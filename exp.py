from ultralytics.models.yolo.model import YOLO
from ultralytics.nn.modules.block import AAM
import torch
from torch import nn
from ultralytics.nn.modules.block import Focuss, AMAP
from ultralytics.nn.modules.block import CSP1, CSP2, SPP, Conv, Concat

# model = YOLO("yamls/my.yaml")
model = YOLO("yamls/yolov5nATTENTION.yaml")
x = torch.rand(1, 3, 1024, 1024)
with torch.no_grad():
    y = model.predict(x)
    print(y)
