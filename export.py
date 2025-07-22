from ultralytics.models import YOLO
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model pt path')
parser.add_argument('-s', '--size', help='set model size')
parser.add_argument('-d', '--data', help='dataset yaml')
args = parser.parse_args()

model_path = args.model
data_path = args.data
size = int(args.size)

if (size == None):
    print("DIMENSIUNEA RETELEI NU A FOST SPECIFICATA")
    exit(1)
if (model_path == None):
    print("NU A FOST TRANMIS CA PARAMETRU CALEA CATRE MODEL")
    exit(1)
if (not os.path.isfile(model_path)) or (not model_path.endswith(".pt")):
    print("FISIERUL TRANSMIS CA PARAMETRU PENTRU MODEL NU ESTE VALID")
    exit(1)
if (not os.path.isfile(data_path) or (not data_path.endswith(".yaml"))):
    print("INVALID FISIERUL DE CONFIGURARE PENTRU SETUL DE DATE")
    exit(1)

model=YOLO(model_path)
model.export(format='tflite', int8=True, imgsz=(size, size), batch=1, data=data_path, half=False)
