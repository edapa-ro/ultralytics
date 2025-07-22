from ultralytics.models import YOLO
import argparse
import os
import re
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

parser = argparse.ArgumentParser()
parser.add_argument('-yml', '--yaml', help ='yaml file')
parser.add_argument('-m', '--model', help='model pt path')
parser.add_argument('-d', '--data', help='data yaml')
parser.add_argument('-s', '--size', help='image size')
args = parser.parse_args()

model_yaml = args.yaml
data_yaml = args.data
model_path = args.model
size = int(args.size)
model = None

if (data_yaml == None):
    print("NU SA TRANSMIS CA PARAMETRUL UN FISIER TIP YAML PENTRU SETUL DE DATE") 
    exit(1)
if not os.path.isfile(data_yaml):
    print("FISIERUL TRANSMIS NU ESTE UN YAML VALID")
    exit(1)
if (not model_path == None):
    if (not os.path.isfile(model_path)) or (not model_path.endswith(".pt")):
        print("FISIERUL TRANSMIS CA PARAMETRU PENTRU MODEL NU ESTE VALID")
        exit(1)
if (not model_yaml == None):
    if (not os.path.isfile(model_yaml)) or (not model_yaml.endswith(".yaml")):
        print("FISIERUL YAML TRANSMIS CA PARAMETRU ESTE INVALID")
        exit(1)
if model_yaml == None and model_path == None:
    print("NU AI TRANSMIS CA PARAMETRU NICI MODELUL NICI YAML")
    exit(1) 
elif model_yaml == None and (not model_path == None):
    print("SE VA CONTINUA ANTRENAREA")
    model = YOLO(model_path)
elif (not model_yaml == None) and (model_path == None):
    print("MODELUL S-A CREAT DIN YAML")
    model = YOLO(model_yaml)
    filename = os.path.basename(model_yaml)          # 'yolov5s.yaml'
    filename_no_ext = os.path.splitext(filename)[0]  # 'yolov5s'
    model.save(f"{filename_no_ext}.pt")
elif (not model_yaml == None) and (not model_path == None):
    print("SE VA CONTINUA ANTRENAREA, SE VA IGNORA YAML")
    model = YOLO(model_path)

if model == None:
    print("NU S-A PUTUT INITIALIZA MODELUL")
    exit(1)


print("SE INCEPE ACUM ANTRENAREA")
model.train(
    data=data_yaml,
    epochs=20,
    imgsz=size,
    batch=4,
    save_period=3,
    model=model,
    resume=False,
    verbose=False
)

# print("SE MUTA ACUM CEL MAI BUN MODEL ANTRENAT")
# path ="runs/detect"
# pattern = r'^train\d*$'
# files_folders = os.listdir(path)
# if (len(files_folders) == 0):
#     print("NU EXISTA REZULTATE DE ANTRENARE") 
#     exit(1)

# train_sessions = [dir for dir in files_folders if re.match(pattern, dir) and (os.path.isdir(path + "/" + dir))]
# train_sessions.sort()
# best_parameters_model = path + "/" + train_sessions[-1] + "/weights/best.pt"
# if os.path.isfile(best_parameters_model):
#     print("MODELUL A FOST GASIT. VA FI MUTAT SI REDENUMIT.")
#     os.rename(best_parameters_model, "./best.pt")
# else:
#     print("MODELUL NU A FOST GASIT")