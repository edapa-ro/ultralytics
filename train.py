from ultralytics.models import YOLO
import argparse
import os

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

def initialize_model_from_yaml_or_model(model_yaml, model_path):
    if model_yaml == None and model_path == None:
        print("NU AI TRANSMIS CA PARAMETRU NICI MODELUL NICI YAML")
        return None
    is_valid_yaml = (not model_yaml == None) and os.path.isfile(model_yaml) and model_yaml.endswith(".yaml")
    is_valid_path = (not model_path == None) and os.path.isfile(model_path) and model_path.endswith(".pt")
    if (is_valid_yaml and is_valid_path):
        print("SE VA CONTINUA ANTRENAREA SE IGNORA YAML")
        return YOLO(model_path)
    if (is_valid_yaml and not is_valid_path):
        print("SE VA CREEA MODELUL DIN FISIERUL YAML, PATH-UL MODELULUI NU ESTE VALID")
        return YOLO(model_yaml)
    if (not is_valid_yaml and is_valid_path):
        print("SE VA CONTINUA ANTRENAREA SE IGNORA YAML, YAML NU ESTE VALID")
        return YOLO(model_path)
    if (not is_valid_yaml and not is_valid_path):
        print("NU ESTE VALID NICI PATH-UL NICI YAML-UL, cum e posibil")
        return None

if (not os.path.isfile(data_yaml)) or (not data_yaml.endswith(".yaml")):
    print("FISIERUL YAML TRANSMIS CA PARAMETRU PENTRU SETUL DE DATE ESTE INVALID")
    exit(1)
model = initialize_model_from_yaml_or_model(model_yaml, model_path)
if (model == None): 
    print("NU S-A PUTUT INITIALIZA MODELUL\n")
    exit(1)

print(size)
print("SE INCEPE ACUM ANTRENAREA")
model.train(
    data=data_yaml,
    epochs=200,
    imgsz=size,
    rect=False,
    batch=64,
    save_period=20,
    model=model,
    resume=False,
    verbose=False,
    patience=50
)