from roboflow import Roboflow
import os
import shutil
import yaml
from dotenv import load_dotenv
load_dotenv()  

DATASET_PATH="standford-1"
YAML_FILE_NAME="mystandford.yaml"
FINAL_CONFIG_LOCATION="ultralytics/cfg/datasets"
FINAL_DATASET_LOCATION="datasets"

def handle_existing_dataset():
    if os.path.isfile(os.path.join(DATASET_PATH, "README.dataset.txt")):
        os.remove(os.path.join(DATASET_PATH, "README.dataset.txt"))
    if os.path.isfile(os.path.join(DATASET_PATH, "README.roboflow.txt")):
        os.remove(os.path.join(DATASET_PATH, "README.roboflow.txt"))
    new_yaml_file = {
        "path": DATASET_PATH,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": ["Person"]
    }
    configuration_yaml_path = os.path.join(FINAL_CONFIG_LOCATION, YAML_FILE_NAME)
    if not os.path.exists(configuration_yaml_path):
        with open(configuration_yaml_path, 'w') as file:
            file.write(yaml.dump(new_yaml_file))
    if not os.path.exists(os.path.join(FINAL_DATASET_LOCATION, DATASET_PATH)):
        shutil.move(DATASET_PATH, os.path.join(FINAL_DATASET_LOCATION, DATASET_PATH))
    

def download_dataset():
    if os.path.exists(DATASET_PATH) or os.path.exists(f"datasets/{DATASET_PATH}"):
        print("Dataset already exists. Skipping download.")
    else:
        rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
        project = rf.workspace("yolodetection-upvcu").project("standford-jntug-cws9f")
        version = project.version(1)
        dataset = version.download("yolov8")
    handle_existing_dataset()

download_dataset()