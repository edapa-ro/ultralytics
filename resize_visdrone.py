from PIL import Image
import os
import argparse
parser = argparse.ArgumentParser()
import sys 
import yaml
from ultralytics.utils import SETTINGS

def check_yaml_is_valid(yaml_file):
    is_valid_yaml = (not yaml_file == None) and os.path.isfile(yaml_file) and yaml_file.endswith(".yaml")
    if not is_valid_yaml:
        sys.exit("FISIERUL YAML DAT LIPSESTE SAU ESTE INVALID")
    return yaml_file

def check_yaml_is_visdrone(yaml_file):
    keyword_list = {"visdrone", "VisDrone", "visDrone"}
    for keyword in keyword_list:
        if keyword in yaml_file: 
            return True
    print("FISIERUL YAML DAT CA PARAMETRU NU ESTE DE LA VISDRONE")
    return False

def get_images_folders(yaml_file):
    with open(yaml_file, 'r') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)
         
    dataset_dir = os.path.join(SETTINGS['datasets_dir'], data['path'])
    return {
        "train" : os.path.join(dataset_dir, data['train']),
        "val" : os.path.join(dataset_dir, data['val']),
        "test" : os.path.join(dataset_dir, data['test']),
    }


def is_resize_necessary(images_folder : str, target_size : tuple[int, int]):
    for _, img_file in enumerate(os.listdir(images_folder)):
        img_path = os.path.join(images_folder, img_file)
        with Image.open(img_path) as img:
            if (img.size != target_size):
                return True
    return False

def resize_all_photos(images_folder : str, target_size : tuple[int, int]):
    for img_file in os.listdir(images_folder):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join(images_folder, img_file)
            with Image.open(img_path) as img:
                if img.size == target_size: continue
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                resized_img.save(img_path)


parser.add_argument('-y', '--yaml', help ='data path')
parser.add_argument('-s', "--size", help ="size to resize to")
args = parser.parse_args()
size = args.size
yaml_file = args.yaml
if (size == None):
    sys.exit("DIMENSIUNEA NU A FOST SPECIFICATA")

target_size = (int(size), int(size))
yaml_file = check_yaml_is_valid(yaml_file)
if (check_yaml_is_visdrone(yaml_file)):
    folder_dictionaries = get_images_folders(yaml_file)
    print(folder_dictionaries)
    for value in folder_dictionaries.values():
        print(f"incepem sa facem redimensionare pt {value}")
        if (is_resize_necessary(value, target_size)):
            resize_all_photos(value, target_size)