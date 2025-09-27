import gdown
import pandas as pd
import os
import yaml
import re
import shutil
from zipfile import ZipFile
# MEDIEVAL_VILLAGE NU E BINE LA NUME

POSSIBLE_LOCATION_DATA_FOLDER = [
    "train/images",
    "train/labels",
    "val/images",
    "val/labels",
    "test/images",
    "test/labels"
]
INTERMEDIARY_ARCHIVES_LOCATION = "archives_data"
ROOT_DATASET_LOCATION = "dataset"

def get_drive_id(url):
    match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)
    return None

def get_archives_locations() -> list[tuple[str, str]]:
    df = pd.read_csv("data_info.csv")
    names = df['archive_name'].to_list()
    links = df['archive_link'].to_list()
    out_list = []
    for (name, link) in zip(names, links):
        out_list.append((name, get_drive_id(link)))
    return out_list

def download_all_archives(archives_list : list[tuple[str, str]]):
    for (archive_name, archive_id) in archives_list:
        print(f"Analizeaza arhiva {archive_name}.")
        if os.path.exists(archive_name) or os.path.exists(f"{INTERMEDIARY_ARCHIVES_LOCATION}/{archive_name}"):
            continue
        print(f"Donwload la arhiva {archive_name}")
        url = f"https://drive.google.com/uc?id={archive_id}"
        gdown.download(url, quiet=False)
        shutil.move(archive_name, INTERMEDIARY_ARCHIVES_LOCATION)
        print()
        
def move_all_archives_into_data(archives_list : list[tuple[str, str]]):
    for (archive_name, _) in archives_list:
        if (os.path.exists(f"{INTERMEDIARY_ARCHIVES_LOCATION}/{archive_name}")):
            continue
        shutil.move(archive_name, f"{INTERMEDIARY_ARCHIVES_LOCATION}/{archive_name}")

def open_all_archives(archives_list):
    for (archive_name, _) in archives_list:
        filename = os.path.splitext(os.path.basename(f"{INTERMEDIARY_ARCHIVES_LOCATION}/{archive_name}"))[0]
        if (os.path.exists(f"{INTERMEDIARY_ARCHIVES_LOCATION}/{filename}")):
            continue
        with ZipFile(f"{INTERMEDIARY_ARCHIVES_LOCATION}/{archive_name}", 'r') as zObject:
            zObject.extractall(path=INTERMEDIARY_ARCHIVES_LOCATION)

def create_file_structure():    
    for location in POSSIBLE_LOCATION_DATA_FOLDER:
        path = os.path.join(ROOT_DATASET_LOCATION, location)
        if (os.path.isdir(path)): continue
        os.makedirs(path)

def merge_all_datasets(archives_list : list[tuple[str, str]]):
    archives_names = [os.path.splitext(os.path.basename(archive_name[0]))[0] for archive_name in archives_list]
    for filename in archives_names: 
        for location in POSSIBLE_LOCATION_DATA_FOLDER:
            source = os.path.join(f"{INTERMEDIARY_ARCHIVES_LOCATION}/{filename}", location)
            destination = os.path.join(ROOT_DATASET_LOCATION, location)
            if os.path.isdir(source) and os.path.isdir(destination): 
                all_files_paths = [os.path.join(source, file)  for file in os.listdir(source)]
                for file_path in all_files_paths:
                    shutil.move(file_path, destination)
            else:
                print("NU ESTE NIMIC ACOLO")

def create_yaml_file():
    data_yaml = {
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": [
            "Person"
            ]
    }  
    new_path = os.path.join(ROOT_DATASET_LOCATION, "dataset.yaml")
    with open(new_path, 'w') as file:
        yaml.dump(data_yaml, file)


if (not os.path.isdir(INTERMEDIARY_ARCHIVES_LOCATION)):
    os.makedirs(INTERMEDIARY_ARCHIVES_LOCATION)
if (not os.path.isdir(ROOT_DATASET_LOCATION)):
    os.makedirs(ROOT_DATASET_LOCATION)

archives_list = get_archives_locations()
download_all_archives(archives_list)
move_all_archives_into_data(archives_list)
open_all_archives(archives_list)
create_file_structure()
merge_all_datasets(archives_list)
create_yaml_file()

