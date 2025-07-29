import torch
import torchvision
import fiftyone as fo
import os 


def load_combined_dataset():
    dataset_dir = "./datasets/downtest/"
    dataset = fo.Dataset("downtest_combined")
    dataset.add_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split="test",
        tags="groundtruth",
        label_field="ground_truth"
    )

    pred_dir = "./res/"
    dataset.add_dir(
        dataset_dir=pred_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        split="test",
        tags="test_predictions",
        label_field="predictions",  
    )

    fo.pprint(dataset.stats(include_media=True))
    session = fo.launch_app(dataset)
    session.wait()

load_combined_dataset()

