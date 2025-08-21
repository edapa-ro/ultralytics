import argparse
import fiftyone as fo
import os 


def load_combined_dataset(dataset : fo.Dataset, predictions_dir : str):
    for sample in dataset:
        filename = os.path.basename(sample.filepath).replace(".png", ".txt")
        with open(os.path.join(predictions_dir, filename), "r") as f:
            detections = []
            for line in f:
                 values = line.split()
                 out  =[float(val) for val in values[1:5]]
                 out[0] = out[0] - out[2] / 2
                 out[1] = out[1] - out[3] / 2
                 detections.append(fo.Detection(
                     label="person",
                     bounding_box=out,
                     confidence=float(values[5])
                 ))
            sample["predictions"] = fo.Detections(detections=detections)
        sample.save()


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', help='test_data')
parser.add_argument('-l', '--labels', help='predictions_labels')
args = parser.parse_args()

label_path = args.labels
data_path = args.data
if (data_path == None):
    print("NU A FOST TRANSMIS SETUL DE DATE CA PARAMETRU PENTRU VISUALIZARE")
    exit(1)
if (not os.path.isdir(data_path)):
    print("NU A FOST TRANSMIS O CALE VALIDA PENTRU SETUL DE DATE")
    exit(1)
    
if (label_path == None):
    print("NU A FOST TRANSMISA LOCATIA PREDICTIILOR FACUTE CA PARAMETRU PENTRU VISUALIZARE")
    exit(1)
if (not os.path.isdir(label_path)):
    print("NU A FOST TRANSMISA O CALE VALIDA PENTRU PREDICTII")
    exit(1)


# dataset_dir = "./datasets/downtest/"
#     predictions_dir = "./res/labels/test/"

dataset = fo.Dataset("downtest_combined")
dataset.add_dir(
    dataset_dir=data_path,
    dataset_type=fo.types.YOLOv5Dataset,
    split="test",
    tags="groundtruth",
    label_field="ground_truth"
)

load_combined_dataset(dataset, predictions_dir=label_path)
fo.pprint(dataset.stats(include_media=True))
session = fo.launch_app(dataset)
session.wait()

