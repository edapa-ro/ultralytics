from ultralytics.models import YOLO
import argparse
import os
import fiftyone as fo
import fiftyone.utils.coco as fouc

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model pt path')
parser.add_argument('-d', '--data', help='data yaml')
parser.add_argument('-s', '--size', help='image size')
args = parser.parse_args()

data_yaml = args.data
model_path = args.model
size = int(args.size)
model = None

def initialize_model_from_yaml_or_model(model_path):
    if model_path == None:
        print("NU AI TRANSMIS CA PARAMETRU NICI MODELUL NICI YAML")
        return None
    is_valid_path = (not model_path == None) and os.path.isfile(model_path) and model_path.endswith(".pt")
    if is_valid_path:
        return YOLO(model_path)
    return None 
    
if (not os.path.isfile(data_yaml)) or (not data_yaml.endswith(".yaml")):
    print("FISIERUL YAML TRANSMIS CA PARAMETRU PENTRU SETUL DE DATE ESTE INVALID")
    exit(1)
model = initialize_model_from_yaml_or_model(model_path)
if (model == None): 
    print("NU S-A PUTUT INITIALIZA MODELUL\n")
    exit(1)

print(size)
print("SE INCEPE ACUM VALIDAREA")
results = model.val(
    data=data_yaml,
    imgsz=size,
    conf=0.25,
    iou=0.60,
    rect=False,
    batch=8,
    save_conf=True,
    workers=4,
    verbose=True,
    save_json=True,
    save_txt=True
)

print(results.confusion_matrix.to_df())
print(results.summary())

# # Paths to your data
# ground_truth_json = "/path/to/ground_truth/annotations.json"  # COCO ground truth
# predictions_json = "/path/to/yolo/predictions.json"           # YOLO predictions
# images_dir = "/path/to/images"                                # folder with images

# # Step 1: Load ground truth dataset (COCO format)
# dataset = fouc.from_coco(
#     dataset_name="yolo_val_dataset",
#     dataset_type="detections",
#     data_path=images_dir,
#     labels_path=ground_truth_json,
# )

# # Step 2: Import predictions into the dataset
# fouc.add_coco_labels(
#     dataset,
#     "yolo_predictions",  # field name for predictions
#     predictions_json,
# )

# # Step 3: Launch the FiftyOne app
# session = fo.launch_app(dataset)

# # (Optional) Step 4: Evaluate mAP, precision, recall, etc.
# results = dataset.evaluate_detections(
#     "yolo_predictions",
#     gt_field="detections",
#     eval_key="eval",
#     method="coco",
# )

# # Print evaluation results
# print(results.metrics())
