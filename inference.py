import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from pycoral.utils import edgetpu
import tflite_runtime.interpreter as tflite
from pycoral.utils.edgetpu import list_edge_tpus
import argparse
import os
import yaml

from nms import non_max_suppression_v8
if (len(list_edge_tpus()) == 0): 
    print("NU ESTE CONECTAT USB GOOGLE CORAL LA LAPTOP")
    exit(1)
print(list_edge_tpus())


labels = ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def plot_one_box_pil(box, image, label=None, color=(255, 0, 0), line_width=3, size=640):
    draw = ImageDraw.Draw(image)
    c1 = (int(box[0] * size), int(box[1] * size))
    c2 = (int(box[2] * size), int(box[3] * size))
    draw.rectangle([c1, c2], outline=color, width=line_width)
    
    if label:
        font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_origin = (c1[0], c1[1] - text_height if c1[1] - text_height > 0 else c1[1])
        draw.rectangle(
            [text_origin, (text_origin[0] + text_width, text_origin[1] + text_height)],
            fill=color
        )
        draw.text(text_origin, label, fill=(255, 255, 255), font=font)

    return image

def execute_testing(location, model_path, conf_thres = 0.15, iou_thres = 0.15):
    # Load EdgeTPU delegate
    delegates = [edgetpu.load_edgetpu_delegate()]
    interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=delegates)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_zero = input_details[0]['quantization'][1]
    input_scale = input_details[0]['quantization'][0]
    output_zero = output_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]

    if input_scale < 1e-9: input_scale = 1.0
    if output_scale < 1e-9: output_scale = 1.0

    print("Input shape", input_details[0]['shape'])
    print("Output shape", output_details[0]['shape'])

    filenames = os.listdir(location)
    prediction=None 
    for filename in filenames:
        img_path = location + "/" + filename 
        img = Image.open(img_path).resize((1024, 1024)).convert("RGB")
        img_array = np.expand_dims(np.array(img), axis=0)  # Add batch dimension
        x = img_array.astype('float32') / 255.0
        x = (x / input_scale) + input_zero
        x = x.astype(np.int8)
        
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index']).astype('float32')
        prediction = (prediction - output_zero) * output_scale
        print('%.1fms' % ((time.perf_counter() - start) * 1000))

        prediction = prediction.transpose(0, 2, 1)
        print("Prediction shape:", prediction[0].shape)
        nms_result = non_max_suppression_v8(prediction, conf_thres, iou_thres, None, False, max_det=300)
        print("Number of objects found:", nms_result[0].shape[0])


        for i in range(nms_result[0].shape[0]):
            cls_id = int(nms_result[0][i][5])
            conf = int(nms_result[0][i][4] * 100)
            label = f"{labels[cls_id]} {conf}%"
            plot_one_box_pil(nms_result[0][i][:4], img, label=label, line_width=2, size=1024)

        # Save result
        output_path = "res/drawn-" + filename 
        img.save(output_path)
        print(f"Saved result to {output_path}")
    del interpreter
    del delegates


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model pt path')
parser.add_argument('-d', '--data', help='test_data')
args = parser.parse_args()

model_path = args.model
data_yaml = args.data 
if (model_path == None):
    print("NU A FOST TRANSMIS MODELUL CA PARAMETRU")
    exit(1)
if (not os.path.isfile(model_path)) or (not model_path.endswith(".tflite")):
    print("FISIERUL TRANSMIS CA PARAMETRU PENTRU MODEL NU ESTE VALID")
    exit(1)
if (not os.path.isfile(data_yaml)) or (not data_yaml.endswith(".yaml")):
    print("FISIERUL TRANSMIS PENTRU SETUL DE DATE NU ESTE VALID")
    exit(1)

with open(data_yaml, 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
data_key = 'test' if 'test' in data.keys() else 'val'
path = 'datasets/' + data['path'] + '/' + data[data_key]

execute_testing(
    location=path,
    model_path=model_path
)

