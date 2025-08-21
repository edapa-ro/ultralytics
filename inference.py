import time
import numpy as np
from PIL import Image
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
SIZE=1024

def save_file_label_yolo_format(img :np.array, nms_result : np.array, filename : str, save_dir):
    with open(save_dir + "/" + filename.replace(".png", ".txt"), "w") as f: 
            for i in range(nms_result[0].shape[0]):
                cls_id = str(int(nms_result[0][i][5]))
                conf = nms_result[0][i][4]
                if (not np.all(nms_result[0][i][:4] > 0)):
                    continue
                c1 = nms_result[0][i][:2]
                c2 = nms_result[0][i][2:4]
                width =  np.abs(c2[0] - c1[0])
                height = np.abs(c2[1] - c1[1])
                center = c1 + (c2 - c1) / 2 
                array = np.array([center[0], center[1], width, height, conf])
                out_str = cls_id + " " + np.array2string(array).replace("]", "").replace("[", "") + "\n"
                f.write(out_str) 
    # img.save(save_dir + "/" + filename)


def execute_testing(location, model_path, save_dir, conf_thres = 0.20, iou_thres = 0.35):
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
    # data = {}
    for filename in filenames:
        img_path = location + "/" + filename 
        img = Image.open(img_path).resize((SIZE, SIZE)).convert("RGB")
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

        # save_file_label_fiftyone_format(data, img, nms_result, filename)
        save_file_label_yolo_format(img, nms_result, filename,save_dir)
    del interpreter
    del delegates


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='model pt path')
parser.add_argument('-d', '--data', help='test_data')
parser.add_argument('-s', '--save', help='save location')
args = parser.parse_args()

model_path = args.model
data_yaml = args.data 
save_dir = args.save

if (model_path == None):
    print("NU A FOST TRANSMIS MODELUL CA PARAMETRU PENTRU INFERENCE")
    exit(1)
if (not os.path.isfile(model_path)) or (not model_path.endswith(".tflite")):
    print("FISIERUL TRANSMIS CA PARAMETRU PENTRU MODEL NU ESTE VALID PENTRU INFERENCE")
    exit(1)
if (not os.path.isfile(data_yaml)) or (not data_yaml.endswith(".yaml")):
    print("FISIERUL TRANSMIS PENTRU SETUL DE DATE NU ESTE VALID PENTRU INFERENCE")
    exit(1)
if save_dir == None:
    print("NU A FOST TRANSMIS CA PARAMETRU UNDE VA FI SALVAT REZULTATUL")
    exit(1)
try:
    os.mkdir(save_dir)
    print(f"Directory '{save_dir}' created successfully.")
except FileExistsError:
    print(f"Directory '{save_dir}' already exists.")
except PermissionError:
    print(f"Permission denied: Unable to create '{save_dir}'.")
except Exception as e:
    print(f"An error occurred: {e}")

with open(data_yaml, 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
data_key = 'test' if ('test' in data.keys() and (not data['test'] == None))  else 'val'
path = 'datasets/' + data['path'] + '/' + data[data_key]
print(path)

execute_testing(
    location=path,
    model_path=model_path,
    save_dir=save_dir
)


