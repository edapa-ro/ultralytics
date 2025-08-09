import argparse
import os
from pyparsing import Any, Union
import torch
import yaml
from typing import Tuple

SHAPE_VALUE_LAYER = []

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

def process_conv_layer(input : Tuple[int, int, int], colector : int,
                       number_repetitions : int, parameters_list : list[Any]) -> Tuple[int, int, int]:
    c2 =  parameters_list[0]
    k = 1 if len(parameters_list) < 2 else int(parameters_list[1])
    s = 1 if len(parameters_list) < 3 else int(parameters_list[2])
    p = None if len(parameters_list) < 4 else int(parameters_list[3])
    g = 1 if len(parameters_list) < 5 else int(parameters_list[4])
    d = 1 if len(parameters_list) < 6 else int(parameters_list[5])
    p = autopad(k, p, d) if p == None else p
    h = int((input[1] + 2 * p - d * (k - 1) - 1) / s + 1)
    w = int((input[2] + 2 * p - d * (k - 1) - 1) / s + 1)
    return (c2, h, w)

def process_C3_C2_layer(input : Tuple[int, int, int], colector : int, 
                        number_repetitions : int, parameters_list : list[Any]) -> Tuple[int, int, int]:
    return (parameters_list[0], input[1], input[2])  

def process_concat_layer(input : Tuple[int, int, int], colector : Union[list[int], int], 
                         number_repetitions : int, parameters_list : list[Any]) -> Tuple[int, int, int]:
    first_input  = input if  colector[0] == -1 else  SHAPE_VALUE_LAYER[colector[0]]
    second_input = input if  colector[1] == -1 else  SHAPE_VALUE_LAYER[colector[1]]
    if (second_input[1] != first_input[1]):
        print(f"Warning: Concatenation inputs have different channels: {first_input[1]} and {second_input[1]}.")
        exit(1)
    if (second_input[2] != first_input[2]):
        print(f"Warning: Concatenation inputs have different height: {first_input[2]} and {second_input[2]}.")
        exit(1)
    print(first_input, second_input, colector)
    return (first_input[0] + second_input[0], first_input[1], first_input[2])

def process_CSP1_layer(input : Tuple[int, int, int], colector : int, number_repetitions : int,
                 parameters_list : list[Any]) -> Tuple[int, int, int]:
    c2 =  parameters_list[0]
    n = 1 if len(parameters_list) < 2 else int(parameters_list[1])
    s = 1 if len(parameters_list) < 3 else int(parameters_list[2])
    return (c2, int(input[1] / s), int(input[2] / s))


def process_CSP2_layer(input : Tuple[int, int, int], colector : int, 
                             number_repetitions : int, parameters_list : list[Any]) -> Tuple[int, int, int]:
    return (parameters_list[0], input[1], input[2])

def process_SPPF_layer(input : Tuple[int, int, int], colector : int, 
                       number_repetitions : int, parameters_list : list[Any]) -> Tuple[int, int, int]:
    return (parameters_list[0], input[1], input[2])

def process_Upsample_layer(input : Tuple[int, int, int], colector : int, 
                            number_repetitions : int, parameters_list : list[Any]) -> Tuple[int, int, int]:
    scale_factor = int(parameters_list[1]) if len(parameters_list) >= 2 else -1
    return (input[0], input[1] * scale_factor, input[2] * scale_factor)

def process_AAM_SPP_layer(input : Tuple[int, int, int], colector : int,
                          number_repetitions : int, parameters_list : list[Any]) -> Tuple[int, int, int]:
    return (parameters_list[0], input[1], input[2])

def process_Detect_layer(input : Tuple[int, int, int], colector : list[int],
                         number_repetitions : int, parameters_list : list[Any]) -> Tuple[int, int, int]:
    for idx in colector:
        print(SHAPE_VALUE_LAYER[idx])
    return (parameters_list[0], input[1], input[2])

def select_layer_type(input : Tuple[int, int, int], layer : list) -> Tuple[int, int, int]:
    colector = layer[0]
    number_repetitions = int(layer[1])
    layer_type = str(layer[2])
    parameters_list = layer[3]
    if layer_type == 'Conv' or layer_type == 'Debug_Conv':
        return process_conv_layer(input, colector, number_repetitions, parameters_list)
    elif layer_type == 'C3' or layer_type == 'C2':
        return process_C3_C2_layer(input, colector, number_repetitions, parameters_list)
    elif layer_type == 'Concat' or layer_type == 'Debug_Concat':
        return process_concat_layer(input, colector, number_repetitions, parameters_list)
    elif layer_type == 'CSP1':
        return process_CSP1_layer(input, colector, number_repetitions, parameters_list)
    elif layer_type == 'CSP2':
        return process_CSP2_layer(input, colector, number_repetitions, parameters_list)
    elif layer_type == 'SPPF':
        return process_SPPF_layer(input, colector, number_repetitions, parameters_list)
    elif layer_type == 'nn.Upsample':
        return process_Upsample_layer(input, colector, number_repetitions, parameters_list)
    elif layer_type == 'SPP' or layer_type == 'AAM':
        return process_AAM_SPP_layer(input, colector, number_repetitions, parameters_list)
    elif layer_type == 'Detect':
        return process_Detect_layer(input, colector, number_repetitions, parameters_list)
    else:
        print(f"Layer type {layer_type} is not recognized or not implemented.")    
    return input 

parser = argparse.ArgumentParser()
parser.add_argument('-y', '--yaml', help='yaml file path', type=str, required=True)
parser.add_argument('-s', '--size', help='image size', type=int, default=1024)
args = parser.parse_args()

yaml_path = args.yaml
image_size = args.size
if (yaml_path == None):
    print("NU A FOST TRANSMIS YAML-UL CA PARAMETRU PENTRU ANALIZARE")
    exit(1)
if (not os.path.isfile(yaml_path)) or (not yaml_path.endswith(".yaml")):
    print("FISIERUL TRANSMIS CA PARAMETRU PENTRU YAML NU ESTE VALID PENTRU ANALIZARE")
    exit(1)
if image_size == None or image_size <= 0:
    print("DIMENSIUNEA IMAGINII TREBUIE SA FIE UN NUMAR POZITIV")
    exit(1)

with open(yaml_path, 'r') as f:
    data = yaml.load(f, Loader=yaml.SafeLoader)
    


backbone_layers = data['backbone']
backbone_layers.extend(data['head'])
for idx, layer in enumerate(backbone_layers):
    old_tuple = SHAPE_VALUE_LAYER[-1] if len(SHAPE_VALUE_LAYER) > 0 else (3, image_size, image_size)
    resulting_tuple = select_layer_type(old_tuple, layer)
    SHAPE_VALUE_LAYER.append(resulting_tuple)
    print(f"Layer {idx}: {resulting_tuple}")

# for idx, layer_shape in enumerate(SHAPE_VALUE_LAYER):
#     print(f"Layer {idx}: {layer_shape}")


from ultralytics.models import YOLO
model = YOLO(yaml_path)
x = torch.rand(1, 3, image_size, image_size)
model(x)

# print(end='\n\n')
# for layer in header_layers:
#     print(layer, layer[0], layer[1], layer[2], layer[3])