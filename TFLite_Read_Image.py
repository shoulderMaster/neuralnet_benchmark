import glob
import argparse
import io
import os
import time
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
from tflite_runtime.interpreter import Interpreter

def load_labels(path):
    with open(path, 'r') as f:
        return {int(i): line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, image) :
    """Returns a sorted array of classification results."""
    interpreter.invoke()
    interpreter.get_output_details()
    bbox, classes, confidence, _ = [
        np.squeeze(interpreter.get_tensor(output_details['index']))
        for output_details in interpreter.get_output_details()
    ]

    return bbox, classes, confidence

def classify_image(interpreter, image, min_cf=0.5):
    set_input_tensor(interpreter, image)
    bbox, classes, confidence = get_output_tensor(interpreter, image)
    output = sorted([(cf, c, b) for b, c, cf in zip(bbox, classes, confidence) if cf > min_cf])
    print(output)
    return output

def fetch_args() :
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument( '--model', help='File path of .tflite file.', default="detect.tflite")
    parser.add_argument('--labels', help='File path of labels file.', default="labelmap.txt")
    return parser.parse_args()

def draw_rectangle(draw, bbox) :
    bbox = [int(pos*300)  if pos > 0 else 0 for pos in bbox]
    bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]
    print(bbox)
    draw.rectangle(bbox)

def detect_object(img) :
    results = classify_image(interpreter_, img)
    for cf, cls, bbox in results :
        draw = ImageDraw.Draw(img)
        draw_rectangle(draw, bbox)
        img.show()

if __name__ == "__main__" :
    args = fetch_args()
    labels_ = load_labels(args.labels)
    interpreter_ = Interpreter(args.model)
    interpreter_.allocate_tensors()
    a = glob.glob('*.jpg')

    for i in a:
        with Image.open(i) as img :
            detect_object(img)
