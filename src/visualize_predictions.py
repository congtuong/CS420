import os 
import json
import glob

from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw

image_path = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/splited/A/train/images')
labels_path = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/splited/A/train/labels')
output_path = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/visualize/A/val')

if not os.path.exists(output_path):
    os.makedirs(output_path)

def convert_yolo_to_xywh(box, img_width, img_height):
    # convert yolo format to x, y, width, height
    x = (box[0] - box[2]/2) * img_width
    if x < 1e-3:
        x = 0
    y = (box[1] - box[3]/2) * img_height
    if y < 1e-3:
        y = 0
    width = box[2] * img_width
    height = box[3] * img_height
    return [x, y, width, height]


def visualize_predictions(image_path, predictions_path, output_path):
    image_files = list(image_path.glob('*.png'))
    for file in tqdm(image_files):
        name = file.stem
        image = Image.open(file)
        img_width, img_height = image.size
        draw = ImageDraw.Draw(image)
        with open(predictions_path / f'{name}.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                category = int(line[0])
                box = [float(x) for x in line[1:5]]
                box = convert_yolo_to_xywh(box, img_width, img_height)
                category = int(line[0])
                color = {
                    0: 'red',
                    1: 'green',
                    2: 'blue',
                    3: 'yellow',
                    4: 'purple',
                }
                draw.rectangle([box[0], box[1], box[0] + box[2], box[1] + box[3]], outline=color[category], width=3)
        image.save(output_path / f'{name}.png')
    
visualize_predictions(image_path, labels_path, output_path)