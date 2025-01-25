import os 
import json
import glob

from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw

image_path = Path('/mlcv1/Datasets/NvidiaAICityChallenge/NvidiaAIC2024/images')
labels_path = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/visualize/preds')
output_path = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/visualize_codetr_0.1')
output_path.mkdir(parents=True, exist_ok=True)

def visualize_predictions(image_path, predictions_path, output_path):
    image_files = list(image_path.glob('*.png'))
    print(len(image_files))
    color = {
        0: 'red',
        1: 'green',
        2: 'blue',
        3: 'yellow',
        4: 'purple',
    }
    conf_thres = 0.1
    for file in tqdm(image_files):
        name = file.stem
        image = Image.open(file)
        img_width, img_height = image.size
        draw = ImageDraw.Draw(image)
        with open(predictions_path / f'{name}.json', 'r') as f:
            f = json.load(f)
            labels = f['labels']
            boxes = f['bboxes']
            scores = f['scores']
            for i,score in enumerate(scores):
                if score > conf_thres:
                    category = int(labels[i])
                    draw.rectangle(boxes[i], outline=color[category], width=3)
        image.save(output_path / f'{name}.png')
    
visualize_predictions(image_path, labels_path, output_path)
