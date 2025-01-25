from pathlib import Path
from PIL import Image, ImageFilter
from tqdm import tqdm
import numpy as np
import shutil

image_path = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/data/yolo_all_classes/train/images')
label_path = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/data/yolo_all_classes/train/labels')

output_path = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/data/yolo_all_classes_gray/train/images')
label_output_path = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/data/yolo_all_classes_gray/train/labels')

if not output_path.exists():
    output_path.mkdir(parents=True)

if not label_output_path.exists():
    label_output_path.mkdir(parents=True)

def day2night(img):
    arrImg = np.array(img.convert('RGB'))
    arr = arrImg * np.array([0.08, 0.1, 0.17])
    ret = (255*arr/arr.max()).astype(np.uint8)
    ret = Image.fromarray(ret)
    ret = ret.filter(ImageFilter.GaussianBlur(1.3))
    return ret

def convert_to_grayscale(image_path, output_path):
    image_files = list(image_path.glob('*.png'))
    for file in tqdm(image_files):
        name = file.stem
        if name.split('_')[1] == 'N':
            image = Image.open(file)
            image = image.convert('L')
            image.save(output_path / f'{name}.png')
            if 'test' not in str(image_path):
                shutil.copy(label_path / f'{name}.txt', output_path / f'{name}.txt')
        else:
            image = day2night(Image.open(file))
            image = image.convert('L')
            image.save(output_path / f'{name}.png')
            if 'test' not in str(image_path):
                shutil.copy(label_path / f'{name}.txt', output_path / f'{name}.txt')
        

convert_to_grayscale(image_path, output_path)
