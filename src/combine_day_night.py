from pathlib import Path
from tqdm import tqdm
import shutil

input_A = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/grayscale/A')
input_N = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/grayscale/N')

output_path = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/grayscale/AN')

if not (output_path / 'train' / 'images').exists():
    (output_path / 'train' / 'images').mkdir(parents=True)   
if not (output_path / 'val' / 'images').exists():
    (output_path / 'val' / 'images').mkdir(parents=True)
if not (output_path / 'train' / 'labels').exists():
    (output_path / 'train' / 'labels').mkdir(parents=True)
if not (output_path / 'val' / 'labels').exists():
    (output_path / 'val' / 'labels').mkdir(parents=True)
    
train_img = list(input_A.glob('train/images/*.png')) + list(input_N.glob('train/images/*.png'))
val_img = list(input_A.glob('val/images/*.png')) + list(input_N.glob('val/images/*.png'))
train_label = list(input_A.glob('train/labels/*.txt')) + list(input_N.glob('train/labels/*.txt'))
val_label = list(input_A.glob('val/labels/*.txt')) + list(input_N.glob('val/labels/*.txt'))
for img in tqdm(train_img):
    shutil.copy(img, output_path / 'train' / 'images')
for img in tqdm(val_img):
    shutil.copy(img, output_path / 'val' / 'images')
for label in tqdm(train_label):
    shutil.copy(label, output_path / 'train' / 'labels')
for label in tqdm(val_label):
    shutil.copy(label, output_path / 'val' / 'labels')
    
