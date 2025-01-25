import os 
import glob

from pathlib import Path
from tqdm import tqdm
input_path = Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/splited')

def count_max_det(input_path):
    max_det = 0
    for _time in tqdm(input_path.glob('*')):
        for _det in tqdm(_time.glob('val/labels/*.txt')):
            with open(_det, 'r') as file:
                lines = file.readlines()
                max_det = max(max_det, len(lines))
    return max_det

print(count_max_det(input_path))
