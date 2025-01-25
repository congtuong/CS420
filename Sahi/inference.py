from sahi.utils.yolov8 import download_yolov8s_model
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
from IPython.display import Image

# Download YOLOv8 model
yolov8_model_path = "models/yolov8s.pt"
download_yolov8s_model(yolov8_model_path)