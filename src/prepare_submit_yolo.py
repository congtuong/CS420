import glob
import json
import os

from tqdm import tqdm
from PIL import Image
from pathlib import Path

input = Path(
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/runs/inference/1920_night_80e/labels"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/runs/inference/1920_91/labels"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/yolor/runs/inference/test_1280_val_10/test_1280_val_10/labels"
    "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/runs/inference/test_1920_val_10_day/labels"
)

output = Path(
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/yolov6l6_1920_91.json"
    "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/test_yolov6_1920_val_10_day.json"
)


def get_image_Id(img_name):
    img_name = img_name.split(".png")[0]
    sceneList = ["M", "A", "E", "N"]
    cameraIndx = int(img_name.split("_")[0].split("camera")[1])
    sceneIndx = sceneList.index(img_name.split("_")[1])
    frameIndx = int(img_name.split("_")[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId


def convert_yolo_to_xywh(box, img_width, img_height):
    # convert yolo format to x, y, width, height
    x = (box[0] - box[2] / 2) * img_width
    if x < 1e-3:
        x = 0
    y = (box[1] - box[3] / 2) * img_height
    if y < 1e-3:
        y = 0
    width = box[2] * img_width
    height = box[3] * img_height
    return [x, y, width, height]


def main():
    input_files = list(input.glob("*.txt"))
    predictions = []
    map_dict = {0: 3, 1: 1, 3: 1, 2: 2, 5: 0, 7: 4}
    for file in tqdm(input_files):
        name = file.stem
        with open(file, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                imageId = get_image_Id(name)
                category = int(line[0])

                if category not in map_dict:
                    print("Category not found")
                    continue
                
                box = [float(x) for x in line[1:]]
                image = Image.open(input.parent / f"{name}.png")
                img_width, img_height = image.size
                box = convert_yolo_to_xywh(box, img_width, img_height)
                score = float(line[-1])
                prediction = {
                    "image_id": imageId,
                    "category_id": map_dict[category],
                    "bbox": box,
                    "score": score,
                }
                predictions.append(prediction)
    json.dump(predictions, open(output, "w+"), indent=4)


if __name__ == "__main__":
    main()
