import os
import json
import glob

from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw

image_path = Path("/mlcv1/Datasets/NvidiaAICityChallenge/NvidiaAIC2024/images")
# image_path = Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/data/yolo_all_classes_91/test_10")

labels_path = Path(
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions_codetr.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/baotg/classifications/source/codetr_beit.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/2024_03_09T14_37_49_124633_st4r3.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/yolor/predictions_moe.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/tuongbck/ensemble/predictions.json"
    # "/mlcv1/WorkingSpace/Personal/hienht/aic24/mmdetection/pred_dino_origin.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/predictions_tuongbck_iy5yw.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_original_val_10.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_original_val_10.json"
    "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/tuongbck/ensemble/predictions_best_ensemble.json"
)

output_path = Path(
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/visualize_coco_codetr"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/visualizations/highest_6102"
    "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/visualize_coco"
)
conf = 0.3

if not os.path.exists(output_path):
    os.makedirs(output_path)

# def convert_yolo_to_xywh(box, img_width, img_height):
#     # convert yolo format to x, y, width, height
#     x = (box[0] - box[2]/2) * img_width
#     if x < 1e-3:
#         x = 0
#     y = (box[1] - box[3]/2) * img_height
#     if y < 1e-3:
#         y = 0
#     width = box[2] * img_width
#     height = box[3] * img_height
#     return [x, y, width, height]


def get_image_Id(img_name):
    img_name = img_name.split(".png")[0]
    sceneList = ["M", "A", "E", "N"]
    cameraIndx = int(img_name.split("_")[0].split("camera")[1])
    sceneIndx = sceneList.index(img_name.split("_")[1])
    frameIndx = int(img_name.split("_")[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId


def visualize_predictions(image_path, predictions_path, output_path):
    color = {
        0: "red",
        1: "green",
        2: "blue",
        3: "yellow",
        4: "purple",
    }
    # map_dict = {0: 3, 1: 1, 3: 1, 2: 2, 5: 0, 7: 4}
    map_dict = {
        v: v for v in range(5)
    }
    image_files = list(image_path.glob("*.png"))
    predictions = json.load(open(predictions_path, "r"))
    for file in tqdm(image_files):
        name = file.stem
        image = Image.open(file)
        draw = ImageDraw.Draw(image)
        for pred in predictions:
            if (
                pred["image_id"] == get_image_Id(name)
                and pred["category_id"] in list(map_dict.keys())
                and pred["score"] > conf
            ):
                x, y, width, height = pred["bbox"]
                x1, y1, x2, y2 = x, y, x + width, y + height
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline=color[map_dict[pred["category_id"]]],
                    width=2,
                )
                # draw.text((x1, y1), f'{pred["category_id"]}', fill='red')
        image.save(output_path / f"{name}.png")


visualize_predictions(image_path, labels_path, output_path)
