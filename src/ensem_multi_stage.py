import json

from ensemble_boxes import *
from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np
from tqdm import tqdm
from nms import greedy_nms, nms

def get_image_Id(img_name):
    img_name = img_name.split(".png")[0]
    sceneList = ["M", "A", "E", "N"]
    cameraIndx = int(img_name.split("_")[0].split("camera")[1])
    sceneIndx = sceneList.index(img_name.split("_")[1])
    frameIndx = int(img_name.split("_")[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId


def convert_bbox(bbox, width, height):
    return [
        bbox[0] / width,
        bbox[1] / height,
        (bbox[0] + bbox[2]) / width,
        (bbox[1] + bbox[3]) / height,
    ]
    
def load_predictions(input_path, image_path):
    bbox_dict = {}
    scores_dict = {}
    labels_dict = {}
    width_dict = {}
    height_dict = {}
    ensem_num = len(input_path)
    for img in image_path.glob("*.png"):
        bbox_dict[get_image_Id(img.stem)] = [[] for _ in range(ensem_num)]
        scores_dict[get_image_Id(img.stem)] = [[] for _ in range(ensem_num)]
        labels_dict[get_image_Id(img.stem)] = [[] for _ in range(ensem_num)]
        width, height = Image.open(img).size
        width_dict[get_image_Id(img.stem)] = width
        height_dict[get_image_Id(img.stem)] = height
    for idx, pred in tqdm(enumerate(input_path)):
        with open(pred) as f:
            data = json.load(f)
        for item in data:
            # if idx in [1,2] and item["category_id"] in [0, 4] and item["score"] < 0.2:
            #     continue
            bbox_dict[item["image_id"]][idx].append(
                convert_bbox(
                    item["bbox"],
                    width_dict[item["image_id"]],
                    height_dict[item["image_id"]],
                )
            )
            scores_dict[item["image_id"]][idx].append(item["score"])
            labels_dict[item["image_id"]][idx].append(item["category_id"])
        print(f"Loaded {pred}")
    return bbox_dict, scores_dict, labels_dict, width_dict, height_dict

def ensemble(bbox_dict,
            scores_dict, 
            labels_dict, 
            width_dict, 
            height_dict, 
            vis=False,
            iou_thr=0.65,
            skip_box_thr=0,
            mode = "avg",
            weights=[1,1],
            result_path=Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/tuongbck/ensemble"),
            color = {
                0: "red",
                1: "green",
                2: "blue",
                3: "yellow",
                4: "purple",
            },
            name="predictions_codetr.json"):
    result = []

    for image_path in tqdm(test_path.glob("*.png")):
        image_id = image_path.stem
        id = get_image_Id(image_path.stem)
        boxes_list = bbox_dict[id]
        scores_list = scores_dict[id]
        labels_list = labels_dict[id]
        width_list = width_dict[id]
        height_list = height_dict[id]

        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
            conf_type=mode,
            allows_overflow=False,
        )
        
        for i in range(len(boxes)):
            boxes[i][0] *= width_list
            boxes[i][1] *= height_list
            boxes[i][2] *= width_list
            boxes[i][3] *= height_list

        # Visualize the result
        
        if vis:
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)
            for i,box in enumerate(boxes):
                # if scores[i] < 0.03:
                draw.rectangle(box.tolist(), outline=color[int(labels[i])], width=2)
                # draw.text((box[0], box[1]), f"{int(labels[i])}", fill=color[int(labels[i])])
            image.save(result_path / "images" / f"{image_id}.png")

        for i in range(len(boxes)):
            boxes[i] = [
                boxes[i][0],
                boxes[i][1],
                boxes[i][2] - boxes[i][0],
                boxes[i][3] - boxes[i][1],
            ]

        for i in range(len(boxes)):
            # print(f"{id} {labels[i]} {boxes[i].tolist()} {scores[i]}")
            # if scores[i] > 0.1:
            result.append(
                {
                    "image_id": id,
                    "category_id": int(labels[i]),
                    "bbox": boxes[i].tolist(),
                    "score": scores[i],
                }
            )
    with open(result_path / name, "w") as f:
        json.dump(result, f)

codetr_path_list = [
    Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions_codetr_0_001.json"),
    # Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/predictions_yolo_0.65_0.03.json"),
    # Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/baotg/ensemble/yolo80.json"),
    # Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/predictions_codetr_day_night.json"),
    Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/predictions_codetr_pretrain_finetune.json"),
]

lower_path_list = [
    Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/predictions_yolo_0.65_0.03.json"),
    Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/baotg/ensemble/yolo80.json"),
    Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/predictions_codetr_day_night.json"),
]

test_path = Path("/mlcv1/Datasets/NvidiaAICityChallenge/NvidiaAIC2024/images")

color = {
    0: "red",
    1: "green",
    2: "blue",
    3: "yellow",
    4: "purple",
}


result_path = Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/tuongbck/ensemble")
result_path.mkdir(exist_ok=True, parents=True)
(result_path / "images").mkdir(exist_ok=True, parents=True)


vis = False
iou_thr = 0.65
nms_thr = 1
skip_box_thr = 0
sigma = 0.01
weights = [1,1]

# weights = [3,2,2]

imbalance_class = [0,4]
imbalance_class_threshold = [0.01,0.01]

car_threshold = 0.01
bike_threshold = 0.03
perdestrian_threshold = 0.03

print("Stage 1: ")

bbox_dict, scores_dict, labels_dict, width_dict, height_dict = load_predictions(
    codetr_path_list, test_path
)

ensemble(bbox_dict, scores_dict, labels_dict, width_dict, height_dict, vis=False, name="predictions_codetr.json",weights=[1,1])      

bbox_dict, scores_dict, labels_dict, width_dict, height_dict = load_predictions(
    lower_path_list, test_path
)

ensemble(bbox_dict, scores_dict, labels_dict, width_dict, height_dict, vis=False, name="predictions_lower.json",weights=[2,1,2])

stage2_path = [
    Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/tuongbck/ensemble/predictions_codetr.json"),
    Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/tuongbck/ensemble/predictions_lower.json"),
]

print("Stage 2: ")
bbox_dict, scores_dict, labels_dict, width_dict, height_dict = load_predictions(
    stage2_path, test_path
)

ensemble(bbox_dict, scores_dict, labels_dict, width_dict, height_dict, vis=False, name="predictions.json",weights=[2,1],mode="avg")



