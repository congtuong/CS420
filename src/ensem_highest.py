import json

from ensemble_boxes import *
from PIL import Image, ImageDraw
from pathlib import Path
import numpy as np
from tqdm import tqdm

# from nms import greedy_nms, ori_nms


# 0.2731, iou = 0.65, conf = 0.4, skip = 0.25
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions_codetr_0_001.json"),
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/predictions_yolo_0.65_0.03.json" # ),
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/baotg/ensemble/yolo80.json"),
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/predictions_codetr_day_night.json" # ),
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/predictions_codetr_pretrain_finetune.json" # ),
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_original_sahi_1080.json" # ),
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_woodscape_finetune_1280_ratio.json" # ),
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_closs.json"),

# 0.2761, iou = 0.65, conf = 0.4, skip = 0.1
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions_codetr_0_001.json"),
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/predictions_codetr_day_night.json")
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/predictions_codetr_pretrain_finetune.json")
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_original_sahi_1080.json")
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_woodscape_finetune_1280_ratio.json")
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_closs.json")
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions/predictions_fisheye_an_finetune_custom_120.json")
# Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions/yolor-test_0.65_.03_best_f.json")

input_path_list = [
    # CO-DETR original 16e
    # ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions_codetr_0_001.json", 2),
    # ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_original_max_91.json", 1),
    # ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_original_300objects.json", 1),
    # ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/baotg/classifications/source/2024_03_09T14_37_49_124633_st4r3.json", 1),
    # ('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/baotg/classifications/source/codetr_highest_processed.json', 1),
    (
        "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_max_91_epoch20.json",
        1,
    ),
    (
        "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_split_1_epoch16.json",
        2,
    ),
    # CO-DETR train on day / night 20epochs
    (
        "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/predictions_codetr_day_night.json",
        1,
    ),
    # Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/baotg/classifications/source/predictions_codetr_day_night_processed.json'),
    # CO-DETR pretrain Woodscape 10e -> finetune 20e
    (
        "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/predictions_codetr_pretrain_finetune.json",
        1,
    ),
    # sahi width=1080, height=1080, CO-DETR original 16e
    # ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_original_sahi_1080.json", 1),
    # sahi width=1280, height keep ratio, CO-DETR pretrain Woodscape 10e -> finetune 20e
    (
        "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_woodscape_finetune_1280_ratio.json",
        2,
    ),
    # CO-DETR original with Custom Loss, 16e
    ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_closs.json", 1),
    # YOLOR, 1920
    ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/yolor_1920.json", 2),
    # ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/yolov6l6_1920_91.json", 2),
    # yolov6l6 with only AN dataset
    # (
    #     "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions/predictions_fisheye_an_finetune_custom_120.json",
    #     1,
    # ),
    (
        "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/yolov6l6_1920_epoch80_day_night.json",
        1,
    ),
    (
        "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/yolor_1920_epoch80_day_night.json",
        1,
    ),
    # yolor 80 class best_f.pt checkpoint
    # ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions/yolor-test_0.65_.03_best_f.json", 1, 0.3),
    # ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions/yolor-test_0.65_.03_best_f.json", 1),
    # ("/mlcv1/WorkingSpace/Personal/phatnc/submit_yolov9_ttam.json", 2),
    # dino
    # ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/hienht/mmdetection/predictions/dino_e31.json", 1),
    # deta
    # ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/hienht/detrex/prediction_deta.json", 1),
    # ("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/prediction_deta_180k.json", 1),
    # Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_square_e10.json')
    # ddq lor
    # ('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/ddq.json', 1)
    # Khiem
    # Path(
    #     "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/baotg/classifications/source/2024_03_09T14_37_49_124633_st4r3.json"
    # ),
    # yolor 80 class last.pt checkpoint
    # Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions/yolor-test_0.65_.03_last.json")
    # yolov6l6 full data
    # Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/predictions/predictions_yolov6_all_data_5class.json")
    # original codetr sahi
    # Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/tuongbck/sahi/results_sahi_co_dino_original.json")
    # codetr square no overlap
    # Path("/mlcv1/WorkingSpac//Personal/tuongbck/AIC2024/tuongbck/sahi/results_sahi_co_dino_12g_train_square.json")
    # codetr square overlap 0.2
    # Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/tuongbck/sahi/results_sahi_co_dino_12g_train_square_overlap_02.json")
    # Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/confluence/predictions_yolo_0.65_0.03.json')
    # -------------------------
    # Path("/mlcv1/WorkingSpace/Personal/hienht/aic24/mmdetection/beit_nms.json"),
    # Path(
    #     "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_woodscape_finetune_1080.json"
    # ),
    # Path(
    #     "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_woodscape_finetune_1280.json"
    # ),
    # Path("/mlcv1/WorkingSpace/Personal/hienht/aic24/mmdetection/pred_dino_origin.json"),
    # Path("/mlcv1/WorkingSpace/Personal/hienht/aic24/NWD/result_no_thr.json"),
    # Path('/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/baotg/classifications/source/hightest_beit_processed.json')
]

# NOTE: highest score, iou 1.65, conf 0.3, after conf 0.4, 7 models dau tien --> F1 = 0.273

# min_size = 10
vis = False
after_conf = 0.1
iou_thr = 0.65
nms_thr = 0.5
skip_box_thr = 0.1
conf_list = []

# conf_list = [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]
test_path = Path("/mlcv1/Datasets/NvidiaAICityChallenge/NvidiaAIC2024/images")

color = {
    0: "red",
    1: "green",
    2: "blue",
    3: "yellow",
    4: "purple",
}


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


def xywh_2_yolo(bbox, width, height):
    x = bbox[0] + bbox[2] / 2
    y = bbox[1] + bbox[3] / 2
    w = bbox[2]
    h = bbox[3]
    return x / width, y / height, w / width, h / height


def calculate_iou(box1, box2):
    intersect_box = [
        max(box1[0], box2[0]),
        max(box1[1], box2[1]),
        min(box1[2], box2[2]),
        min(box1[3], box2[3]),
    ]
    intersect_area = max(intersect_box[2] - intersect_box[0], 0) * max(
        intersect_box[3] - intersect_box[1], 0
    )
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersect_area / (area1 + area2 - intersect_area)
    return iou, area1, area2, intersect_area


def calculate_distance(middle, box):
    box_middle = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    return ((middle[0] - box_middle[0]) ** 2 + (middle[1] - box_middle[1]) ** 2) / (
        middle[0] ** 2 + middle[1] ** 2
    )


def l2_distance(middle, box):
    box_middle = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

    return ((middle[0] - box_middle[0]) ** 2 + (middle[1] - box_middle[1]) ** 2) ** 0.5


def load_predictions(input_path, image_path, conf_thres):
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

    for idx, pred in enumerate(input_path):
        with open(pred[0]) as f:
            data = json.load(f)

        for item in data:
            if len(pred) == 3 and item["score"] < pred[2]:
                continue

            # if idx >= len(conf_thres) or item["score"] > conf_thres[idx]:
            weight = 1

            bbox_dict[item["image_id"]][idx].append(
                convert_bbox(
                    item["bbox"],
                    width_dict[item["image_id"]],
                    height_dict[item["image_id"]],
                )
            )

            # item["score"] * weight * distance
            scores_dict[item["image_id"]][idx].append(item["score"] * weight)

            labels_dict[item["image_id"]][idx].append(item["category_id"])

        # for k in bbox_dict.keys():
        #     bbox = bbox_dict[k][idx]
        #     scores = scores_dict[k][idx]
        #     labels = labels_dict[k][idx]

        # boxes, scores, labels = weighted_boxes_fusion(
        #     [bbox],
        #     [scores],
        #     [labels],
        #     # weights=[
        #     #     t[1] for t in input_path_list
        #     # ],
        #     iou_thr=iou_thr,
        #     # skip_box_thr=skip_box_thr,
        #     conf_type="avg",
        #     allows_overflow=False,
        # )

        # bbox_dict[k][idx] = boxes
        # scores_dict[k][idx] = scores
        # labels_dict[k][idx] = labels

        print(f"Loaded {pred}")

    return bbox_dict, scores_dict, labels_dict, width_dict, height_dict


bbox_dict, scores_dict, labels_dict, width_dict, height_dict = load_predictions(
    input_path_list, test_path, conf_list
)

result_path = Path("/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/tuongbck/ensemble")
result_path.mkdir(exist_ok=True, parents=True)
(result_path / "images").mkdir(exist_ok=True, parents=True)

result = []

sigma = 0.1
imbalance_class = [0, 4]
imbalance_class_threshold = [0.03, 0.03]


for image_path in tqdm(list(test_path.glob("*.png"))):
    image_id = image_path.stem
    id = get_image_Id(image_path.stem)
    boxes_list = bbox_dict[id]
    scores_list = scores_dict[id]
    labels_list = labels_dict[id]
    width_list = width_dict[id]

    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=[t[1] for t in input_path_list],
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
        # conf_type="avg",
        # conf_type="avg",
        conf_type="box_and_model_avg",
        allows_overflow=False,
    )

    # Visualize the result

    # boxes, scores, labels = weighted_boxes_fusion(
    #     [boxes], [scores], [labels], weights=None, iou_thr=0.65
    # )

    for i in range(len(boxes)):
        boxes[i][0] *= width_dict[id]
        boxes[i][1] *= height_dict[id]
        boxes[i][2] *= width_dict[id]
        boxes[i][3] *= height_dict[id]

    # Class names  Bus  Bike  Car  Pedestrian  Truck
    # Category ID  0  1      2  3          4
    # calculate the IOU, if exceed the 0.95 then take the largest score one

    # remove_indices = set()
    # for i in range(len(boxes)):
    #     for j in range(len(boxes)):
    #         if i == j:
    #             continue
    #         if i in remove_indices or j in remove_indices:
    #             continue

    #         iou, area1, area2, intersect_area = calculate_iou(boxes[i], boxes[j])
    #         # if iou > 0.999 and iou<1:
    #         #     if scores[i] > scores[j]:
    #         #         remove_indices.add(j)
    #         #     else:
    #         #         remove_indices.add(i)
    #         #     continue

    #         if iou > 0.95 and iou<1:
    #             if intersect_area / area1 > 0.9 or intersect_area / area2 > 0.9:
    #                 if scores[i] > scores[j]:
    #                     remove_indices.add(j)
    #                 else:
    #                     remove_indices.add(i)
    #             # if car and bus take the car
    #             if labels[i] == 0 and labels[j] == 2 or labels[i] == 2 and labels[j] == 0:
    #                 # if labels[i] == 0:
    #                 #     remove_indices.add(i)
    #                 # else:
    #                 #     remove_indices.add(j)
    #                 if scores[i] > scores[j]:
    #                     remove_indices.add(j)
    #                 else:
    #                     remove_indices.add(i)

    #             # if car and truck take largest score
    #             if labels[i] == 2 and labels[j] == 4 or labels[i] == 4 and labels[j] == 2:
    #                 if scores[i] > scores[j]:
    #                     remove_indices.add(j)
    #                 else:
    #                     remove_indices.add(i)
    #         # check if a box is inside another box, take the larger score
    #                 # if labels[i] == 2 and scores[i] > scores[j]:
    #                 #     scores[i] += scores[j]
    #                 #     remove_indices.add(j)
    #                 # else:
    #                 #     remove_indices.add(i)
    #                 # continue
    # #             # in car and car take the largest score
    # #             if labels[i] == 2 and labels[j] == 2:
    # #                 if scores[i] > scores[j]:
    # #                     remove_indices.add(j)
    # #                 else:
    # #                     remove_indices.add(i)
    # #                 continue
    # #             # in car and bike take the largest score
    # #             if labels[i] == 2 and labels[j] == 1 or labels[i] == 1 and labels[j] == 2:
    # #                 if scores[i] > scores[j]:
    # #                     remove_indices.add(j)
    # #                 else:
    # #                     remove_indices.add(i)
    # #                 continue
    # #             # if bike and pedestrian take the largest score
    # #             if labels[i] == 1 and labels[j] == 3 or labels[i] == 3 and labels[j] == 1:
    # #                 if scores[i] > scores[j]:
    # #                     remove_indices.add(j)
    # #                 else:
    # #                     remove_indices.add(i)
    # #                 continue

    # remove_indices = list(remove_indices)
    # # print(f"Remove {len(remove_indices)} boxes")
    # boxes = np.delete(boxes, remove_indices, axis=0)
    # scores = np.delete(scores, remove_indices)
    # labels = np.delete(labels, remove_indices)

    remove_indices = set()
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i == j:
                continue
            if i in remove_indices or j in remove_indices:
                continue

            iou, area1, area2, intersect_area = calculate_iou(boxes[i], boxes[j])
            # if iou > 0.999 and iou<1:
            #     if scores[i] > scores[j]:
            #         remove_indices.add(j)
            #     else:
            #         remove_indices.add(i)
            #     continue
            if intersect_area / area1 > 0.8 or intersect_area / area2 > 0.8:
                if (scores[i] < 0.4 and scores[j] < 0.4) or (
                    labels[j] != labels[i] and iou > 0.5
                ):
                    if scores[i] > scores[j]:
                        remove_indices.add(j)
                    else:
                        remove_indices.add(i)

                if labels[i] == labels[j]:
                    if scores[i] > scores[j]:
                        remove_indices.add(j)
                    else:
                        remove_indices.add(i)

            if iou > 0.95 and iou < 1:

                # if car and bus take the car
                if (
                    labels[i] == 0
                    and labels[j] == 2
                    or labels[i] == 2
                    and labels[j] == 0
                ):
                    # if labels[i] == 0:
                    #     remove_indices.add(i)
                    # else:
                    #     remove_indices.add(j)
                    if scores[i] > scores[j]:
                        remove_indices.add(j)
                    else:
                        remove_indices.add(i)

                # if car and truck take largest score
                if (
                    labels[i] == 2
                    and labels[j] == 4
                    or labels[i] == 4
                    and labels[j] == 2
                ):
                    if scores[i] > scores[j]:
                        remove_indices.add(j)
                    else:
                        remove_indices.add(i)
            # check if a box is inside another box, take the larger score
            # if labels[i] == 2 and scores[i] > scores[j]:
            #     scores[i] += scores[j]
            #     remove_indices.add(j)
            # else:
            #     remove_indices.add(i)
            # continue
    #             # in car and car take the largest score
    #             if labels[i] == 2 and labels[j] == 2:
    #                 if scores[i] > scores[j]:
    #                     remove_indices.add(j)
    #                 else:
    #                     remove_indices.add(i)
    #                 continue
    #             # in car and bike take the largest score
    #             if labels[i] == 2 and labels[j] == 1 or labels[i] == 1 and labels[j] == 2:
    #                 if scores[i] > scores[j]:
    #                     remove_indices.add(j)
    #                 else:
    #                     remove_indices.add(i)
    #                 continue
    #             # if bike and pedestrian take the largest score
    #             if labels[i] == 1 and labels[j] == 3 or labels[i] == 3 and labels[j] == 1:
    #                 if scores[i] > scores[j]:
    #                     remove_indices.add(j)
    #                 else:
    #                     remove_indices.add(i)
    #                 continue

    remove_indices = list(remove_indices)
    # print(f"Remove {len(remove_indices)} boxes")
    boxes = np.delete(boxes, remove_indices, axis=0)
    scores = np.delete(scores, remove_indices)
    labels = np.delete(labels, remove_indices)

    # perfrom nms for each class
    # result_boxes = []
    # result_scores = []
    # result_labels = []
    # # _,_,indices = ori_nms(boxes, scores, nms_thr)
    # for label in range(5):
    #     indices = np.where(labels == label)[0]
    #     if len(indices) > 0:
    #         boxes_label = boxes[indices]
    #         scores_label = scores[indices]
    #         boxes_label, scores_label,indices = ori_nms(boxes_label, scores_label,nms_thr)
    #         result_boxes.extend(boxes_label)
    #         result_scores.extend(scores_label)
    #         result_labels.extend([label] * len(boxes_label))
    # boxes = np.array(result_boxes)
    # scores = np.array(result_scores)
    # labels = np.array(result_labels)

    # boxes_label, scores_label,indices = ori_nms(boxes, scores,0.4)

    # boxes = boxes[indices]
    # scores = scores[indices]
    # labels = labels[indices]

    # perform nms for bike
    # bike_indices = np.where(labels == 1)[0]
    # if len(bike_indices) > 0:
    #     bike_boxes = boxes[bike_indices]
    #     bike_scores = scores[bike_indices]
    #     bike_boxes, bike_scores, bike_indices = ori_nms(bike_boxes, bike_scores, 0.4)
    #     boxes = np.delete(boxes, bike_indices, axis=0)
    #     scores = np.delete(scores, bike_indices)
    #     labels = np.delete(labels, bike_indices)
    # boxes = np.concatenate((boxes, bike_boxes), axis=0)
    # scores = np.concatenate((scores, bike_scores), axis=0)
    # labels = np.concatenate((labels, [1] * len(bike_boxes)), axis=0)

    # for i in range(len(boxes)):

    # 0.3, 0.4 -> 0.2889
    # 0.25, 0.35 -> 0.288
    # 0.2, 0.3 -> 0.29

    bus_threshold = 0.30
    car_threshold = 0.20
    bike_threshold = 0.20
    perdestrian_threshold = 0.20
    truck_threshold = 0.30

    conf_list = [
        bus_threshold,
        bike_threshold,
        car_threshold,
        perdestrian_threshold,
        truck_threshold,
    ]

    remove_indices = set()

    for i in range(len(boxes)):
        if labels[i] == 0 and scores[i] < conf_list[0]:
            remove_indices.add(i)
        if labels[i] == 1 and scores[i] < conf_list[1]:
            remove_indices.add(i)
        if labels[i] == 2 and scores[i] < conf_list[2]:
            remove_indices.add(i)
        if labels[i] == 3 and scores[i] < conf_list[3]:
            remove_indices.add(i)
        if labels[i] == 4 and scores[i] < conf_list[4]:
            remove_indices.add(i)

    labels = np.delete(labels, list(remove_indices))
    scores = np.delete(scores, list(remove_indices))
    boxes = np.delete(boxes, list(remove_indices), axis=0)

    # only keep the box of imbalance class if the score is larger than threshold
    if vis:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        for i, box in enumerate(boxes):
            if scores[i] > after_conf:
                draw.rectangle(box.tolist(), outline=color[int(labels[i])], width=2)
                draw.text(
                    (box[0] + 1, box[1] + 1),
                    f"{scores[i]:.2f}",
                    fill=color[int(labels[i])],
                )
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
        if scores[i] > after_conf:
            score = scores[i]
            # if int(labels[i]) in imbalance_class:
            #     if score < 0.3:
            #         score = 1-score

            result.append(
                {
                    "image_id": id,
                    "category_id": int(labels[i]),
                    "bbox": boxes[i].tolist(),
                    "score": score,
                }
            )

    # yolo_res = []
    # for i in range(len(boxes)):
    #     if scores[i] > after_conf:
    #         yolo_res.append(
    #             str(int(labels[i]))
    #             + " "
    #             + " ".join(
    #                 [
    #                     str(x)
    #                     for x in xywh_2_yolo(boxes[i], width_dict[id], height_dict[id])
    #                 ]
    #             )
    #         )

    # with open(result_path / "yolo_labels" / f"{image_id}.txt", "w") as f:
    #     f.write("\n".join(yolo_res))

with open(result_path / "predictions_baotg_test_121.json", "w+") as f:
    json.dump(result, f)
