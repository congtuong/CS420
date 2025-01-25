from sahi.model import Yolov6DetectionModel
from sahi.predict import get_prediction, get_sliced_prediction, predict
from pathlib import Path
import json
import cv2


def get_image_Id(img_name):
    img_name = img_name.split('.png')[0]
    sceneList = ['M', 'A', 'E', 'N']
    cameraIndx = int(img_name.split('_')[0].split('camera')[1])
    sceneIndx = sceneList.index(img_name.split('_')[1])
    frameIndx = int(img_name.split('_')[2])
    imageId = int(str(cameraIndx)+str(sceneIndx)+str(frameIndx))
    return imageId

    
mapping_colors = {
    0: (0, 0, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (255, 255, 0),
    4: (0, 255, 255),
}


detection_model = Yolov6DetectionModel(
    model_path='/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/YOLOv6/runs/train/baotg_80_classes_sliced_640/exp6/weights/best_ckpt.pt',
    confidence_threshold=0.1,
    device="cuda:0", # or 'cuda:0'
    image_size=640,

    category_mapping={
        # "0": "bus",
        # "1": "bike",
        # "2": "car",
        # "3": "perdestrian",
        # "4": "truck",
        "5": "bus",
        "1": "bike",
        "3": "bike",
        "2": "car",
        "0": "perdestrian",
        "7": "truck",
    }
)

mapping = {
    5: 0,
    1: 1,
    3: 1,
    2: 2,
    0: 3,
    7: 4
}

#result = get_prediction("demo/demo_data/highway.jpg", detection_model)

root_img = Path('/mlcv1/Datasets/NvidiaAICityChallenge/NvidiaAIC2024/images/')
results = []

for idx, p_img in enumerate(root_img.glob("*.png")):
    img = cv2.imread(str(p_img))
    old_img = img.copy()

    old_shape = img.shape
    img = cv2.resize(img, (1230, int(1230 * img.shape[0] / img.shape[1])))

    # ratio = (img.shape[0] / old_shape[0], img.shape[1] / old_shape[1])
    ratio = (old_shape[0] / img.shape[0], old_shape[1] / img.shape[1])

    result = get_sliced_prediction(
        img,
        detection_model,
        slice_height=480,
        slice_width=480,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    data = result.object_prediction_list
    for object_prediction in data:
        coco_prediction = object_prediction.to_coco_prediction()
        coco_prediction.image_id = get_image_Id(p_img.name)
        coco_prediction_json = coco_prediction.json

        # convert the bbox to bbox in original image
        bbox = coco_prediction_json['bbox']
        bbox[0] = bbox[0] * ratio[1]
        bbox[1] = bbox[1] * ratio[0]
        bbox[2] = bbox[2] * ratio[1]
        bbox[3] = bbox[3] * ratio[0]
        coco_prediction_json['bbox'] = bbox

        coco_prediction_json.pop('category_name')
        coco_prediction_json.pop('segmentation')
        coco_prediction_json.pop('iscrowd')
        coco_prediction_json.pop('area')
        coco_prediction_json['category_id'] = int(mapping[coco_prediction_json['category_id']])

        # if coco_prediction_json['score'] < 0.3:
        #     continue
        
        results.append(coco_prediction_json)

        # print(coco_prediction_json)
        # input()

        cv2.rectangle(
            old_img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
            mapping_colors[coco_prediction_json['category_id']],
            2
        )

    cv2.imwrite(f"demo_data/{p_img.name}", old_img)
    # result.export_visuals(export_dir="demo_data/", file_name=p_img.name)

    if (idx + 1) % 100 == 0:
        print(f"Processed {idx + 1} images")

json.dump(results, open('results_sahi.json', 'w+'))