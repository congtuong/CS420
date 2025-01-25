import json
from pathlib import Path
from tqdm import tqdm

input_json = Path(
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/visualize_sahi_square/preds"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/visualize_sahi_woodscape_finetune_new_pipeline/preds"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/visualize_sahi_woodscape_finetune_1280_ratio/preds"
    # "/mlcv1/WorkingSpace/Personal/hienht/aic24/mmdetection/ddq_prediction/preds"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/visualize_codetr_closs/preds"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/visualize_sahi_codetr_closs/preds"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/predictions/codetr_max_91_epoch20/preds"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/predictions/codetr_closs_val/preds"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/predictions/codetr_original_val/preds"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/predictions/codetr_pretrain_val/preds"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/predictions/codetr_pretrain_sahi_val/preds"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/predictions/codetr_original_day_val/preds"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/predictions/codetr_original_val_10_sahi/preds"
    "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/predictions/codetr_original_val_10_day_sahi/preds"
)
output_submit = Path(
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_original_sahi_1080.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_woodscape_finetune_1280.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_woodscape_finetune_1280_ratio.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/ddq.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_closs.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_sahi_closs.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_max_91_epoch20.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_closs_val.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_original_val.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_pretrain_sahi_val.json"
    # "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_original_day_val.json"
    "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/codetr_original_val_10_day_sahi.json"
)
conf_thes = 0.001


def get_image_Id(img_name):
    img_name = img_name.split(".png")[0]
    sceneList = ["M", "A", "E", "N"]
    cameraIndx = int(img_name.split("_")[0].split("camera")[1])
    sceneIndx = sceneList.index(img_name.split("_")[1])
    frameIndx = int(img_name.split("_")[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId


json_list = input_json.glob("*.json")
submit = []

for j in tqdm(json_list):
    name = j.stem.replace(".png", "")
    file = json.load(open(j, "r"))
    scores = file["scores"]
    labels = file["labels"]
    boxes = file["bboxes"]
    for i, score in enumerate(scores):
        if score > conf_thes:
            submit.append(
                {
                    "image_id": get_image_Id(name),
                    "category_id": int(labels[i]),
                    "bbox": [
                        boxes[i][0],
                        boxes[i][1],
                        boxes[i][2] - boxes[i][0],
                        boxes[i][3] - boxes[i][1],
                    ],
                    "score": round(float(score), 5),
                }
            )

with open(output_submit, "w") as f:
    json.dump(submit, f)
    # for checking the result
    # json.dump(submit, f, indent=4)
print("Done")
