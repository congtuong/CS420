import streamlit as st
import sys
from PIL import Image
import numpy as np
import random
import cv2
import torch
import torchvision.transforms as transforms
sys.path.insert(0, str(Path(__file__).parent.parent / "training" / "CoDETR"))

from mmdet.apis import (
    inference_detector,
    init_detector,
)

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def infer_codetr(image, model):
    """
    Inference using Co-DETR model

    Args:
        args: parsed arguments
        image: image to be processed

    Returns:
        list of results
    """

    results = []

    id_img = 0
    predictions = inference_detector(model, image)
    # print(predictions)
    
    detections = []
    cls_preds = []
    results_scores = []

    for box_result in predictions:
        box_result = box_result.pred_instances
        bboxes = box_result.bboxes.cpu().numpy()
        classes = box_result.labels.cpu().numpy()
        scores = box_result.scores.cpu().numpy()
        
        for j in range(len(classes)):
            # results.append(
            #     {
            #         "image_id": img_path,
            #         "category_id": classes[j],
            #         "bbox": convert_to_yolo_format(bboxes[j], (height, width)),
            #         "score": scores[j],
            #     }
            # )
            detections.append(bboxes[j])
            cls_preds.append(classes[j])
            results_scores.append(scores[j])

        # for class_id in range(4):
        #     bboxes = box_result[class_id]

        #     for bbox in bboxes:
        #         x1, y1, x2, y2, score = bbox

        #         if score < args.score_thr:
        #             continue

        #         results.append(
        #             {
        #                 "image_id": img_path,
        #                 "category_id": class_id,
        #                 "bbox": convert_to_yolo_format(
        #                     [x1, y1, x2, y2], (height, width)
        #                 ),
        #                 "score": score,
        #             }
        #         )

    return detections, cls_preds, results_scores


def draw_detections(image, detections, classes, scores, conf_thres=0.2):
    """
    detection have: [[x1, y1, x2, y2]]
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if detections is not None:
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)

        # Bounding-box colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
        bbox_colors = random.sample(colors, n_cls_preds)

        # Draw bounding boxes and labels of detections
        for bbox, cls_pred, score in zip(detections, classes, scores):
            if score < conf_thres:
                continue
            x1, y1, x2, y2 = bbox[:4]
            box_w = x2 - x1
            box_h = y2 - y1

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            bbox = patches.Rectangle(
                (x1, y1),
                box_w,
                box_h,
                linewidth=2,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(bbox)
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )
    plt.axis("off")
    return fig

config = "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/work_dirs/co_dino_original/co_dino_original.py"
checkpoint = "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/CoDETR/mmdetection/work_dirs/co_dino_original/epoch_16.pth"

codert_model = init_detector(config, checkpoint, device="cuda:0")
print(codert_model)
classes = ["bus", "bike", "car", "pedestrian", "truck"]

st.title("Fisheye Object Detection")

conf_thres = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
nms_thres = st.slider("NMS Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.05)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    image = cv2.imread(uploaded_file)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detections, result_classes, result_scores = infer_codetr(img, codert_model)

    fig = draw_detections(img, detections, result_classes, result_scores, conf_thres=conf_thres)

    st.pyplot(fig)