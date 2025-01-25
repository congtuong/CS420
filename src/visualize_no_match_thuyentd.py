from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageDraw
import argparse
import json
import shutil
import os
import numpy as np
from matplotlib.patches import Ellipse
from shapely.geometry import Polygon, LineString, Point
from pprint import pprint

root_img_base = Path(
    "/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/visualize_highest_baseline"
)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--gt",
    type=str,
    default="/mlcv1/WorkingSpace/Personal/tuongbck/AIC2024/src/temp/predictions_baotg_6038.json",
)
parser.add_argument("--pred", type=str, required=True)
parser.add_argument("--conf", type=float, default=0)
args = parser.parse_args()


def calc_iou_individual(pred_box, gt_box):
    """Calculate IoU of single predicted and ground truth box
    Args:
        pred_box (list of floats): location of predicted object as
            [xmin, ymin, xmax, ymax]
        gt_box (list of floats): location of ground truth object as
            [xmin, ymin, xmax, ymax]
    Returns:
        float: value of the IoU for the two boxes.
    Raises:
        AssertionError: if the box is obviously malformed
    """
    x1_t, y1_t, x2_t, y2_t = gt_box
    x1_p, y1_p, x2_p, y2_p = pred_box

    if (x1_p > x2_p) or (y1_p > y2_p):
        raise AssertionError(
            "Prediction box is malformed? pred box: {}".format(pred_box)
        )

    if (x1_t > x2_t) or (y1_t > y2_t):
        raise AssertionError(
            "Ground Truth box is malformed? true box: {}".format(gt_box)
        )

    if x2_t < x1_p or x2_p < x1_t or y2_t < y1_p or y2_p < y1_t:
        return 0.0

    far_x = np.min([x2_t, x2_p])
    near_x = np.max([x1_t, x1_p])
    far_y = np.min([y2_t, y2_p])
    near_y = np.max([y1_t, y1_p])

    inter_area = (far_x - near_x + 1) * (far_y - near_y + 1)
    true_box_area = (x2_t - x1_t + 1) * (y2_t - y1_t + 1)
    pred_box_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    iou = inter_area / (true_box_area + pred_box_area - inter_area)
    return iou


def get_single_image_results(gt_boxes, pred_boxes, iou_thr=0.5):
    """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
    Args:
        gt_boxes (list of list of floats): list of locations of ground truth
            objects as [xmin, ymin, xmax, ymax]
        pred_boxes (dict): dict of dicts of 'boxes' (formatted like `gt_boxes`)
            and 'scores'
        iou_thr (float): value of IoU to consider as threshold for a
            true prediction.
    Returns:
        dict: true positives (int), false positives (int), false negatives (int)
    """

    all_pred_indices = range(len(pred_boxes))
    all_gt_indices = range(len(gt_boxes))

    if len(all_pred_indices) == 0:
        tp = 0
        fp = 0
        fn = len(gt_boxes)
        return {"true_pos": tp, "false_pos": fp, "false_neg": fn}

    if len(all_gt_indices) == 0:
        tp = 0
        fp = len(pred_boxes)
        fn = 0
        return {"true_pos": tp, "false_pos": fp, "false_neg": fn}

    gt_idx_thr = []
    pred_idx_thr = []
    ious = []
    for ipb, pred_box in enumerate(pred_boxes):
        for igb, gt_box in enumerate(gt_boxes):
            iou = calc_iou_individual(pred_box, gt_box)

            if iou > iou_thr:
                gt_idx_thr.append(igb)
                pred_idx_thr.append(ipb)
                ious.append(iou)

    args_desc = np.argsort(ious)[::-1]
    if len(args_desc) == 0:
        # No matches
        tp = 0
        fp = len(pred_boxes)
        fn = len(gt_boxes)

        return [None, None]
    else:
        gt_match_idx = []
        pred_match_idx = []

        for idx in args_desc:
            gt_idx = gt_idx_thr[idx]
            pr_idx = pred_idx_thr[idx]
            # If the boxes are unmatched, add them to matches

            if (gt_idx not in gt_match_idx) and (pr_idx not in pred_match_idx):
                gt_match_idx.append(gt_idx)
                pred_match_idx.append(pr_idx)

        tp = len(gt_match_idx)
        fp = len(pred_boxes) - len(pred_match_idx)
        fn = len(gt_boxes) - len(gt_match_idx)

        fp_list = list(set(range(len(pred_boxes))) - set(pred_match_idx))
        fn_list = list(set(range(len(gt_boxes))) - set(gt_match_idx))

        return [gt_match_idx, pred_match_idx, fp_list, fn_list]


def get_image_Id(img_name):
    img_name = img_name.split(".png")[0]
    sceneList = ["M", "A", "E", "N"]
    cameraIndx = int(img_name.split("_")[0].split("camera")[1])
    sceneIndx = sceneList.index(img_name.split("_")[1])
    frameIndx = int(img_name.split("_")[2])
    imageId = int(str(cameraIndx) + str(sceneIndx) + str(frameIndx))
    return imageId


def load_data(path):
    data = json.load(open(path, "r"))
    res = {}

    for item in data:
        image_id = item["image_id"]

        if image_id not in res:
            res[image_id] = []

        res[image_id].append(item)

    return res


def draw_boxes(draw, boxes, color, label, width=2, score=None):
    for i, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=width)
        draw.text(
            (box[0], max(0, box[1] - 15)), f"{mapping_cls[label[i]]}", fill="cyan"
        )

        if score:
            draw.text(
                (box[0], max(0, box[1] - 30)),
                f"{score[i]:.2f}",
                fill="red",
            )


print("Loading data...")
gt_data = load_data(args.gt)

pred_data = load_data(args.pred)
pred_data = {
    key: [item for item in pred_data[key] if item["score"] > args.conf]
    for key in pred_data
}


print("Comparing data...")
image_path = Path("/mlcv1/Datasets/NvidiaAICityChallenge/NvidiaAIC2024/images")
image_files = list(image_path.glob("*.png"))
output_path = Path("./output_visualize_no_match_thuyentd_inside_outside_095_095_095_075_095")
mapping_cls = {
    0: "bus",
    1: "bike",
    2: "car",
    3: "person",
    4: "truck",
}

if output_path.exists():
    shutil.rmtree(output_path)

os.makedirs(output_path)


SPLIT = 100
distances_total = list()
scores_total = list()

def create_points_begin_from_the_center_2(
    image_width, image_height, x_center, y_center, step
):
    c = int(step / 2)
    for x in range(x_center - c, image_width, step):
        for y in range(y_center - c, image_height, step):
            yield (x, y)
    for x in range(x_center - c, 0, -step):
        for y in range(y_center - c, image_height, step):
            yield (x, y)
    for x in range(x_center - c, image_width, step):
        for y in range(y_center - c, 0, -step):
            yield (x, y)
    for x in range(x_center - c, 0, -step):
        for y in range(y_center - c, 0, -step):
            yield (x, y)


def create_squares_begin_from_the_center_2(
    image_width, image_height, x_center, y_center, step, pad=False
):
    step_ = 0
    if pad:
        step_ = step
    c = 0
    for x in range(x_center - c, 0 - step_, -step):
        for y in range(y_center - c + step, image_height + step_, step):
            yield (x, y, x + step, y, x + step, y + step, x, y + step)

    for x in range(x_center - c + step, image_width + step_, step):
        for y in range(y_center - c + step, image_height + step_, step):
            yield (x, y, x + step, y, x + step, y + step, x, y + step)

    for x in range(x_center - c + step, image_width + step_, step):
        for y in range(y_center - c, 0 - step_, -step):
            yield (x, y, x + step, y, x + step, y + step, x, y + step)

    for x in range(x_center - c, 0 - step_, -step):
        for y in range(y_center - c, 0 - step_, -step):
            yield (x, y, x + step, y, x + step, y + step, x, y + step)


for idx_img,file in tqdm(enumerate(image_files), total = len(image_files)):
    img = Image.open(file)
    draw = ImageDraw.Draw(img)

    img_width, img_height = img.size
    ratio = 0.15
    ellipse = Ellipse(
        (img_width / 2, img_height / 2),
        (1 - 2 * ratio) * img_width,
        (1 - 2 * ratio) * img_height,
    )
    vertices = ellipse.get_verts()
    ellipse = Polygon(vertices)

    x_elip,y_elip = ellipse.exterior.xy
    # PIL draws this ellipse in a weird way, so we need to flip the y-axis

    
    # plt.plot(x,y)

    img_gt = Image.open(root_img_base / f"{file.stem}.png")

    key = get_image_Id(file.stem)
    gt_boxes = [item["bbox"] for item in gt_data[key]]
    pred_boxes = [item["bbox"] for item in pred_data[key]]

    gt_boxes = [
        [item[0], item[1], item[0] + item[2], item[1] + item[3]] for item in gt_boxes
    ]
    pred_boxes = [
        [item[0], item[1], item[0] + item[2], item[1] + item[3]] for item in pred_boxes
    ]

    res = get_single_image_results(gt_boxes, pred_boxes, 0.8)

    tp = res[0]
    fp = res[2]  # baseline la no, nhung predictions la yes
    fn = res[3]  # baseline la yes, nhung predictions la no

    # draw_boxes(draw, gt_boxes, "red")
    # draw_boxes(draw, pred_boxes, "green")

    # #######################

    boxes_fn = [gt_boxes[i] for i in fn]
    cates_fn = [gt_data[key][i]["category_id"] for i in fn]
    scores_fn = [gt_data[key][i]["score"] for i in fn]
    inside = list()
    outside = list()
    for box, cate, score in zip(boxes_fn, cates_fn, scores_fn):
        center_anno_points = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        point = Point(*center_anno_points)
        distance = ellipse.centroid.distance(point)
        d = dict(
            # center=center_anno_points, 
                 what="fn", cate=cate, score=f"{score:.4f}", distance=f"{distance:.0f}")
        if ellipse.contains(point): inside.append(d)
        else:outside.append(d)
    boxes_gt = [gt_boxes[i] for i in tp]
    cates_gt = [gt_data[key][i]["category_id"] for i in tp]
    scores_gt = [gt_data[key][i]["score"] for i in tp]
    # for box, cate, score in zip(boxes_gt, cates_gt, scores_gt):
    #     center_anno_points = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
    #     point = Point(*center_anno_points)
    #     d = dict(center=center_anno_points, what="gt", cate=cate, score=f"{score:.4f}, distance=f"{distance:.4f}")
    #     if ellipse.contains(point): inside.append(d)
    #     else: outside.append(d)
    
    boxes_fp = [pred_boxes[i] for i in fp]
    cates_fp = [pred_data[key][i]["category_id"] for i in fp]
    scores_fp = [pred_data[key][i]["score"] for i in fp]
    for box, cate, score in zip(boxes_fp, cates_fp, scores_fp):
        center_anno_points = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
        point = Point(*center_anno_points)
        distance = ellipse.centroid.distance(point)
        d = dict(
            # center=center_anno_points, 
                 what="fp", cate=cate, score=f"{score:.4f}", distance=f"{distance:.0f}")
        if ellipse.contains(point): inside.append(d)
        else: outside.append(d)
    # print("=====================================")

    # input()
    # #######################

    # draw_boxes(
    #     draw,
    #     [gt_boxes[i] for i in tp],
    #     "blue",
    #     [gt_data[key][i]["category_id"] for i in tp],
    #     3,
    #     [gt_data[key][i]["score"] for i in tp],
    # )

    draw_boxes(
        draw,
        [gt_boxes[i] for i in fn],
        "green",
        [gt_data[key][i]["category_id"] for i in fn],
        3,
        [gt_data[key][i]["score"] for i in fn],
    )

    draw_boxes(
        draw,
        [pred_boxes[i] for i in fp],
        "red",
        [pred_data[key][i]["category_id"] for i in fp],
        3,
        [pred_data[key][i]["score"] for i in fp],
    )
    # distances_image = [float(d["distance"]) for d in inside]
    # scores_image = [float(d["score"]) for d in inside]
    
    # distances_total.extend(distances_image)
    # scores_total.extend(scores_image)
    
    
    # numpy make correlation between distance and score
    
    if len(outside) > 0:
    # if len(inside) > 0:
    # if len(inside) > 0 or len(outside) > 0:
        # pprint(f"{inside = }")
        # pprint(f"{outside = }")
        # pprint(inside)
        pprint(outside)
        concat_img = Image.new("RGB", (img.width + img_gt.width, img.height))
        concat_img.paste(img, (0, 0))
        concat_img.paste(img_gt, (img.width, 0))
        draw_ = ImageDraw.Draw(img)
        x_elip,y_elip = ellipse.exterior.xy
        # print(x_elip,y_elip)
        # x_elip,y_elip = np.array(x_elip), np.array(y_elip)
        # y_elip = [-y_ for y_ in y_elip]
        draw_.polygon(list(zip(x_elip,y_elip)),outline ="red", fill = None, width=2)

        concat_img.save(output_path / f"{file.stem}.png")
        img.save(output_path / f"{file.stem}.png")
        print("Image saved at", output_path / f"{file.stem}.png")
        # input()
#     if idx_img == 1000:
#         break
# distances_total = np.array(distances_total)
# scores_total = np.array(scores_total)
# SPLIT = 250
# # split distances into bins and calculate mean score of each bin
# bins = range(0, 2600, SPLIT)
# distances_total_bin = np.digitize(distances_total, bins)
# print(distances_total_bin)
# print(distances_total)
# print(scores_total)
# # put scores into bins
# # scores_total_bin = np.zeros_like(distances_total_bin, dtype=float)
# scores_total_bin = [0] * len(distances_total_bin)
# count_of_that_bin = [0] * len(distances_total_bin)
# for i in range(len(distances_total_bin)):
#     scores_total_bin[distances_total_bin[i]] += scores_total[i]
#     count_of_that_bin[distances_total_bin[i]] += 1

# count_of_that_bin = np.array([1 if x == 0 else x for x in count_of_that_bin])
# scores_total_bin = np.array(scores_total_bin) / np.array(count_of_that_bin)
# scores_total_bin = scores_total_bin.tolist()
# scores_total_bin = zip(range(len(scores_total_bin)), scores_total_bin, bins)
# scores_total_bin = list(scores_total_bin)

# for i in range(len(scores_total_bin)):
#     print(f"Bin {i}: {scores_total_bin[i]} {bins[i]}")

# print(scores_total_bin)