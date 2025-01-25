import numpy as np

def nms(bounding_boxes, confidence_score, threshold):
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return [], []

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)
    indices = []

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        indices.append(index)

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    return picked_boxes, picked_score, indices

def greedy_nms(dets, iou_thr):
    """Pure python implementation of traditional greedy NMS.

    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        iou_thr (float): Drop the boxes that overlap with current
            maximum > thresh.

    Returns:
        numpy.array: Retained boxes.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_idx = scores.argsort()[::-1]

    keep = []
    while sorted_idx.size > 0:
        i = sorted_idx[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[sorted_idx[1:]])
        yy1 = np.maximum(y1[i], y1[sorted_idx[1:]])
        xx2 = np.minimum(x2[i], x2[sorted_idx[1:]])
        yy2 = np.minimum(y2[i], y2[sorted_idx[1:]])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (areas[i] + areas[sorted_idx[1:]] - inter)

        retained_idx = np.where(iou <= iou_thr)[0]
        sorted_idx = sorted_idx[retained_idx + 1]

    return keep


def soft_nms(dets, method='linear', iou_thr=0.3, sigma=0.5, score_thr=0.001):
    """Pure python implementation of soft NMS as described in the paper
    `Improving Object Detection With One Line of Code`_.

    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        method (str): Rescore method. Only can be `linear`, `gaussian`
            or 'greedy'.
        iou_thr (float): IOU threshold. Only work when method is `linear`
            or 'greedy'.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_thr (float): Boxes that score less than the.

    Returns:
        numpy.array: Retained boxes.

    .. _`Improving Object Detection With One Line of Code`:
        https://arxiv.org/abs/1704.04503
    """
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    retained_box = []
    while dets.size > 0:
        max_idx = np.argmax(dets[:, 4], axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        retained_box.append(dets[0, :-1])

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        yy1 = np.maximum(dets[0, 1], dets[1:, 1])
        xx2 = np.minimum(dets[0, 2], dets[1:, 2])
        yy2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_thr] -= iou[iou > iou_thr]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_thr] = 0

        dets[1:, 4] *= weight
        retained_idx = np.where(dets[1:, 4] >= score_thr)[0]
        dets = dets[retained_idx + 1, :]

    return np.vstack(retained_box)