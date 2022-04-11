#!/usr/bin/env python3
# 53907afe-531b-11ea-a595-00505601122b
# b7ea974c-d389-11e8-a4be-00505601122b
import argparse
from typing import Callable, Tuple
import unittest

import numpy as np

BACKEND = np  # or you can use `tf` for TensorFlow implementation

# Bounding boxes and anchors are expected to be Numpy/TensorFlow tensors,
# where the last dimension has size 4.
Tensor = np.ndarray  # or use `tf.Tensor` if you use TensorFlow backend

# For bounding boxes in pixel coordinates, the 4 values correspond to:
TOP: int = 0
LEFT: int = 1
BOTTOM: int = 2
RIGHT: int = 3


def _ob_height(x: Tensor) -> Tensor:
    return BACKEND.maximum(x[..., BOTTOM] - x[..., TOP], 0)


def _ob_width(x: Tensor) -> Tensor:
    return BACKEND.maximum(x[..., RIGHT] - x[..., LEFT], 0)


def _ob_x_center(x: Tensor) -> Tensor:
    return _ob_width(x) / 2 + x[..., LEFT]


def _ob_y_center(x: Tensor) -> Tensor:
    return _ob_height(x) / 2 + x[..., TOP]


def bboxes_area(bboxes: Tensor) -> Tensor:
    """Compute area of given set of bboxes.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    If the bboxes.shape is [..., 4], the output shape is bboxes.shape[:-1].
    """
    return BACKEND.maximum(bboxes[..., BOTTOM] - bboxes[..., TOP], 0) * BACKEND.maximum(
        bboxes[..., RIGHT] - bboxes[..., LEFT], 0
    )


def bboxes_iou(xs: Tensor, ys: Tensor) -> Tensor:
    """Compute IoU of corresponding pairs from two sets of bboxes xs and ys.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    Note that broadcasting is supported, so passing inputs with
    xs.shape=[num_xs, 1, 4] and ys.shape=[1, num_ys, 4] will produce output
    with shape [num_xs, num_ys], computing IoU for all pairs of bboxes from
    xs and ys. Formally, the output shape is np.broadcast(xs, ys).shape[:-1].
    """
    intersections = BACKEND.stack(
        [
            BACKEND.maximum(xs[..., TOP], ys[..., TOP]),
            BACKEND.maximum(xs[..., LEFT], ys[..., LEFT]),
            BACKEND.minimum(xs[..., BOTTOM], ys[..., BOTTOM]),
            BACKEND.minimum(xs[..., RIGHT], ys[..., RIGHT]),
        ],
        axis=-1,
    )

    xs_area, ys_area, intersections_area = (
        bboxes_area(xs),
        bboxes_area(ys),
        bboxes_area(intersections),
    )

    return intersections_area / (xs_area + ys_area - intersections_area)


def bboxes_to_fast_rcnn(anchors: Tensor, bboxes: Tensor) -> Tensor:
    """Convert `bboxes` to a Fast-R-CNN-like representation relative to `anchors`.

    The `anchors` and `bboxes` are arrays of four-tuples (top, left, bottom, right);
    you can use the TOP, LEFT, BOTTOM, RIGHT constants as indices of the
    respective coordinates.

    The resulting representation of a single bbox is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - log(bbox_height / anchor_height)
    - log(bbox_width / anchor_width)

    If the anchors.shape is [anchors_len, 4], bboxes.shape is [anchors_len, 4],
    the output shape is [anchors_len, 4].
    """
    out = BACKEND.zeros(anchors.shape, dtype=anchors.dtype)

    out[..., 0] = (_ob_y_center(bboxes) - _ob_y_center(anchors)) / _ob_height(anchors)
    out[..., 1] = (_ob_x_center(bboxes) - _ob_x_center(anchors)) / _ob_width(anchors)
    out[..., 2] = BACKEND.log(_ob_height(bboxes) / _ob_height(anchors))
    out[..., 3] = BACKEND.log(_ob_width(bboxes) / _ob_width(anchors))

    return out


def bboxes_from_fast_rcnn(anchors: Tensor, fast_rcnns: Tensor) -> Tensor:
    """Convert Fast-R-CNN-like representation relative to `anchor` to a `bbox`.

    The anchors.shape is [anchors_len, 4], fast_rcnns.shape is [anchors_len, 4],
    the output shape is [anchors_len, 4].
    """
    out = BACKEND.zeros(anchors.shape, dtype=anchors.dtype)
    y = fast_rcnns[..., 0] * _ob_height(anchors) + _ob_y_center(anchors)
    x = fast_rcnns[..., 1] * _ob_width(anchors) + _ob_x_center(anchors)
    h = _ob_height(anchors) * BACKEND.exp(fast_rcnns[..., 2])
    w = _ob_width(anchors) * BACKEND.exp(fast_rcnns[..., 3])

    out[..., TOP] = y - h / 2
    out[..., LEFT] = x - w / 2
    out[..., BOTTOM] = out[..., TOP] + h
    out[..., RIGHT] = out[..., LEFT] + w

    return out


def bboxes_training(
    anchors: Tensor, gold_classes: Tensor, gold_bboxes: Tensor, iou_threshold: float
) -> Tuple[Tensor, Tensor]:
    """Compute training data for object detection.

    Arguments:
    - `anchors` is an array of four-tuples (top, left, bottom, right)
    - `gold_classes` is an array of zero-based classes of the gold objects
    - `gold_bboxes` is an array of four-tuples (top, left, bottom, right)
      of the gold objects
    - `iou_threshold` is a given threshold

    Returns:
    - `anchor_classes` contains for every anchor either 0 for background
      (if no gold object is assigned) or `1 + gold_class` if a gold object
      with `gold_class` is assigned to it
    - `anchor_bboxes` contains for every anchor a four-tuple
      `(center_y, center_x, height, width)` representing the gold bbox of
      a chosen object using parametrization of Fast R-CNN; zeros if no
      gold object was assigned to the anchor

    Algorithm:
    - First, for each gold object, assign it to an anchor with the largest IoU
      (the one with smaller index if there are several). In case several gold
      objects are assigned to a single anchor, use the gold object with smaller
      index.
    - For each unused anchors, find the gold object with the largest IoU
      (again the one with smaller index if there are several), and if the IoU
      is >= iou_threshold, assign the object to the anchor.
    """
    anchor_classes = BACKEND.zeros([anchors.shape[0]], dtype=BACKEND.int32)
    anchor_bboxes = BACKEND.zeros([anchors.shape[0], 4], dtype=BACKEND.float32)

    # TODO: First, for each gold object, assign it to an anchor with the
    # largest IoU (the one with smaller index if there are several). In case
    # several gold objects are assigned to a single anchor, use the gold object
    # with smaller index.
    for i, bbox in zip(range(gold_bboxes.shape[0]), gold_bboxes):
        idx = BACKEND.argmax(bboxes_iou(bbox, anchors))
        if anchor_classes[idx] == 0 or (1 + gold_classes[i]) < anchor_classes[idx]:
            anchor_classes[idx] = 1 + gold_classes[i]
            anchor_bboxes[idx] = bboxes_to_fast_rcnn(anchors[idx], bbox)

    # TODO: For each unused anchor, find the gold object with the largest IoU
    # (again the one with smaller index if there are several), and if the IoU
    # is >= threshold, assign the object to the anchor.
    for i, anchor in zip(range(anchors.shape[0]), anchors):
        if anchor_classes[i] > 0:
            continue

        iou = bboxes_iou(anchor, gold_bboxes)
        idx = BACKEND.argmax(iou)
        if iou[idx] >= iou_threshold:
            anchor_classes[i] = 1 + gold_classes[idx]
            anchor_bboxes[i] = bboxes_to_fast_rcnn(anchors[i], gold_bboxes[idx])

    return anchor_classes, anchor_bboxes


def main(args: argparse.Namespace) -> Tuple[Callable, Callable, Callable]:
    return bboxes_to_fast_rcnn, bboxes_from_fast_rcnn, bboxes_training


class Tests(unittest.TestCase):
    def test_bboxes_to_from_fast_rcnn(self):
        data = [
            [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
            [[0, 0, 10, 10], [5, 0, 15, 10], [0.5, 0, 0, 0]],
            [[0, 0, 10, 10], [0, 5, 10, 15], [0, 0.5, 0, 0]],
            [[0, 0, 10, 10], [0, 0, 20, 30], [0.5, 1, np.log(2), np.log(3)]],
            [[0, 9, 10, 19], [2, 10, 5, 16], [-0.15, -0.1, -1.20397, -0.51083]],
            [[5, 3, 15, 13], [7, 7, 10, 9], [-0.15, 0, -1.20397, -1.60944]],
            [[7, 6, 17, 16], [9, 10, 12, 13], [-0.15, 0.05, -1.20397, -1.20397]],
            [[5, 6, 15, 16], [7, 7, 10, 10], [-0.15, -0.25, -1.20397, -1.20397]],
            [[6, 3, 16, 13], [8, 5, 12, 8], [-0.1, -0.15, -0.91629, -1.20397]],
            [[5, 2, 15, 12], [9, 6, 12, 8], [0.05, 0, -1.20397, -1.60944]],
            [[2, 10, 12, 20], [6, 11, 8, 17], [0, -0.1, -1.60944, -0.51083]],
            [[10, 9, 20, 19], [12, 13, 17, 16], [-0.05, 0.05, -0.69315, -1.20397]],
            [[6, 7, 16, 17], [10, 11, 12, 14], [0, 0.05, -1.60944, -1.20397]],
            [[2, 2, 12, 12], [3, 5, 8, 8], [-0.15, -0.05, -0.69315, -1.20397]],
        ]
        # First run on individual anchors, and then on all together
        for anchors, bboxes, fast_rcnns in [map(lambda x: [x], row) for row in data] + [
            zip(*data)
        ]:
            anchors, bboxes, fast_rcnns = [
                np.array(data, np.float32) for data in [anchors, bboxes, fast_rcnns]
            ]
            np.testing.assert_almost_equal(
                bboxes_to_fast_rcnn(anchors, bboxes), fast_rcnns, decimal=3
            )
            np.testing.assert_almost_equal(
                bboxes_from_fast_rcnn(anchors, fast_rcnns), bboxes, decimal=3
            )

    def test_bboxes_training(self):
        anchors = np.array(
            [[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]],
            np.float32,
        )
        for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
            [
                [1],
                [[14.0, 14, 16, 16]],
                [0, 0, 0, 2],
                [[0, 0, 0, 0]] * 3 + [[0, 0, np.log(0.2), np.log(0.2)]],
                0.5,
            ],
            [
                [2],
                [[0.0, 0, 20, 20]],
                [3, 0, 0, 0],
                [[0.5, 0.5, np.log(2), np.log(2)]] + [[0, 0, 0, 0]] * 3,
                0.26,
            ],
            [
                [2],
                [[0.0, 0, 20, 20]],
                [3, 3, 3, 3],
                [
                    [y, x, np.log(2), np.log(2)]
                    for y in [0.5, -0.5]
                    for x in [0.5, -0.5]
                ],
                0.24,
            ],
            [
                [0, 1],
                [[3, 3, 20, 18], [10, 1, 18, 21]],
                [0, 0, 0, 1],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [-0.35, -0.45, 0.53062, 0.40546],
                ],
                0.5,
            ],
            [
                [0, 1],
                [[3, 3, 20, 18], [10, 1, 18, 21]],
                [0, 0, 2, 1],
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [-0.1, 0.6, -0.22314, 0.69314],
                    [-0.35, -0.45, 0.53062, 0.40546],
                ],
                0.3,
            ],
            [
                [0, 1],
                [[3, 3, 20, 18], [10, 1, 18, 21]],
                [0, 1, 2, 1],
                [
                    [0, 0, 0, 0],
                    [0.65, -0.45, 0.53062, 0.40546],
                    [-0.1, 0.6, -0.22314, 0.69314],
                    [-0.35, -0.45, 0.53062, 0.40546],
                ],
                0.17,
            ],
        ]:
            gold_classes, anchor_classes = np.array(gold_classes, np.int32), np.array(
                anchor_classes, np.int32
            )
            gold_bboxes, anchor_bboxes = np.array(gold_bboxes, np.float32), np.array(
                anchor_bboxes, np.float32
            )
            computed_classes, computed_bboxes = bboxes_training(
                anchors, gold_classes, gold_bboxes, iou
            )
            np.testing.assert_almost_equal(computed_classes, anchor_classes, decimal=3)
            np.testing.assert_almost_equal(computed_bboxes, anchor_bboxes, decimal=3)


if __name__ == "__main__":
    unittest.main()
