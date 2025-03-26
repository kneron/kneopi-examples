# ******************************************************************************
#  Copyright (c) 2022. Kneron Inc. All rights reserved.                        *
# ******************************************************************************
from typing import List

from utils.ExampleValue import ExampleBoundingBox, ExampleYoloResult

import os
import sys
import numpy as np

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '../..'))

import kp

YOLO_V3_CELL_BOX_NUM = 3
YOLO_V3_BOX_FIX_CH = 5
NMS_THRESH_YOLOV3 = 0.45
NMS_THRESH_YOLOV5 = 0.5
MAX_POSSIBLE_BOXES = 2000
MODEL_SHIRNK_RATIO_TYV3 = [32, 16]
MODEL_SHIRNK_RATIO_V5 = [8, 16, 32]
YOLO_MAX_DETECTION_PER_CLASS = 100

TINY_YOLO_V3_ANCHERS = np.array([
    [[81, 82], [135, 169], [344, 319]],
    [[23, 27], [37, 58], [81, 82]]
])

YOLO_V5_ANCHERS = np.array([
    [[10, 13], [16, 30], [33, 23]],
    [[30, 61], [62, 45], [59, 119]],
    [[116, 90], [156, 198], [373, 326]]
])


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _iou(box_src, boxes_dst):
    max_x1 = np.maximum(box_src[0], boxes_dst[:, 0])
    max_y1 = np.maximum(box_src[1], boxes_dst[:, 1])
    min_x2 = np.minimum(box_src[2], boxes_dst[:, 2])
    min_y2 = np.minimum(box_src[3], boxes_dst[:, 3])

    area_intersection = np.maximum(0, (min_x2 - max_x1)) * np.maximum(0, (min_y2 - max_y1))
    area_src = (box_src[2] - box_src[0]) * (box_src[3] - box_src[1])
    area_dst = (boxes_dst[:, 2] - boxes_dst[:, 0]) * (boxes_dst[:, 3] - boxes_dst[:, 1])
    area_union = area_src + area_dst - area_intersection

    iou = area_intersection / area_union

    return iou


def _boxes_scale(boxes, hardware_preproc_info: kp.HwPreProcInfo):
    """
    Kneron hardware image pre-processing will do cropping, resize, padding by following ordering:
    1. cropping
    2. resize
    3. padding
    """
    ratio_w = hardware_preproc_info.img_width / hardware_preproc_info.resized_img_width
    ratio_h = hardware_preproc_info.img_height / hardware_preproc_info.resized_img_height

    # rollback padding
    boxes[..., :4] = boxes[..., :4] - np.array([hardware_preproc_info.pad_left, hardware_preproc_info.pad_top,
                                                hardware_preproc_info.pad_left, hardware_preproc_info.pad_top])

    # scale coordinate
    boxes[..., :4] = boxes[..., :4] * np.array([ratio_w, ratio_h, ratio_w, ratio_h])

    return boxes


def post_process_tiny_yolo_v3(inference_float_node_output_list: List[kp.InferenceFloatNodeOutput],
                              hardware_preproc_info: kp.HwPreProcInfo,
                              thresh_value: float,
                              with_sigmoid: bool = True) -> ExampleYoloResult:
    """
    Tiny YOLO V3 post-processing function.

    Parameters
    ----------
    inference_float_node_output_list : List[kp.InferenceFloatNodeOutput]
        A floating-point output node list, it should come from
        'kp.inference.generic_inference_retrieve_float_node()'.
    hardware_preproc_info : kp.HwPreProcInfo
        Information of Hardware Pre Process.
    thresh_value : float
        The threshold of YOLO postprocessing, range from 0.0 ~ 1.0
    with_sigmoid: bool, default=True
        Do sigmoid operation before postprocessing.

    Returns
    -------
    yolo_result : utils.ExampleValue.ExampleYoloResult
        YoloResult object contained the post-processed result.

    See Also
    --------
    kp.core.connect_devices : To connect multiple (including one) Kneron devices.
    kp.inference.generic_inference_retrieve_float_node : Retrieve single node output data from raw output buffer.
    kp.InferenceFloatNodeOutput
    kp.HwPreProcInfo
    utils.ExampleValue.ExampleYoloResult
    """
    feature_map_list = []
    candidate_boxes_list = []

    for i in range(len(inference_float_node_output_list)):
        anchor_offset = int(inference_float_node_output_list[i].shape[1] / YOLO_V3_CELL_BOX_NUM)
        feature_map = inference_float_node_output_list[i].ndarray.transpose((0, 2, 3, 1))
        feature_map = _sigmoid(feature_map) if with_sigmoid else feature_map
        feature_map = feature_map.reshape((feature_map.shape[0],
                                           feature_map.shape[1],
                                           feature_map.shape[2],
                                           YOLO_V3_CELL_BOX_NUM,
                                           anchor_offset))

        ratio_w = hardware_preproc_info.model_input_width / inference_float_node_output_list[i].shape[3]
        ratio_h = hardware_preproc_info.model_input_height / inference_float_node_output_list[i].shape[2]
        nrows = inference_float_node_output_list[i].shape[2]
        ncols = inference_float_node_output_list[i].shape[3]
        grids = np.expand_dims(np.stack(np.meshgrid(np.arange(ncols), np.arange(nrows)), 2), axis=0)

        for anchor_idx in range(YOLO_V3_CELL_BOX_NUM):
            feature_map[..., anchor_idx, 0:2] = (feature_map[..., anchor_idx, 0:2] + grids) * np.array(
                [ratio_h, ratio_w])
            feature_map[..., anchor_idx, 2:4] = (feature_map[..., anchor_idx, 2:4] * 2) ** 2 * TINY_YOLO_V3_ANCHERS[i][
                anchor_idx]

            feature_map[..., anchor_idx, 0:2] = feature_map[..., anchor_idx, 0:2] - (
                        feature_map[..., anchor_idx, 2:4] / 2.)
            feature_map[..., anchor_idx, 2:4] = feature_map[..., anchor_idx, 0:2] + feature_map[..., anchor_idx, 2:4]

        feature_map = _boxes_scale(boxes=feature_map,
                                   hardware_preproc_info=hardware_preproc_info)

        feature_map_list.append(feature_map)

    predict_bboxes = np.concatenate(
        [np.reshape(feature_map, (-1, feature_map.shape[-1])) for feature_map in feature_map_list], axis=0)
    predict_bboxes[..., 5:] = np.repeat(predict_bboxes[..., 4][..., np.newaxis],
                                        predict_bboxes[..., 5:].shape[1],
                                        axis=1) * predict_bboxes[..., 5:]
    predict_bboxes_mask = (predict_bboxes[..., 5:] > thresh_value).sum(axis=1)
    predict_bboxes = predict_bboxes[predict_bboxes_mask >= 1]

    # nms
    for class_idx in range(5, predict_bboxes.shape[1]):
        candidate_boxes_mask = predict_bboxes[..., class_idx] > thresh_value
        class_good_box_count = candidate_boxes_mask.sum()
        if class_good_box_count == 1:
            candidate_boxes_list.append(
                ExampleBoundingBox(
                    x1=round(float(predict_bboxes[candidate_boxes_mask, 0][0]), 4),
                    y1=round(float(predict_bboxes[candidate_boxes_mask, 1][0]), 4),
                    x2=round(float(predict_bboxes[candidate_boxes_mask, 2][0]), 4),
                    y2=round(float(predict_bboxes[candidate_boxes_mask, 3][0]), 4),
                    score=round(float(predict_bboxes[candidate_boxes_mask, class_idx][0]), 4),
                    class_num=class_idx - 5
                )
            )
        elif class_good_box_count > 1:
            candidate_boxes = predict_bboxes[candidate_boxes_mask].copy()
            candidate_boxes = candidate_boxes[candidate_boxes[:, class_idx].argsort()][::-1]

            for candidate_box_idx in range(candidate_boxes.shape[0] - 1):
                # origin python version post-processing
                if 0 != candidate_boxes[candidate_box_idx][class_idx]:
                    remove_mask = _iou(box_src=candidate_boxes[candidate_box_idx],
                                        boxes_dst=candidate_boxes[candidate_box_idx + 1:]) > NMS_THRESH_YOLOV3
                    candidate_boxes[candidate_box_idx + 1:][remove_mask, class_idx] = 0

            good_count = 0
            for candidate_box_idx in range(candidate_boxes.shape[0]):
                if candidate_boxes[candidate_box_idx, class_idx] > 0:
                    candidate_boxes_list.append(
                        ExampleBoundingBox(
                            x1=round(float(candidate_boxes[candidate_box_idx, 0]), 4),
                            y1=round(float(candidate_boxes[candidate_box_idx, 1]), 4),
                            x2=round(float(candidate_boxes[candidate_box_idx, 2]), 4),
                            y2=round(float(candidate_boxes[candidate_box_idx, 3]), 4),
                            score=round(float(candidate_boxes[candidate_box_idx, class_idx]), 4),
                            class_num=class_idx - 5
                        )
                    )
                    good_count += 1

                    if YOLO_MAX_DETECTION_PER_CLASS == good_count:
                        break

    for idx, candidate_boxes in enumerate(candidate_boxes_list):
        candidate_boxes_list[idx].x1 = 0 if (candidate_boxes_list[idx].x1 + 0.5 < 0) else int(
                candidate_boxes_list[idx].x1 + 0.5)
        candidate_boxes_list[idx].y1 = 0 if (candidate_boxes_list[idx].y1 + 0.5 < 0) else int(
                candidate_boxes_list[idx].y1 + 0.5)
        candidate_boxes_list[idx].x2 = int(hardware_preproc_info.img_width - 1) if (
                candidate_boxes_list[idx].x2 + 0.5 > hardware_preproc_info.img_width - 1) else int(candidate_boxes_list[idx].x2 + 0.5)
        candidate_boxes_list[idx].y2 = int(hardware_preproc_info.img_height - 1) if (
                candidate_boxes_list[idx].y2 + 0.5 > hardware_preproc_info.img_height - 1) else int(candidate_boxes_list[idx].y2 + 0.5)

    return ExampleYoloResult(
        class_count=predict_bboxes.shape[1] - 5,
        box_count=len(candidate_boxes_list),
        box_list=candidate_boxes_list
    )


def post_process_yolo_v5(inference_float_node_output_list: List[kp.InferenceFloatNodeOutput],
                         hardware_preproc_info: kp.HwPreProcInfo,
                         thresh_value: float,
                         with_sigmoid: bool = True) -> ExampleYoloResult:
    """
    YOLO V5 post-processing function.

    Parameters
    ----------
    inference_float_node_output_list : List[kp.InferenceFloatNodeOutput]
        A floating-point output node list, it should come from
        'kp.inference.generic_inference_retrieve_float_node()'.
    hardware_preproc_info : kp.HwPreProcInfo
        Information of Hardware Pre Process.
    thresh_value : float
        The threshold of YOLO postprocessing, range from 0.0 ~ 1.0
    with_sigmoid: bool, default=True
        Do sigmoid operation before postprocessing.

    Returns
    -------
    yolo_result : utils.ExampleValue.ExampleYoloResult
        YoloResult object contained the post-processed result.

    See Also
    --------
    kp.core.connect_devices : To connect multiple (including one) Kneron devices.
    kp.inference.generic_inference_retrieve_float_node : Retrieve single node output data from raw output buffer.
    kp.InferenceFloatNodeOutput
    kp.HwPreProcInfo
    utils.ExampleValue.ExampleYoloResult
    """
    feature_map_list = []
    candidate_boxes_list = []

    for i in range(len(inference_float_node_output_list)):
        anchor_offset = int(inference_float_node_output_list[i].shape[1] / YOLO_V3_CELL_BOX_NUM)
        feature_map = inference_float_node_output_list[i].ndarray.transpose((0, 2, 3, 1))
        feature_map = _sigmoid(feature_map) if with_sigmoid else feature_map
        feature_map = feature_map.reshape((feature_map.shape[0],
                                           feature_map.shape[1],
                                           feature_map.shape[2],
                                           YOLO_V3_CELL_BOX_NUM,
                                           anchor_offset))

        ratio_w = hardware_preproc_info.model_input_width / inference_float_node_output_list[i].shape[3]
        ratio_h = hardware_preproc_info.model_input_height / inference_float_node_output_list[i].shape[2]
        nrows = inference_float_node_output_list[i].shape[2]
        ncols = inference_float_node_output_list[i].shape[3]
        grids = np.expand_dims(np.stack(np.meshgrid(np.arange(ncols), np.arange(nrows)), 2), axis=0)

        for anchor_idx in range(YOLO_V3_CELL_BOX_NUM):
            feature_map[..., anchor_idx, 0:2] = (feature_map[..., anchor_idx, 0:2] * 2. - 0.5 + grids) * np.array(
                [ratio_h, ratio_w])
            feature_map[..., anchor_idx, 2:4] = (feature_map[..., anchor_idx, 2:4] * 2) ** 2 * YOLO_V5_ANCHERS[i][
                anchor_idx]

            feature_map[..., anchor_idx, 0:2] = feature_map[..., anchor_idx, 0:2] - (
                    feature_map[..., anchor_idx, 2:4] / 2.)
            feature_map[..., anchor_idx, 2:4] = feature_map[..., anchor_idx, 0:2] + feature_map[..., anchor_idx, 2:4]

        feature_map = _boxes_scale(boxes=feature_map,
                                   hardware_preproc_info=hardware_preproc_info)

        feature_map_list.append(feature_map)

    predict_bboxes = np.concatenate(
        [np.reshape(feature_map, (-1, feature_map.shape[-1])) for feature_map in feature_map_list], axis=0)
    predict_bboxes[..., 5:] = np.repeat(predict_bboxes[..., 4][..., np.newaxis],
                                        predict_bboxes[..., 5:].shape[1],
                                        axis=1) * predict_bboxes[..., 5:]
    predict_bboxes_mask = (predict_bboxes[..., 5:] > thresh_value).sum(axis=1)
    predict_bboxes = predict_bboxes[predict_bboxes_mask >= 1]

    # nms
    for class_idx in range(5, predict_bboxes.shape[1]):
        candidate_boxes_mask = predict_bboxes[..., class_idx] > thresh_value
        class_good_box_count = candidate_boxes_mask.sum()
        if class_good_box_count == 1:
            candidate_boxes_list.append(
                ExampleBoundingBox(
                    x1=round(float(predict_bboxes[candidate_boxes_mask, 0][0]), 4),
                    y1=round(float(predict_bboxes[candidate_boxes_mask, 1][0]), 4),
                    x2=round(float(predict_bboxes[candidate_boxes_mask, 2][0]), 4),
                    y2=round(float(predict_bboxes[candidate_boxes_mask, 3][0]), 4),
                    score=round(float(predict_bboxes[candidate_boxes_mask, class_idx][0]), 4),
                    class_num=class_idx - 5
                )
            )
        elif class_good_box_count > 1:
            candidate_boxes = predict_bboxes[candidate_boxes_mask].copy()
            candidate_boxes = candidate_boxes[candidate_boxes[:, class_idx].argsort()][::-1]

            for candidate_box_idx in range(candidate_boxes.shape[0] - 1):
                if 0 != candidate_boxes[candidate_box_idx][class_idx]:
                    remove_mask = _iou(box_src=candidate_boxes[candidate_box_idx],
                                       boxes_dst=candidate_boxes[candidate_box_idx + 1:]) > NMS_THRESH_YOLOV5
                    candidate_boxes[candidate_box_idx + 1:][remove_mask, class_idx] = 0

            good_count = 0
            for candidate_box_idx in range(candidate_boxes.shape[0]):
                if candidate_boxes[candidate_box_idx, class_idx] > 0:
                    candidate_boxes_list.append(
                        ExampleBoundingBox(
                            x1=round(float(candidate_boxes[candidate_box_idx, 0]), 4),
                            y1=round(float(candidate_boxes[candidate_box_idx, 1]), 4),
                            x2=round(float(candidate_boxes[candidate_box_idx, 2]), 4),
                            y2=round(float(candidate_boxes[candidate_box_idx, 3]), 4),
                            score=round(float(candidate_boxes[candidate_box_idx, class_idx]), 4),
                            class_num=class_idx - 5
                        )
                    )
                    good_count += 1

                    if YOLO_MAX_DETECTION_PER_CLASS == good_count:
                        break

    for idx, candidate_boxes in enumerate(candidate_boxes_list):
        candidate_boxes_list[idx].x1 = 0 if (candidate_boxes_list[idx].x1 + 0.5 < 0) else int(
                candidate_boxes_list[idx].x1 + 0.5)
        candidate_boxes_list[idx].y1 = 0 if (candidate_boxes_list[idx].y1 + 0.5 < 0) else int(
                candidate_boxes_list[idx].y1 + 0.5)
        candidate_boxes_list[idx].x2 = int(hardware_preproc_info.img_width - 1) if (
                candidate_boxes_list[idx].x2 + 0.5 > hardware_preproc_info.img_width - 1) else int(candidate_boxes_list[idx].x2 + 0.5)
        candidate_boxes_list[idx].y2 = int(hardware_preproc_info.img_height - 1) if (
                candidate_boxes_list[idx].y2 + 0.5 > hardware_preproc_info.img_height - 1) else int(candidate_boxes_list[idx].y2 + 0.5)

    return ExampleYoloResult(
        class_count=predict_bboxes.shape[1] - 5,
        box_count=len(candidate_boxes_list),
        box_list=candidate_boxes_list
    )
