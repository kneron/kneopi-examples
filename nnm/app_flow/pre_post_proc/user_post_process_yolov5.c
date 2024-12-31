/*
 * Kneron Example YOLOv5 Post-Processing driver
 *
 * Copyright (C) 2023 Kneron, Inc. All rights reserved.
 *
 */
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "base.h"
#include "ipc.h"
#include "kdpio.h"
#include "user_post_mem_manager.h"
#include "user_post_process_yolov5.h"

/**
 * For the YOLO V5 output feature map channel definition:
 *
 * feature map 0 channel definition:
 *      anchor cell:    |layer_0_anchor_0                                                                                                                       |layer_0_anchor_1                                                                                                       |layer_0_anchor_2
 *      channel:        |box_x_0|box_y_0|box_w_0|box_h_0|box_confidence_0|box_class_0_confidence_0|box_class_0_confidence_1|...|box_class_0_confidence_N|box_x_1|box_y_1|box_w_1|box_h_1|box_confidence_1|box_class_1_confidence_0|box_class_1_confidence_1|...|box_class_1_confidence_N|box_y_2|box_w_2|box_h_2|box_confidence_2|box_class_2_confidence_0|box_class_2_confidence_1|...|box_class_2_confidence_N|
 *
 * feature map 1 channel definition:
 *      anchor cell:    |layer_1_anchor_0                                                                                                                       |layer_1_anchor_1                                                                                                       |layer_1_anchor_2
 *      channel:        |box_x_0|box_y_0|box_w_0|box_h_0|box_confidence_0|box_class_0_confidence_0|box_class_0_confidence_1|...|box_class_0_confidence_N|box_x_1|box_y_1|box_w_1|box_h_1|box_confidence_1|box_class_1_confidence_0|box_class_1_confidence_1|...|box_class_1_confidence_N|box_y_2|box_w_2|box_h_2|box_confidence_2|box_class_2_confidence_0|box_class_2_confidence_1|...|box_class_2_confidence_N|
 *
 * feature map 2 channel definition:
 *      anchor cell:    |layer_2_anchor_0                                                                                                                       |layer_2_anchor_1                                                                                                       |layer_2_anchor_2
 *      channel:        |box_x_0|box_y_0|box_w_0|box_h_0|box_confidence_0|box_class_0_confidence_0|box_class_0_confidence_1|...|box_class_0_confidence_N|box_x_1|box_y_1|box_w_1|box_h_1|box_confidence_1|box_class_1_confidence_0|box_class_1_confidence_1|...|box_class_1_confidence_N|box_y_2|box_w_2|box_h_2|box_confidence_2|box_class_2_confidence_0|box_class_2_confidence_1|...|box_class_2_confidence_N|
 *
 */

/******************************************************************
 * local define values
*******************************************************************/
#define DEFAULT_PROBABILITY_THRESHOLD               (0.6)                       /**< default probability threshold */
#define DEFAULT_IOU_THRESHOLD                       (0.45)                      /**< default IoU threshold */
#define DEFAULT_MAX_YOLO_V5_DETECT_BOX              (500)                       /**< default max detected box */
#define DEFAULT_MAX_YOLO_V5_DETECT_BOX_PER_CLASS    (500)                       /**< default max detected box per class */
#define DEFAULT_NMS_MODE                            (EX_NMS_MODE_SINGLE_CLASS)  /**< default nms mode */
#define MAX_YOLO_V5_DETECT_CANDIDATE_BOX            (500)                       /**< max detected candidate box */
#define EX_MAX_YOLO_ANCHOR_LAYER_NUM                   (5)                         /**< max number of layer for anchor */
#define EX_MAX_YOLO_ANCHOR_CELL_NUM_PER_LAYER          (3)                         /**< max number of anchor per gride cell */
#define YOLO_V5_ANCHOR_NUM_PER_LAYER                (3)                         /**< number of anchor per layer */
#define YOLO_V5_BOX_FIX_CH                          (5)                         /**< first 5 channel offset for each anchor cell; x, y, w, h, confidence score */
#define KDP_COL_MIN                                 (16)                        /**< bytes, i.e. 128 bits */
#define UNPASS_SCORE                                (-999.0f)                   /**< used as box filter */

/******************************************************************
 * local struct defined
*******************************************************************/
/**
 * @brief enum for NMS mode
 */
struct ex_yolo_v5_post_globals_s
{
    struct ex_bounding_box_s boxes[MAX_YOLO_V5_DETECT_CANDIDATE_BOX];           /**< candidate boxes */
    struct ex_bounding_box_s result_tmp_s[MAX_YOLO_V5_DETECT_CANDIDATE_BOX];    /**< candidate boxes temp buffer */
};

/******************************************************************
 * local variable initialization
*******************************************************************/
struct ex_yolo_v5_post_globals_s *yolo_v5_gp    = NULL;
static float prob_threshold                     = DEFAULT_PROBABILITY_THRESHOLD;
static float iou_threshold                      = DEFAULT_IOU_THRESHOLD;
static uint32_t max_detection_box               = DEFAULT_MAX_YOLO_V5_DETECT_BOX;
static uint32_t max_detection_box_per_class     = DEFAULT_MAX_YOLO_V5_DETECT_BOX_PER_CLASS;
static ex_nms_mode_t nms_mode                   = DEFAULT_NMS_MODE;
static float default_anchors[EX_MAX_YOLO_ANCHOR_LAYER_NUM][EX_MAX_YOLO_ANCHOR_CELL_NUM_PER_LAYER][2] \
                                                = {{{10, 13}, {16, 30}, {33, 23}},
                                                   {{30, 61}, {62, 45}, {59, 119}},
                                                   {{116 ,90}, {156, 198}, {373, 326}},
                                                   {{0 ,0}, {0, 0}, {0, 0}},
                                                   {{0 ,0}, {0, 0}, {0, 0}}};
static float anchors[EX_MAX_YOLO_ANCHOR_LAYER_NUM][EX_MAX_YOLO_ANCHOR_CELL_NUM_PER_LAYER][2] \
                                                = {{{0}}};

/******************************************************************
 * local util function
*******************************************************************/
// N/A

/******************************************************************
 * main function
*******************************************************************/
/**
 * @brief yolo v5 post-process reference code
 */
int user_post_yolov5_no_sigmoid(int model_id, struct kdp_image_s *image_p)
{
    (void)model_id;

    /* get pre-post process configuration */
    ex_yolo_post_proc_config_t *yolo_pre_post_proc_config = (ex_yolo_post_proc_config_t *)POSTPROC_PARAMS_P(image_p);

    /* setting pre-post process params */
    if (yolo_pre_post_proc_config->prob_thresh > 0)
        prob_threshold = yolo_pre_post_proc_config->prob_thresh;
    else
        prob_threshold = DEFAULT_PROBABILITY_THRESHOLD;

    if (yolo_pre_post_proc_config->nms_thresh > 0)
        iou_threshold = yolo_pre_post_proc_config->nms_thresh;
    else
        iou_threshold = DEFAULT_IOU_THRESHOLD;

    if (yolo_pre_post_proc_config->max_detection > 0)
    {
        max_detection_box = yolo_pre_post_proc_config->max_detection;
        if (max_detection_box > MAX_YOLO_V5_DETECT_CANDIDATE_BOX)
            max_detection_box = MAX_YOLO_V5_DETECT_CANDIDATE_BOX;
    }
    else
        max_detection_box = DEFAULT_MAX_YOLO_V5_DETECT_BOX;

    if (yolo_pre_post_proc_config->max_detection_per_class > 0)
    {
        max_detection_box_per_class = yolo_pre_post_proc_config->max_detection_per_class;
        if (max_detection_box_per_class > DEFAULT_MAX_YOLO_V5_DETECT_BOX_PER_CLASS)
            max_detection_box_per_class = DEFAULT_MAX_YOLO_V5_DETECT_BOX_PER_CLASS;
    }
    else
        max_detection_box_per_class = DEFAULT_MAX_YOLO_V5_DETECT_BOX_PER_CLASS;

    if (EX_NMS_MODE_END > yolo_pre_post_proc_config->nms_mode)
        nms_mode = yolo_pre_post_proc_config->nms_mode;
    else
        nms_mode = DEFAULT_NMS_MODE;

    uint32_t *p_anchors = (uint32_t *)yolo_pre_post_proc_config->data;
    if (yolo_pre_post_proc_config->anchor_layer_num * yolo_pre_post_proc_config->anchor_cell_num_per_layer > 0 &&
        yolo_pre_post_proc_config->anchor_layer_num <= EX_MAX_YOLO_ANCHOR_LAYER_NUM &&
        yolo_pre_post_proc_config->anchor_cell_num_per_layer <= EX_MAX_YOLO_ANCHOR_CELL_NUM_PER_LAYER)
    {
        for (int i = 0; i < yolo_pre_post_proc_config->anchor_layer_num; i++)
        {
            for (int j = 0; j < (yolo_pre_post_proc_config->anchor_cell_num_per_layer); j++)
            {
                anchors[i][j][0] = (float)*p_anchors++;
                anchors[i][j][1] = (float)*p_anchors++;
            }
        }
    }
    else
    {
        memcpy(anchors, default_anchors, sizeof(default_anchors));
    }

    /* get result buffer */
    struct ex_object_detection_result_s *result = (struct ex_object_detection_result_s *)(POSTPROC_RESULT_MEM_ADDR(image_p));

    /* init working buffer */
    yolo_v5_gp = (struct ex_yolo_v5_post_globals_s*)ex_get_gp((void**)&yolo_v5_gp, sizeof(struct ex_yolo_v5_post_globals_s));

    /* initialize */
    struct ex_bounding_box_s *boxes                                 = yolo_v5_gp->boxes;
    ngs_tensor_t *tensor_yolo                                       = NULL;
    ngs_quantization_parameters_v1_t *quantization_parameters_v1    = NULL;
    int32_t *shape_yolo                                             = NULL;
    float div                                                       = 0;
    float scale                                                     = 0;

    uint32_t scalar_index_list[4]                                   = {0};
    uint32_t scalar_index_list_len                                  = 4;

    int good_box_count                                              = 0;
    int max_candidate_idx                                           = 0;
    int min_candidate_idx                                           = 0;
    int class_num                                                   = 0;
    float stride_row                                                = 0;
    float stride_col                                                = 0;
    int prob_thresh_yolov5_fp                                       = 0;
    float box_confidence                                            = 0;
    int max_score_class                                             = 0;
    int8_t max_score_int                                            = 0;
    int8_t score_temp                                               = 0;
    float max_score                                                 = 0;
    float score                                                     = 0;
    float box_x                                                     = 0;
    float box_y                                                     = 0;
    float box_w                                                     = 0;
    float box_h                                                     = 0;
    struct ex_bounding_box_s new_candidate_item                     = {0};

    /* get class number */
    if (0 != ex_get_output_tensor(image_p, 0, &tensor_yolo))
        goto FUNC_OUT;

    shape_yolo = ex_get_tensor_shape(tensor_yolo);
    if (NULL == shape_yolo)
        goto FUNC_OUT;

    class_num           = (shape_yolo[1] / YOLO_V5_ANCHOR_NUM_PER_LAYER) - YOLO_V5_BOX_FIX_CH;
    result->class_count = class_num;

    /* retrieve ouput feature map & do post process */
    for (int idx = 0; idx < (int)POSTPROC_OUTPUT_NUM(image_p); idx++) {
        /* get output node */
        if (0 != ex_get_output_tensor(image_p, idx, &tensor_yolo))
            goto FUNC_OUT;

        /* get quantization params */
        quantization_parameters_v1 = ex_get_tensor_quantization_parameters_v1(tensor_yolo);
        if (NULL == quantization_parameters_v1)
            goto FUNC_OUT;

        /* get shape */
        shape_yolo = ex_get_tensor_shape(tensor_yolo);
        if (NULL == shape_yolo)
            goto FUNC_OUT;

        /* calculate quantization params */
        if (1 != quantization_parameters_v1->quantized_fixed_point_descriptor_num) {
            printf("error: not support channel-wise quantization\n");
            goto FUNC_OUT;
        }

        if (NGS_DTYPE_FLOAT32 != quantization_parameters_v1->quantized_fixed_point_descriptor[0].scale_dtype) {
            printf("error: only support float32 scale quantization params\n");
            goto FUNC_OUT;
        }

        div         = ex_pow2(quantization_parameters_v1->quantized_fixed_point_descriptor[0].radix);
        scale       = quantization_parameters_v1->quantized_fixed_point_descriptor[0].scale.scale_float32;
        scale       = 1.0f / (div * scale);

        /* calculate stride params */
        stride_row  = (float)DIM_INPUT_ROW(image_p, 0) / (float)shape_yolo[2];
        stride_col  = (float)DIM_INPUT_COL(image_p, 0) / (float)shape_yolo[3];

        /* convert threshold to fp for fast comparison */
        prob_thresh_yolov5_fp = floor(prob_threshold / scale);

        /* traverse every feature map point */
        for (int anchor_idx = 0; anchor_idx < YOLO_V5_ANCHOR_NUM_PER_LAYER; anchor_idx++) {
            for (int row = 0; row < (int)shape_yolo[2]; row++) {
                for (int col = 0; col < (int)shape_yolo[3]; col++) {
                    /* check if the score (4th channel) better than threshold */
                    scalar_index_list[0] = 0;
                    scalar_index_list[1] = anchor_idx * (class_num + YOLO_V5_BOX_FIX_CH) + 4;
                    scalar_index_list[2] = row;
                    scalar_index_list[3] = col;

                    box_confidence = (float)*ex_get_scalar_int8(tensor_yolo, scalar_index_list, scalar_index_list_len);

                    /* filter out small box score */
                    if (box_confidence <= prob_thresh_yolov5_fp)
                        continue;

                    /*
                    * find the maximum scores among all classes
                    * get the predicted class and score in fix
                    */
                    scalar_index_list[0]    = 0;
                    scalar_index_list[1]    = anchor_idx * (class_num + YOLO_V5_BOX_FIX_CH) + YOLO_V5_BOX_FIX_CH;
                    scalar_index_list[2]    = row;
                    scalar_index_list[3]    = col;

                    max_score_class         = 0;
                    max_score_int           = *ex_get_scalar_int8(tensor_yolo, scalar_index_list, scalar_index_list_len);
                    for (int i = 0; i < class_num; i++) {
                        scalar_index_list[0] = 0;
                        scalar_index_list[1] = anchor_idx * (class_num + YOLO_V5_BOX_FIX_CH) + YOLO_V5_BOX_FIX_CH + i;
                        scalar_index_list[2] = row;
                        scalar_index_list[3] = col;

                        score_temp = *ex_get_scalar_int8(tensor_yolo, scalar_index_list, scalar_index_list_len);
                        if (score_temp > max_score_int) {
                            max_score_int = score_temp;
                            max_score_class = i;
                        }
                    }

                    /* filter out small class number */
                    if (max_score_int <= prob_thresh_yolov5_fp)
                        continue;

                    /* get the confidence score in floating */
                    box_confidence  = ex_do_div_scale_optim(box_confidence, scale);
                    max_score       = ex_do_div_scale_optim((float)max_score_int, scale);
                    score           = max_score * box_confidence;

                    /* check if score is larger than threshold we set in floating */
                    if (score > prob_threshold) {
                        if ((MAX_YOLO_V5_DETECT_CANDIDATE_BOX == good_box_count) && (score <= boxes[min_candidate_idx].score))
                            continue;

                        scalar_index_list[0] = 0;
                        scalar_index_list[1] = anchor_idx * (class_num + YOLO_V5_BOX_FIX_CH);
                        scalar_index_list[2] = row;
                        scalar_index_list[3] = col;
                        box_x = (float)*ex_get_scalar_int8(tensor_yolo, scalar_index_list, scalar_index_list_len);

                        scalar_index_list[0] = 0;
                        scalar_index_list[1] = anchor_idx * (class_num + YOLO_V5_BOX_FIX_CH) + 1;
                        scalar_index_list[2] = row;
                        scalar_index_list[3] = col;
                        box_y = (float)*ex_get_scalar_int8(tensor_yolo, scalar_index_list, scalar_index_list_len);

                        scalar_index_list[0] = 0;
                        scalar_index_list[1] = anchor_idx * (class_num + YOLO_V5_BOX_FIX_CH) + 2;
                        scalar_index_list[2] = row;
                        scalar_index_list[3] = col;
                        box_w = (float)*ex_get_scalar_int8(tensor_yolo, scalar_index_list, scalar_index_list_len);

                        scalar_index_list[0] = 0;
                        scalar_index_list[1] = anchor_idx * (class_num + YOLO_V5_BOX_FIX_CH) + 3;
                        scalar_index_list[2] = row;
                        scalar_index_list[3] = col;
                        box_h = (float)*ex_get_scalar_int8(tensor_yolo, scalar_index_list, scalar_index_list_len);

                        box_x = ex_do_div_scale_optim(box_x, scale);
                        box_y = ex_do_div_scale_optim(box_y, scale);
                        box_w = ex_do_div_scale_optim(box_w, scale);
                        box_h = ex_do_div_scale_optim(box_h, scale);

                        box_x = (box_x * 2 - 0.5f + col) * stride_col;
                        box_y = (box_y * 2 - 0.5f + row) * stride_row;
                        box_w *= 2;
                        box_h *= 2;
                        box_w = box_w * box_w * anchors[idx][anchor_idx][0];
                        box_h = box_h * box_h * anchors[idx][anchor_idx][1];

                        /* update candidate box list */
                        memset((void*)&new_candidate_item, 0, sizeof(struct ex_bounding_box_s));

                        new_candidate_item.x1           = (box_x - (box_w / 2));
                        new_candidate_item.y1           = (box_y - (box_h / 2));
                        new_candidate_item.x2           = (box_x + (box_w / 2));
                        new_candidate_item.y2           = (box_y + (box_h / 2));

                        new_candidate_item.score        = score;
                        new_candidate_item.class_num    = max_score_class;

                        ex_update_candidate_bbox_list(&new_candidate_item,
                                                      MAX_YOLO_V5_DETECT_CANDIDATE_BOX,
                                                      boxes,
                                                      &good_box_count,
                                                      &max_candidate_idx,
                                                      &min_candidate_idx);
                    }
                }
            }
        }
    }

    /* do NMS */
    result->box_count = ex_nms_bbox(yolo_v5_gp->boxes,
                                    yolo_v5_gp->result_tmp_s,
                                    class_num,
                                    good_box_count,
                                    max_detection_box,
                                    max_detection_box_per_class,
                                    result->boxes,
                                    prob_threshold,
                                    iou_threshold,
                                    nms_mode);

    /* remap boxes coordinate back to original image */
    for (int i = 0; i < (int)result->box_count; i++)
        ex_remap_bbox(image_p, &result->boxes[i], 0);

FUNC_OUT:
    ex_free_gp((void**) &yolo_v5_gp);

    return result->box_count * sizeof(struct ex_bounding_box_s);
}
