/**
 * @file        postprocess.c
 * @brief       post-process functions
 * @version     0.1
 * @date        2021-03-22
 *
 * @copyright   Copyright (c) 2021 Kneron Inc. All rights reserved.
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "postprocess.h"

#define YOLO_V3_CELL_BOX_NUM 3
#define YOLO_V3_BOX_FIX_CH 5
#define NMS_THRESH_YOLOV3_520 0.45
#define NMS_THRESH_YOLOV5_720 0.5
#define MAX_POSSIBLE_BOXES 2000
#define MODEL_SHIRNK_RATIO_TYV3 32
#define MODEL_SHIRNK_RATIO_V5 8
#define YOLO_MAX_DETECTION_PER_CLASS 100

/* IOU Methods */
enum IOU_TYPE
{
    IOU_UNION = 0,
    IOU_MIN,
};

const float yolo_v3_anchers[3][3][2] = {
    {{81, 82}, {135, 169}, {344, 319}},
    {{23, 27}, {37, 58}, {81, 82}},
    {{0, 0}, {0, 0}, {0, 0}}};

const float yolo_v5_anchers[3][3][2] = {
    {{10, 13}, {16, 30}, {33, 23}},
    {{30, 61}, {62, 45}, {59, 119}},
    {{116, 90}, {156, 198}, {373, 326}}};

static float sigmoid(float x)
{
    float exp_value;
    float return_value;

    exp_value = exp(-x);

    return_value = 1 / (1 + exp_value);

    return return_value;
}

static int float_comparator(float float_num_1, float float_num_2)
{
    float diff = float_num_1 - float_num_2;

    if (diff < 0)
        return 1;
    else if (diff > 0)
        return -1;
    return 0;
}

static int int_comparator(int int_num_1, int int_num_2)
{
    int diff = int_num_1 - int_num_2;

    if (diff < 0)
        return -1;
    else if (diff > 0)
        return 1;
    return 0;
}

static int box_comparator(const void *box_1, const void *box_2)
{
    kp_bounding_box_t *_box_1 = (kp_bounding_box_t *)box_1;
    kp_bounding_box_t *_box_2 = (kp_bounding_box_t *)box_2;

    int res = float_comparator(_box_1->score, _box_2->score);
    if (res != 0)
        return res;

    res = int_comparator(_box_1->x1, _box_2->x1);
    if (res != 0)
        return res;

    res = int_comparator(_box_1->y1, _box_2->y1);
    if (res != 0)
        return res;

    res = int_comparator(_box_1->x2, _box_2->x2);
    if (res != 0)
        return res;

    return int_comparator(_box_1->y2, _box_2->y2);
}

static float overlap(float l1, float r1, float l2, float r2)
{
    float left = l1 > l2 ? l1 : l2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_intersection(kp_bounding_box_t *a, kp_bounding_box_t *b)
{
    float w, h, area;

    w = overlap(a->x1, a->x2, b->x1, b->x2);
    h = overlap(a->y1, a->y2, b->y1, b->y2);

    if (w < 0 || h < 0)
        return 0;

    area = w * h;
    return area;
}

static float box_union(kp_bounding_box_t *a, kp_bounding_box_t *b)
{
    float i, u;

    i = box_intersection(a, b);
    u = (a->y2 - a->y1) * (a->x2 - a->x1) + (b->y2 - b->y1) * (b->x2 - b->x1) - i;

    return u;
}

static float box_iou(kp_bounding_box_t *a, kp_bounding_box_t *b, int nms_type)
{
    float c = 0.;
    switch (nms_type)
    {
    case IOU_MIN:
        if (box_intersection(a, b) / box_intersection(a, a) > box_intersection(a, b) / box_intersection(b, b))
        {
            c = box_intersection(a, b) / box_intersection(a, a);
        }
        else
        {
            c = box_intersection(a, b) / box_intersection(b, b);
        }
        break;
    default:
        if (c < box_intersection(a, b) / box_union(a, b))
        {
            c = box_intersection(a, b) / box_union(a, b);
        }
        break;
    }

    return c;
}

void boxes_scale(kp_bounding_box_t *boxes, int size, kp_hw_pre_proc_info_t *pre_proc_info)
{
    int img_width = pre_proc_info->img_width;
    int img_height = pre_proc_info->img_height;
    int pad_left = pre_proc_info->pad_left;
    int pad_top = pre_proc_info->pad_top;
    float ratio_w = (float)img_width / pre_proc_info->resized_img_width;
    float ratio_h = (float)img_height / pre_proc_info->resized_img_height;

    for (int i = 0; i < size; i++)
    {
        boxes[i].x2 = (boxes[i].x2 - boxes[i].x1) * ratio_w; // w
        boxes[i].y2 = (boxes[i].y2 - boxes[i].y1) * ratio_h; // h
        boxes[i].x1 -= pad_left;
        boxes[i].y1 -= pad_top;
        boxes[i].x1 *= ratio_w;
        boxes[i].y1 *= ratio_h;
        boxes[i].x2 += boxes[i].x1;
        boxes[i].y2 += boxes[i].y1;

        // limit Rectangle
        boxes[i].x1 = ((int)(boxes[i].x1 + 0.5) > 0) ? (int)(boxes[i].x1 + 0.5) : 0;
        boxes[i].y1 = ((int)(boxes[i].y1 + 0.5) > 0) ? (int)(boxes[i].y1 + 0.5) : 0;
        boxes[i].x2 = ((int)(boxes[i].x2 + 0.5) < (img_width - 1)) ? (int)(boxes[i].x2 + 0.5) : img_width - 1;
        boxes[i].y2 = ((int)(boxes[i].y2 + 0.5) < (img_height - 1)) ? (int)(boxes[i].y2 + 0.5) : img_height - 1;
    }
}

int post_process_yolo_v3(kp_inf_float_node_output_t *node_output[], int num_output_node,
                         kp_hw_pre_proc_info_t *pre_proc_info, float thresh_value, kp_yolo_result_t *yoloResult)
{
    int class_count = (node_output[0]->shape[1] / YOLO_V3_CELL_BOX_NUM) - YOLO_V3_BOX_FIX_CH;
    float *box_class_probs = NULL;
    kp_bounding_box_t *possible_boxes = NULL;
    kp_bounding_box_t *temp_boxes = NULL;
    int good_box_count = 0;
    int good_result_count = 0;

    box_class_probs = (float *)malloc(class_count * sizeof(float));
    if (NULL == box_class_probs ) {
        printf("Error! %s(): malloc memory for probs failed\n", __FUNCTION__);
        goto err;
    }

    possible_boxes = (kp_bounding_box_t *)malloc(MAX_POSSIBLE_BOXES * sizeof(kp_bounding_box_t));
    if (NULL == possible_boxes) {
        printf("Error! %s(): malloc memory for boxes failed\n", __FUNCTION__);
        goto err;
    }
    temp_boxes = (kp_bounding_box_t *)malloc(MAX_POSSIBLE_BOXES * sizeof(kp_bounding_box_t));
    if (NULL == temp_boxes) {
        printf("Error! %s(): malloc memory for temp boxes failed\n", __FUNCTION__);
        goto err;
    }

    for (int i = 0; i < num_output_node; i++)
    {
        int grid_w = node_output[i]->shape[3];
        int grid_h = node_output[i]->shape[2];
        int grid_c = node_output[i]->shape[1];

        int width_size = grid_w * grid_c;
        int anchor_offset = width_size / 3;

        float ratio_w = (float)pre_proc_info->model_input_width / grid_w;
        float ratio_h = (float)pre_proc_info->model_input_height / grid_h;

        for (int row = 0; row < grid_h; row++)
        {
            for (int an = 0; an < YOLO_V3_CELL_BOX_NUM; an++)
            {
                float *data = node_output[i]->data + row * width_size + an * anchor_offset;

                float *x_p = data;
                float *y_p = x_p + grid_w;
                float *width_p = y_p + grid_w;
                float *height_p = width_p + grid_w;
                float *score_p = height_p + grid_w;
                float *class_p = score_p + grid_w;

                for (int col = 0; col < grid_w; col++)
                {
                    float box_x = *(x_p + col);
                    float box_y = *(y_p + col);
                    float box_w = *(width_p + col);
                    float box_h = *(height_p + col);
                    float box_confidence = sigmoid(*(score_p + col));

                    float x1, y1, x2, y2;
                    bool first_box = false;

                    for (int j = 0; j < class_count; j++)
                    {
                        box_class_probs[j] = (float)*(class_p + col + j * grid_w);
                    }

                    /* Get scores of all class */
                    for (int j = 0; j < class_count; j++)
                    {
                        float max_score = sigmoid(box_class_probs[j]) * box_confidence;
                        if (max_score >= thresh_value)
                        {
                            if (!first_box)
                            {
                                first_box = true;

                                box_x = (sigmoid(box_x) + col) * ratio_w;
                                box_y = (sigmoid(box_y) + row) * ratio_h;
                                box_w = exp(box_w) * yolo_v3_anchers[i][an][0];
                                box_h = exp(box_h) * yolo_v3_anchers[i][an][1];

                                x1 = box_x - (box_w / 2);
                                y1 = box_y - (box_h / 2);
                                x2 = box_x + (box_w / 2);
                                y2 = box_y + (box_h / 2);
                            }

                            possible_boxes[good_box_count].x1 = x1;
                            possible_boxes[good_box_count].y1 = y1;
                            possible_boxes[good_box_count].x2 = x2;
                            possible_boxes[good_box_count].y2 = y2;
                            possible_boxes[good_box_count].score = max_score;
                            possible_boxes[good_box_count].class_num = j;
                            good_box_count++;

                            if (good_box_count >= MAX_POSSIBLE_BOXES)
                            {
                                printf("post yolo v3: error ! aborted due to too many boxes\n");
                                goto err;
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < class_count; i++)
    {
        kp_bounding_box_t *bbox = possible_boxes;
        kp_bounding_box_t *r_tmp_p = temp_boxes;

        int class_good_box_count = 0;

        for (int j = 0; j < good_box_count; j++)
        {
            if (bbox->class_num == i)
            {
                memcpy(r_tmp_p, bbox, sizeof(kp_bounding_box_t));
                r_tmp_p++;
                class_good_box_count++;
            }
            bbox++;
        }

        if (class_good_box_count == 1)
        {
            if (good_result_count < YOLO_GOOD_BOX_MAX)
            {
                memcpy(&(yoloResult->boxes[good_result_count]), &temp_boxes[0], sizeof(kp_bounding_box_t));
                good_result_count++;
            }
        }
        else if (class_good_box_count >= 2)
        {
            qsort(temp_boxes, class_good_box_count, sizeof(kp_bounding_box_t), box_comparator);
            for (int j = 0; j < class_good_box_count; j++)
            {
                if (temp_boxes[j].score == 0)
                    continue;
                for (int k = j + 1; k < class_good_box_count; k++)
                {
                    if (box_iou(&temp_boxes[j], &temp_boxes[k], IOU_UNION) > NMS_THRESH_YOLOV3_520)
                    {
                        temp_boxes[k].score = 0;
                    }
                }
            }

            int good_count = 0;
            for (int j = 0; j < class_good_box_count; j++)
            {
                if (temp_boxes[j].score > 0 && good_result_count < YOLO_GOOD_BOX_MAX)
                {
                    memcpy(&(yoloResult->boxes[good_result_count]), &temp_boxes[j], sizeof(kp_bounding_box_t));
                    good_result_count++;
                    good_count++;
                }
                if (YOLO_MAX_DETECTION_PER_CLASS == good_count)
                {
                    break;
                }
            }
        }

        // FIXME: find a better policy to filter the detected bounding box result if total box count exceeds YOLO_GOOD_BOX_MAX
        if (good_result_count >= YOLO_GOOD_BOX_MAX)
            break;
    }

    yoloResult->box_count = good_result_count;
    yoloResult->class_count = class_count;

    // convert the coordinate of all bounding boxes to raw image
    boxes_scale(yoloResult->boxes, yoloResult->box_count, pre_proc_info);

    free(box_class_probs);
    free(possible_boxes);
    free(temp_boxes);

    return 0;

err:
    free(box_class_probs);
    free(possible_boxes);
    free(temp_boxes);

    return -1;
}

int post_process_yolo_v5_520(kp_inf_float_node_output_t *node_output[], int num_output_node,
                             kp_hw_pre_proc_info_t *pre_proc_info, float thresh_value, kp_yolo_result_t *yoloResult)
{
    int class_count = (node_output[0]->shape[1] / YOLO_V3_CELL_BOX_NUM) - YOLO_V3_BOX_FIX_CH;
    float *box_class_probs = NULL;
    kp_bounding_box_t *possible_boxes = NULL;
    kp_bounding_box_t *temp_boxes = NULL;

    int good_box_count = 0;
    int good_result_count = 0;

    box_class_probs = (float *)malloc(class_count * sizeof(float));
    if (NULL == box_class_probs) {
        printf("error! malloc %s temp space failed\n", "box class probs");
        goto err;
    }

    possible_boxes = (kp_bounding_box_t *)malloc(MAX_POSSIBLE_BOXES * sizeof(kp_bounding_box_t));
    if (NULL == possible_boxes){
        printf("error! malloc %s temp space failed\n", "possible_boxes");
        goto err;
    }
    temp_boxes = (kp_bounding_box_t *)malloc(MAX_POSSIBLE_BOXES * sizeof(kp_bounding_box_t));
    if (NULL == temp_boxes){
        printf("error! malloc %s temp space failed\n", "temp_boxes");
        goto err;
    }

    for (int i = 0; i < num_output_node; i++)
    {
        int grid_w = node_output[i]->shape[3];
        int grid_h = node_output[i]->shape[2];
        int grid_c = node_output[i]->shape[1];

        int width_size = grid_w * grid_c;
        int anchor_offset = width_size / 3;

        float ratio_w = (float)pre_proc_info->model_input_width / grid_w;
        float ratio_h = (float)pre_proc_info->model_input_height / grid_h;

        for (int row = 0; row < grid_h; row++)
        {
            for (int an = 0; an < YOLO_V3_CELL_BOX_NUM; an++)
            {
                float *data = node_output[i]->data + row * width_size + an * anchor_offset;

                float *x_p = data;
                float *y_p = x_p + grid_w;
                float *width_p = y_p + grid_w;
                float *height_p = width_p + grid_w;
                float *score_p = height_p + grid_w;
                float *class_p = score_p + grid_w;

                for (int col = 0; col < grid_w; col++)
                {
                    float box_x = *(x_p + col);
                    float box_y = *(y_p + col);
                    float box_w = *(width_p + col);
                    float box_h = *(height_p + col);
                    float box_confidence = sigmoid(*(score_p + col));

                    float x1, y1, x2, y2;
                    bool first_box = false;

                    for (int j = 0; j < class_count; j++)
                    {
                        box_class_probs[j] = (float)*(class_p + col + j * grid_w);
                    }

                    /* Get scores of all class */
                    for (int j = 0; j < class_count; j++)
                    {
                        float max_score = sigmoid(box_class_probs[j]) * box_confidence;
                        if (max_score >= thresh_value)
                        {
                            if (!first_box)
                            {
                                first_box = true;

                                box_x = sigmoid(box_x);
                                box_y = sigmoid(box_y);
                                box_w = sigmoid(box_w);
                                box_h = sigmoid(box_h);

                                box_x = ((box_x * 2 - 0.5f + col) * ratio_w);
                                box_y = ((box_y * 2 - 0.5f + row) * ratio_h);
                                box_w *= 2;
                                box_h *= 2;
                                box_w = box_w * box_w * yolo_v5_anchers[i][an][0];
                                box_h = box_h * box_h * yolo_v5_anchers[i][an][1];

                                x1 = (box_x - (box_w / 2));
                                y1 = (box_y - (box_h / 2));
                                x2 = (box_x + (box_w / 2));
                                y2 = (box_y + (box_h / 2));
                            }

                            possible_boxes[good_box_count].x1 = x1;
                            possible_boxes[good_box_count].y1 = y1;
                            possible_boxes[good_box_count].x2 = x2;
                            possible_boxes[good_box_count].y2 = y2;
                            possible_boxes[good_box_count].score = max_score;
                            possible_boxes[good_box_count].class_num = j;
                            good_box_count++;

                            if (good_box_count >= MAX_POSSIBLE_BOXES)
                            {
                                printf("post yolo v5: error ! aborted due to too many boxes\n");
                                goto err;
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < class_count; i++)
    {
        kp_bounding_box_t *bbox = possible_boxes;
        kp_bounding_box_t *r_tmp_p = temp_boxes;

        int class_good_box_count = 0;

        for (int j = 0; j < good_box_count; j++)
        {
            if (bbox->class_num == i)
            {
                memcpy(r_tmp_p, bbox, sizeof(kp_bounding_box_t));
                r_tmp_p++;
                class_good_box_count++;
            }
            bbox++;
        }

        if (class_good_box_count == 1)
        {
            if (good_result_count < YOLO_GOOD_BOX_MAX)
            {
                memcpy(&(yoloResult->boxes[good_result_count]), &temp_boxes[0], sizeof(kp_bounding_box_t));
                good_result_count++;
            }
        }
        else if (class_good_box_count >= 2)
        {
            qsort(temp_boxes, class_good_box_count, sizeof(kp_bounding_box_t), box_comparator);
            for (int j = 0; j < class_good_box_count; j++)
            {
                if (temp_boxes[j].score == 0)
                    continue;
                for (int k = j + 1; k < class_good_box_count; k++)
                {
                    if (box_iou(&temp_boxes[j], &temp_boxes[k], IOU_UNION) > NMS_THRESH_YOLOV3_520)
                    {
                        temp_boxes[k].score = 0;
                    }
                }
            }

            int good_count = 0;
            for (int j = 0; j < class_good_box_count; j++)
            {
                if (temp_boxes[j].score > 0 && good_result_count < YOLO_GOOD_BOX_MAX)
                {
                    memcpy(&(yoloResult->boxes[good_result_count]), &temp_boxes[j], sizeof(kp_bounding_box_t));
                    good_result_count++;
                    good_count++;
                }
                if (YOLO_MAX_DETECTION_PER_CLASS == good_count)
                {
                    break;
                }
            }
        }

        // FIXME: find a better policy to filter the detected bounding box result if total box count exceeds YOLO_GOOD_BOX_MAX
        if (good_result_count >= YOLO_GOOD_BOX_MAX)
            break;
    }

    yoloResult->box_count = good_result_count;
    yoloResult->class_count = class_count;

    // convert the coordinate of all bounding boxes to raw image
    boxes_scale(yoloResult->boxes, yoloResult->box_count, pre_proc_info);

    free(box_class_probs);
    free(possible_boxes);
    free(temp_boxes);

    return 0;

err:
    free(box_class_probs);
    free(possible_boxes);
    free(temp_boxes);

    return -1;
}

// Use to record candidate bounding box properties of each class
typedef struct candidate_boxes
{
    int boxes_count;
    float *boxes; // size should be maximum number of possible bounding boxes
    float **scores; // size should be yolo_v5 class number * maximum number of possible bounding boxes
} candidate_boxes;

int post_process_yolo_v5_720(kp_inf_float_node_output_t *node_output[], int num_output_node,
                             kp_hw_pre_proc_info_t *pre_proc_info, float thresh_value, kp_yolo_result_t *yoloResult)
{
    int class_count = (node_output[0]->shape[1] / YOLO_V3_CELL_BOX_NUM) - YOLO_V3_BOX_FIX_CH;
    int offset = 0;
    float *updated_boxes = NULL;
    kp_bounding_box_t *temp_boxes = NULL;
    int good_result_count = 0;
    candidate_boxes cand_boxes = {0};
    cand_boxes.boxes_count = 0;

    updated_boxes = (float *)malloc(node_output[0]->shape[3] * node_output[0]->shape[2] * 4 * sizeof(float));
    if (NULL == updated_boxes){
        printf("error! malloc failed\n");
        goto err;
    }

    for (int i = 0; i < num_output_node; i++)
        cand_boxes.boxes_count += node_output[i]->shape[3] * node_output[i]->shape[2] * 3;

    cand_boxes.boxes = (float *)malloc(cand_boxes.boxes_count * 4 * sizeof(float)); // 4 means (x1, y1, x2, y2)
    if(NULL == cand_boxes.boxes) {
        printf("error! malloc failed\n");
        goto err;
    }
    cand_boxes.scores = (float **)malloc(class_count * sizeof(float *));
    if(NULL == cand_boxes.scores) {
        printf("error! malloc failed\n");
        goto err;
    }

    for (int i = 0; i < class_count; i++) {
        cand_boxes.scores[i] = (float*)malloc(cand_boxes.boxes_count * sizeof(float));
        if(NULL == cand_boxes.scores[i]) {
            printf("error! malloc failed\n");
            goto err;
        }
    }

    for (int i = 0; i < num_output_node; i++)
    {
        int ratio_w = pre_proc_info->model_input_width / node_output[i]->shape[3];
        int ratio_h = pre_proc_info->model_input_height / node_output[i]->shape[2];
        int nrows = node_output[i]->shape[2];
        int ncols = node_output[i]->shape[3];
        int nchs = node_output[i]->shape[1];

        // following comments are from kdp
        // node 0 : 1 x 255 x 80 x 80, reshape to 1 x 3 x 85 x 80 x 80, transpose to 1 x 3 x 80 x 80 x 85
        // node 1 : 1 x 255 x 40 x 40, reshape to 1 x 3 x 85 x 40 x 40, transpose to 1 x 3 x 40 x 40 x 85
        // node 2 : 1 x 255 x 20 x 20, reshape to 1 x 3 x 85 x 20 x 20, transpose to 1 x 3 x 20 x 20 x 85
        int stride1 = (nchs / YOLO_V3_CELL_BOX_NUM) * nrows * ncols;
        int stride2 = nrows * ncols;
        int stride3 = ncols;

        // scan scores -> divide to 3 (YOLO_V3_CELL_BOX_NUM) hunks of data
        for (int k = 0; k < YOLO_V3_CELL_BOX_NUM; k++)
        {
            // update anchor
            for (int row = 0; row < nrows; row++)
            {
                for (int col = 0; col < ncols; col++)
                {
                    float *boxes = &node_output[i]->data[k * stride1];

                    int index = row * stride3 + col;
                    float box_x = boxes[index + 0 * stride2];
                    float box_y = boxes[index + 1 * stride2];
                    float box_w = boxes[index + 2 * stride2];
                    float box_h = boxes[index + 3 * stride2];
                    float grid_x = (float)col;
                    float grid_y = (float)row;

                    box_w = (box_w * box_w);
                    box_h = (box_h * box_h);
                    float _x = (box_x * 2 - 0.5 + grid_x) * ratio_w;
                    float _y = (box_y * 2 - 0.5 + grid_y) * ratio_h;
                    float _w = box_w * 4 * yolo_v5_anchers[i][k][0];
                    float _h = box_h * 4 * yolo_v5_anchers[i][k][1];
                    float xleft = (_x - _w / 2);
                    float yleft = (_y - _h / 2);

                    updated_boxes[4 * index + 0] = xleft;
                    updated_boxes[4 * index + 1] = yleft;
                    updated_boxes[4 * index + 2] = xleft + _w;
                    updated_boxes[4 * index + 3] = yleft + _h;
                }
            }

            // Collect all boxes candidates from 3 nodes and put them in a 1D long array
            memcpy(&cand_boxes.boxes[(offset + k * nrows * ncols) * 4], updated_boxes, 4 * nrows * ncols * sizeof(float));

            // Find probability of each bounding box
            float *box_prob = &node_output[i]->data[k * stride1 + 4 * stride2];

            // Find probability of each class
            for (int c = YOLO_V3_BOX_FIX_CH; c < nchs / YOLO_V3_CELL_BOX_NUM; c++)
            {
                float *class_prob = &node_output[i]->data[k * stride1 + c * stride2];
                int count = 0;
                for (int row = 0; row < nrows; row++)
                {
                    for (int col = 0; col < ncols; col++)
                    {
                        // Find score by multiplying probability of each bounding box and that of each class
                        cand_boxes.scores[c - YOLO_V3_BOX_FIX_CH][offset + k * nrows * ncols + count] = box_prob[row * ncols + col] * class_prob[row * ncols + col];
                        count++;
                    }
                }
            }
        }
        offset += (YOLO_V3_CELL_BOX_NUM * nrows * ncols);
    }

    temp_boxes = (kp_bounding_box_t *)malloc(cand_boxes.boxes_count * sizeof(kp_bounding_box_t));
    if (NULL == temp_boxes) {
        printf("error! malloc temp working buffer failed\n");
        goto err;
    }

    for (int i = 0; i < class_count; i++)
    {
        kp_bounding_box_t *r_tmp_p = temp_boxes;

        int class_good_box_count = 0;

        for (int box_idx = 0; box_idx < cand_boxes.boxes_count; box_idx++)
        {
            if (cand_boxes.scores[i][box_idx] > thresh_value)
            {
                memcpy(r_tmp_p, cand_boxes.boxes + 4 * box_idx, 4 * sizeof(float)); // copy (x1, y1, x2, y2) //-V::512
                r_tmp_p->class_num = i;
                r_tmp_p->score = cand_boxes.scores[i][box_idx];

                r_tmp_p++;
                class_good_box_count++;
            }
        }

        if (class_good_box_count == 1)
        {
            if (good_result_count < YOLO_GOOD_BOX_MAX)
            {
                memcpy(&(yoloResult->boxes[good_result_count]), &temp_boxes[0], sizeof(kp_bounding_box_t));
                good_result_count++;
            }
        }
        else if (class_good_box_count >= 2)
        {
            qsort(temp_boxes, class_good_box_count, sizeof(kp_bounding_box_t), box_comparator);
            for (int j = 0; j < class_good_box_count; j++)
            {
                if (temp_boxes[j].score == 0)
                    continue;
                for (int k = j + 1; k < class_good_box_count; k++)
                {
                    if (box_iou(&temp_boxes[j], &temp_boxes[k], IOU_UNION) > NMS_THRESH_YOLOV5_720)
                    {
                        temp_boxes[k].score = 0;
                    }
                }
            }

            int good_count = 0;
            for (int j = 0; j < class_good_box_count; j++)
            {
                if (temp_boxes[j].score > 0 && good_result_count < YOLO_GOOD_BOX_MAX)
                {
                    memcpy(&(yoloResult->boxes[good_result_count]), &temp_boxes[j], sizeof(kp_bounding_box_t));
                    good_result_count++;
                    good_count++;
                }
                if (YOLO_MAX_DETECTION_PER_CLASS == good_count)
                {
                    break;
                }
            }
        }

        // FIXME: find a better policy to filter the detected bounding box result if total box count exceeds YOLO_GOOD_BOX_MAX
        if (good_result_count >= YOLO_GOOD_BOX_MAX)
            break;
    }

    yoloResult->box_count = good_result_count;
    yoloResult->class_count = class_count;

    // convert the coordinate of all bounding boxes to raw image
    boxes_scale(yoloResult->boxes, yoloResult->box_count, pre_proc_info);

    free(updated_boxes);

    free(cand_boxes.boxes);
    for(int i = 0; i < class_count; i++)
        free(cand_boxes.scores[i]);
    free(cand_boxes.scores);

    free(temp_boxes);

    return 0;

err:
    if (NULL != updated_boxes) {
        free(updated_boxes);
    }

    if (NULL != cand_boxes.boxes) {
        free(cand_boxes.boxes);
    }

    if (NULL != cand_boxes.scores) {
        for (int i = 0; i < class_count; i++) {
            if (NULL != cand_boxes.scores[i]) {
                free(cand_boxes.scores[i]);
            }
        }

        free(cand_boxes.scores);
    }

    return 0;
}
