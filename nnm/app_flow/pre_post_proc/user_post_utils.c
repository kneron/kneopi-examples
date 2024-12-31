/*
 * Utility functions for the postprocess functions.
 *
 * Copyright (C) 2021 Kneron, Inc. All rights reserved.
 *
 */
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include "ncpu_gen_struct.h"
#include "user_post_utils.h"

/******************************************************************
 * local define values
*******************************************************************/
#define KDP_COL_MIN     (16)        /**< bytes, i.e. 128 bits */
#define UNPASS_SCORE    (-999.0f)   /**< used as box filter */

/******************************************************************
 * function
*******************************************************************/
/**
 * @brief optimized dequantization function.
 */
float ex_do_div_scale_optim(float v, float scale) {
    return (v * scale);
}

/**
 * @brief get round up 16 value
 */
uint32_t ex_align_16(uint32_t num) {
    return ((num + (KDP_COL_MIN - 1)) & ~(KDP_COL_MIN - 1));
}

/**
 * @brief get the int8 scalar from tensor.
 */
int8_t *ex_get_scalar_int8(ngs_tensor_t* tensor, uint32_t* scalar_index_list, uint32_t scalar_index_list_len)
{
    int8_t *scalar                                      = NULL;
    int32_t npu_data_buf_offset                         = 0;

    ngs_tensor_shape_info_v2_t *tensor_shape_info_v2    = NULL;

    if ((NULL == tensor) ||
        (NULL == scalar_index_list)) {
        printf("get scalar int8 fail: NULL pointer paramaters ...\n");
        goto FUNC_OUT_ERROR;
    }

    if ((DRAM_FMT_16W1C8B == tensor->data_layout) ||
        (DRAM_FMT_1W16C8B == tensor->data_layout) ||
        (DRAM_FMT_RAW8B == tensor->data_layout)) {
        if (NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_1 == tensor->tensor_shape_info.version) {
            printf("unsupported model tensor shape info ...\n");
            goto FUNC_OUT_ERROR;
        } else if (NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_2 == tensor->tensor_shape_info.version) {
            tensor_shape_info_v2 = &(tensor->tensor_shape_info.tensor_shape_info_data.v2);

            if (tensor_shape_info_v2->shape_len != scalar_index_list_len) {
                printf("get scalar int8 fail: invalid scalar index list length ...\n");
                goto FUNC_OUT_ERROR;
            }

            for (int axis = 0; axis < (int)tensor_shape_info_v2->shape_len; axis++) {
                if ((int)scalar_index_list[axis] < tensor_shape_info_v2->shape[axis]) {
                    npu_data_buf_offset += scalar_index_list[axis] * tensor_shape_info_v2->stride_npu[axis];
                } else {
                    printf("get scalar int8 fail: invalid scalar index ...\n");
                    goto FUNC_OUT_ERROR;
                }
            }
        } else {
            printf("get scalar int8 fail: invalid source tensor shape version ...\n");
            goto FUNC_OUT_ERROR;
        }
    } else {
        printf("get scalar int8 fail: invalid NPU data layout ...\n");
        goto FUNC_OUT_ERROR;
    }

    scalar = &(((int8_t *)tensor->base_pointer)[npu_data_buf_offset]);

    return scalar;

FUNC_OUT_ERROR:
    return scalar;
}

/**
 * @brief get the tensor shape.
 */
int *ex_get_tensor_shape(ngs_tensor_t *tensor)
{
    int *shape                                          = NULL;
    ngs_tensor_shape_info_t *tensor_shape_info          = NULL;
    ngs_tensor_shape_info_v1_t *tensor_shape_info_v1    = NULL;
    ngs_tensor_shape_info_v2_t *tensor_shape_info_v2    = NULL;

    if (NULL == tensor){
        printf("get tensor shape fail: NULL pointer input parameter ...\n");
        goto FUNC_OUT;
    }

    tensor_shape_info       = &(tensor->tensor_shape_info);
    tensor_shape_info_v1    = &(tensor_shape_info->tensor_shape_info_data.v1);
    tensor_shape_info_v2    = &(tensor_shape_info->tensor_shape_info_data.v2);

    if (NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_1 == tensor_shape_info->version) {
        shape = tensor_shape_info_v1->shape_npu;
    } else if (NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_2 == tensor_shape_info->version) {
        shape = tensor_shape_info_v2->shape;
    } else {
        printf("get tensor shape fail: invalid source tensor shape information version ...\n");
        goto FUNC_OUT;
    }

FUNC_OUT:
    return shape;
}

/**
 * @brief get the tensor quantization parameters - version 1.
 */
ngs_quantization_parameters_v1_t *ex_get_tensor_quantization_parameters_v1(ngs_tensor_t *tensor)
{
    if (NGS_MODEL_QUANTIZATION_PARAMS_VERSION_1 == tensor->quantization_parameters.version)
        return &(tensor->quantization_parameters.quantization_parameters_data.v1);
    else
        printf("get tensor quantization parameter fail: invalid quantization parameter version ...");

    return NULL;
}

/**
 * @brief get the output tensor information.
 */
int ex_get_output_tensor(struct kdp_image_s *image_p, int tensor_index, ngs_tensor_t **output_tensor)
{
    int status = 0;

    if (NULL == image_p) {
        printf("get output tensor fail: NULL pointer paramaters ...\n");
        status = -1;
        goto FUNC_OUT;
    }

    if (NULL == POSTPROC_OUTPUT_TENSOR_LIST(image_p)) {
        printf("get output tensor fail: output tensor uninitialized ...\n");
        status = -1;
        goto FUNC_OUT;
    }

    if (tensor_index >= (int)POSTPROC_OUTPUT_NUM(image_p)) {
        printf("get output tensor fail: out of index ...\n");
        status = -1;
        goto FUNC_OUT;
    }

    *output_tensor = &(POSTPROC_OUTPUT_TENSOR_LIST(image_p)[tensor_index]);

    /* update tensor base addrress on the related address mode device */
    if (NGS_MODEL_TENSOR_ADDRESS_MODE_RELATIVE == (*output_tensor)->base_pointer_address_mode) {
        (*output_tensor)->base_pointer = (uintptr_t)(image_p->postproc.output_mem_addr + (*output_tensor)->base_pointer_offset);
    }

FUNC_OUT:
    return status;
}

/**
 * @brief power of two function.
 */
float ex_pow2(int exp)
{
    if (0 <= exp) {
        return (float)(0x1ULL << exp);
    } else {
        return (float)1 / (float)(0x1ULL << abs(exp));
    }
}

/**
 * @brief floating-point comparison.
 */
int ex_float_comparator(float a, float b) {
    float diff = a - b;

    if (diff < 0)
        return 1;
    else if (diff > 0)
        return -1;
    return 0;
}

/**
 * @brief calculate the area of a bounding box.
 */
float ex_box_area(struct ex_bounding_box_s *box)
{
    return fmax(0, box->y2 - box->y1 + 1) * fmax(0, box->x2 - box->x1 + 1);
}

/**
 * @brief score comparison of two bounding box.
 */
int ex_box_score_comparator(const void *pa, const void *pb)
{
    float a, b;

    a = ((struct ex_bounding_box_s *) pa)->score;
    b = ((struct ex_bounding_box_s *) pb)->score;

    /* take box with bigger area: only implement in YOLO V5 (python runner) */
    if (a == b) {
        float area_a = ex_box_area((struct ex_bounding_box_s *)pa);
        float area_b = ex_box_area((struct ex_bounding_box_s *)pb);
        return ex_float_comparator(area_a, area_b);
    }

    return ex_float_comparator(a, b);
}

/**
 * @brief calculate the intersection of two bounding box.
 */
float ex_box_intersection(struct ex_bounding_box_s *a, struct ex_bounding_box_s *b) {
    struct ex_bounding_box_s overlap;

    overlap.x1 = fmax(a->x1, b->x1);
    overlap.y1 = fmax(a->y1, b->y1);
    overlap.x2 = fmin(a->x2, b->x2);
    overlap.y2 = fmin(a->y2, b->y2);

    float area = ex_box_area(&overlap);
    return area;
}

/**
 * @brief calculate the union of two bounding box.
 */
float ex_box_union(struct ex_bounding_box_s *a, struct ex_bounding_box_s *b) {
    float i, u;

    i = ex_box_intersection(a, b);
    u = ex_box_area(a) + ex_box_area(b) - i;

    return u;
}

/**
 * @brief calculate the IoU of two bounding box.
 */
float ex_box_iou(struct ex_bounding_box_s *a, struct ex_bounding_box_s *b) {
    float c;
    float intersection_a_b = ex_box_intersection(a, b);
    float union_a_b = ex_box_union(a, b);

    c = intersection_a_b / union_a_b;

    return c;
}

/**
 * @brief performs NMS on the potential boxes
 */
int ex_nms_bbox(struct ex_bounding_box_s *potential_boxes,
                struct ex_bounding_box_s *temp_results,
                int class_num,
                int good_box_count,
                int max_boxes,
                int single_class_max_boxes,
                struct ex_bounding_box_s *results,
                float score_thresh,
                float iou_thresh,
                float nms_mode) {
    int good_result_count = 0;

    // check overlap between all boxes and not just those from same class
    if (nms_mode == EX_NMS_MODE_ALL_CLASS) {
        if (good_box_count == 1) {
            memcpy(&results[good_result_count], &potential_boxes[0], sizeof(struct ex_bounding_box_s));
            good_result_count++;
        } else if (good_box_count >= 2) {
            // sort boxes based on the score
            qsort(potential_boxes, good_box_count, sizeof(struct ex_bounding_box_s), ex_box_score_comparator);
            for (int j = 0; j < good_box_count; j++) {
                // if the box score is too low or is already filtered by previous box
                if (potential_boxes[j].score < score_thresh)
                    continue;

                // filter out overlapping, lower score boxes
                for (int k = j + 1; k < good_box_count; k++)
                    if (ex_box_iou(&potential_boxes[j], &potential_boxes[k]) > iou_thresh)
                        potential_boxes[k].score = UNPASS_SCORE;

                // keep boxes with highest scores, up to a certain amount
                memcpy(&results[good_result_count], &potential_boxes[j], sizeof(struct ex_bounding_box_s));
                good_result_count++;
                if (good_result_count == max_boxes)
                    break;
            }
        }
    } else {    // check overlap between only boxes from same class
        for (int i = 0; i < class_num; i++) {
            int class_good_result_count = 0;
            if (good_result_count == max_boxes) // break out of outer loop as well for future classes
                break;

            int class_good_box_count = 0;

            // find all boxes of a specific class
            for (int j = 0; j < good_box_count; j++) {
                if (potential_boxes[j].class_num == i) {
                    memcpy(&temp_results[class_good_box_count], &potential_boxes[j], sizeof(struct ex_bounding_box_s));
                    class_good_box_count++;
                }
            }

            if (class_good_box_count == 1) {
                memcpy(&results[good_result_count], temp_results, sizeof(struct ex_bounding_box_s));
                good_result_count++;
            } else if (class_good_box_count >= 2) {
                // sort boxes based on the score
                qsort(temp_results, class_good_box_count, sizeof(struct ex_bounding_box_s), ex_box_score_comparator);
                for (int j = 0; j < class_good_box_count; j++) {
                    // if the box score is too low or is already filtered by previous box
                    if (temp_results[j].score < score_thresh)
                        continue;

                    // filter out overlapping, lower score boxes
                    for (int k = j + 1; k < class_good_box_count; k++)
                        if (ex_box_iou(&temp_results[j], &temp_results[k]) > iou_thresh)
                            temp_results[k].score = UNPASS_SCORE;

                    // keep boxes with highest scores, up to a certain amount
                    if ((good_result_count == max_boxes) || (class_good_result_count == single_class_max_boxes))
                        break;
                    memcpy(&results[good_result_count], &temp_results[j], sizeof(struct ex_bounding_box_s));
                    good_result_count++;
                    class_good_result_count++;
                }
            }
        }
    }

    return good_result_count;
}

/**
 * @brief update candidate bbox list, reserve top max_candidate_num candidate bbox.
 */
int ex_update_candidate_bbox_list(struct ex_bounding_box_s *new_candidate_bbox,
                                  int max_candidate_num,
                                  struct ex_bounding_box_s *candidate_bbox_list,
                                  int *candidate_bbox_num,
                                  int *max_candidate_idx,
                                  int *min_candidate_idx) {

    if ((NULL == new_candidate_bbox) || (NULL == candidate_bbox_list))
        return -1;

    int update_idx = -1;

    if (0 == *candidate_bbox_num) {
        /** add 1-th bbox */
        *max_candidate_idx = 0;
        *min_candidate_idx = 0;
        update_idx = 0;
        (*candidate_bbox_num)++;
        memcpy(&candidate_bbox_list[update_idx], new_candidate_bbox, sizeof(struct ex_bounding_box_s));
    } else {
        if (max_candidate_num > *candidate_bbox_num) {
            /** directly add bbox when the candidate bbox list is not filled */
            update_idx = *candidate_bbox_num;

            if (new_candidate_bbox->score > candidate_bbox_list[*max_candidate_idx].score)
                *max_candidate_idx = update_idx;
            else if (new_candidate_bbox->score < candidate_bbox_list[*min_candidate_idx].score)
                *min_candidate_idx = update_idx;

            (*candidate_bbox_num)++;

            if (0 <= update_idx)
                memcpy(&candidate_bbox_list[update_idx], new_candidate_bbox, sizeof(struct ex_bounding_box_s));
        } else {
            /** update candidate bbox list when candidate bbox list is filled */
            if (new_candidate_bbox->score >= candidate_bbox_list[*max_candidate_idx].score) {
                /** update the largest score candidate index */
                update_idx = *min_candidate_idx;
                *max_candidate_idx = *min_candidate_idx;
            } else if (new_candidate_bbox->score > candidate_bbox_list[*min_candidate_idx].score) {
                update_idx = *min_candidate_idx;
            }

            if (0 <= update_idx) {
                memcpy(&candidate_bbox_list[update_idx], new_candidate_bbox, sizeof(struct ex_bounding_box_s));

                for (int i = 0; i < *candidate_bbox_num; i++) {
                    /** update the smallest score candidate index */
                    if (candidate_bbox_list[i].score < candidate_bbox_list[*min_candidate_idx].score)
                        *min_candidate_idx = i;
                }
            }
        }
    }

    return 0;
}

/**
 * @brief remap one bounding box to original image coordinates.
 */
void ex_remap_bbox(struct kdp_image_s *image_p, struct ex_bounding_box_s *box, int need_scale)
{
    // original box values are percentages, scale to model sizes
    if (need_scale) {
        box->x1 *= DIM_INPUT_COL(image_p, 0);
        box->y1 *= DIM_INPUT_ROW(image_p, 0);
        box->x2 *= DIM_INPUT_COL(image_p, 0);
        box->y2 *= DIM_INPUT_ROW(image_p, 0);
    }

    // scale from model sizes to original input sizes
    box->x1 = (box->x1 - RAW_PAD_LEFT(image_p, 0)) * RAW_SCALE_WIDTH(image_p, 0) + RAW_CROP_LEFT(image_p, 0);
    box->y1 = (box->y1 - RAW_PAD_TOP(image_p, 0)) * RAW_SCALE_HEIGHT(image_p, 0) + RAW_CROP_TOP(image_p, 0);
    box->x2 = (box->x2 - RAW_PAD_LEFT(image_p, 0)) * RAW_SCALE_WIDTH(image_p, 0) + RAW_CROP_LEFT(image_p, 0);
    box->y2 = (box->y2 - RAW_PAD_TOP(image_p, 0)) * RAW_SCALE_HEIGHT(image_p, 0) + RAW_CROP_TOP(image_p, 0);

    // rounding
    box->x1 = (int)(box->x1 + (float)0.5);
    box->y1 = (int)(box->y1 + (float)0.5);
    box->x2 = (int)(box->x2 + (float)0.5);
    box->y2 = (int)(box->y2 + (float)0.5);

    // clip to boundaries of image
    box->x1 = (int)(box->x1 < 0 ? 0 : box->x1);
    box->y1 = (int)(box->y1 < 0 ? 0 : box->y1);
    box->x2 = (int)((box->x2 >= RAW_INPUT_COL(image_p, 0) ? (RAW_INPUT_COL(image_p, 0) - 1) : box->x2));
    box->y2 = (int)((box->y2 >= RAW_INPUT_ROW(image_p, 0) ? (RAW_INPUT_ROW(image_p, 0) - 1) : box->y2));
}
