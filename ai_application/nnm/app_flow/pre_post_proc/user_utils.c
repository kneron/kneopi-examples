/*
 * Utility functions for the postprocess functions.
 *
 * Copyright (C) 2021 Kneron, Inc. All rights reserved.
 *
 */
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <float.h>
#include <limits.h>

#include "ncpu_gen_struct.h"
#include "user_utils.h"

/******************************************************************
 * local define values
*******************************************************************/
#define KDP_COL_MIN     (16)        /**< bytes, i.e. 128 bits */
#define UNPASS_SCORE    (-999.0f)   /**< used as box filter */

/******************************************************************
 * local util function
*******************************************************************/
static int _get_quantization_parameters_v1_information(VMF_NNM_MODEL_QUANTIZATION_PARAMETERS_V1_T* quantization_parameters_v1, int quantized_fixed_point_descriptor_idx, int *radix, float *scale)
{
    if ((NULL == quantization_parameters_v1) ||
        (NULL == radix) ||
        (NULL == scale)) {
        printf("error: NULL pointer input parameter\n");
        return KP_ERROR_INVALID_PARAM_12;
    }

    if (quantized_fixed_point_descriptor_idx >= (int)quantization_parameters_v1->quantized_fixed_point_descriptor_num) {
        printf("error: index of quantized_fixed_point_descriptor out of range\n");
        return KP_ERROR_INVALID_PARAM_12;
    }

    VMF_NNM_MODEL_QUANTIZED_FIXED_POINT_DESCRIPTOR_T *quantized_fixed_point_descriptor = &(quantization_parameters_v1->quantized_fixed_point_descriptor[quantized_fixed_point_descriptor_idx]);

    *radix = quantized_fixed_point_descriptor->radix;

    switch (quantized_fixed_point_descriptor->scale_dtype)
    {
    case NGS_DTYPE_INT8:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_int8;
        break;
    case NGS_DTYPE_INT16:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_int16;
        break;
    case NGS_DTYPE_INT32:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_int32;
        break;
    case NGS_DTYPE_UINT8:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_uint8;
        break;
    case NGS_DTYPE_UINT16:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_uint16;
        break;
    case NGS_DTYPE_UINT32:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_uint32;
        break;
    case NGS_DTYPE_FLOAT32:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_float32;
        break;
    default:
        printf("error: get invalide KneronKNE_DataType_enum_t ...\n");
        return KP_ERROR_INVALID_MODEL_21;
    }

    return KP_SUCCESS;
}

static int _kneron_round(float _input)
{
    if (_input > (float)INT_MAX) {
        return INT_MAX;
    } else if (_input < (float)INT_MIN) {
        return INT_MIN;
    }

    int f               = (int)_input;
    float fractionPart  = _input - f;

    if (fractionPart < 0) {
        fractionPart *= -1;
    }

    bool oddFlag = true;

    if (f % 2 == 0) {
        oddFlag = false;
    }

    if (fractionPart > 0.5) {
        // always round to nearest
        if (_input >= 0) {
            ++f;
        } else {
            --f;
        }
    } else if (fractionPart == 0.5) {
        // ties to even depending odd or even
        if (oddFlag) {
            if (_input >= 0) {
                ++f;
            } else {
                --f;
            }
        }
    }

    return f;
}

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
    uint32_t npu_channel_group_stride_tmp               = 0;
    uint32_t npu_channel_group_stride                   = 0;
    uint32_t channel_idx                                = 0;

    if ((NULL == tensor) ||
        (NULL == scalar_index_list)) {
        printf("get scalar int8 fail: NULL pointer paramaters ...\n");
        goto FUNC_OUT;
    }

    if ((DRAM_FMT_16W1C8B == tensor->data_layout) ||
        (DRAM_FMT_1W16C8B == tensor->data_layout) ||
        (DRAM_FMT_1W16C8B_CH_COMPACT == tensor->data_layout) ||
        (DRAM_FMT_RAW8B == tensor->data_layout)) {
        if (NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_1 == tensor->tensor_shape_info.version) {
            printf("unsupported model tensor shape info ...\n");
            goto FUNC_OUT;
        } else if (NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_2 == tensor->tensor_shape_info.version) {
            tensor_shape_info_v2 = &(tensor->tensor_shape_info.tensor_shape_info_data.v2);

            if (tensor_shape_info_v2->shape_len != scalar_index_list_len) {
                printf("get scalar int8 fail: invalid scalar index list length ...\n");
                goto FUNC_OUT;
            }

            for (int axis = 0; axis < (int)tensor_shape_info_v2->shape_len; axis++) {
                if ((int)scalar_index_list[axis] < tensor_shape_info_v2->shape[axis]) {
                    npu_data_buf_offset += scalar_index_list[axis] * tensor_shape_info_v2->stride_npu[axis];
                } else {
                    printf("get scalar int8 fail: invalid scalar index ...\n");
                    goto FUNC_OUT;
                }
            }

            if (DRAM_FMT_1W16C8B == tensor->data_layout) {
                for (int axis = 0; axis < (int)tensor_shape_info_v2->shape_len; axis++) {
                    if (1 == tensor_shape_info_v2->stride_npu[axis]) {
                        channel_idx = axis;
                        continue;
                    }

                    npu_channel_group_stride_tmp = tensor_shape_info_v2->stride_npu[axis] * tensor_shape_info_v2->shape[axis];
                    if (npu_channel_group_stride_tmp > npu_channel_group_stride)
                        npu_channel_group_stride = npu_channel_group_stride_tmp;
                }

                npu_channel_group_stride -= 16;

                /* npu_data_buf_offset += (scalar_index_list[channel_idx] / 16) * npu_channel_group_stride; */
                npu_data_buf_offset += (scalar_index_list[channel_idx] >> 4) * npu_channel_group_stride;
            }
        } else {
            printf("get scalar int8 fail: invalid source tensor shape version ...\n");
            goto FUNC_OUT;
        }
    } else {
        printf("get scalar int8 fail: invalid NPU data layout ...\n");
        goto FUNC_OUT;
    }

    scalar = &(((int8_t *)tensor->base_pointer)[npu_data_buf_offset]);

FUNC_OUT:
    return scalar;
}

/**
 * @brief get the int16 scalar from tensor.
 */
int16_t ex_get_scalar_int16(ngs_tensor_t* tensor, uint32_t* scalar_index_list, uint32_t scalar_index_list_len)
{
    uint16_t scalar_unsigned                            = 0;
    int16_t scalar                                      = 0;
    uint32_t npu_data_buf_offset                        = 0;

    ngs_tensor_shape_info_v2_t *tensor_shape_info_v2    = NULL;
    uint16_t npu_data_high_bit_offset                   = 16;
    uint32_t npu_channel_group_stride_tmp               = 0;
    uint32_t npu_channel_group_stride                   = 0;
    uint32_t channel_idx                                = 0;

    if ((NULL == tensor) ||
        (NULL == scalar_index_list)) {
        printf("get scalar int16 fail: NULL pointer paramters ...\n");
        goto FUNC_OUT;
    }

    if ((DRAM_FMT_1W16C8BHL == tensor->data_layout) ||
        (DRAM_FMT_1W16C8BHL_CH_COMPACT == tensor->data_layout) ||
        (DRAM_FMT_4W4C8BHL == tensor->data_layout) ||
        (DRAM_FMT_16W1C8BHL == tensor->data_layout) ||
        (DRAM_FMT_8W1C16B == tensor->data_layout) ||
        (DRAM_FMT_RAW16B == tensor->data_layout)) {
        if (NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_1 == tensor->tensor_shape_info.version) {
            printf("unsupported model tensor shape info ...\n");
            goto FUNC_OUT;
        } else if (NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_2 == tensor->tensor_shape_info.version) {
            tensor_shape_info_v2 = &(tensor->tensor_shape_info.tensor_shape_info_data.v2);

            if (tensor_shape_info_v2->shape_len != scalar_index_list_len) {
                printf("get scalar int16 fail: invalid scalar index list length ...\n");
                goto FUNC_OUT;
            }

            for (int axis = 0; axis < (int)tensor_shape_info_v2->shape_len; axis++) {
                if ((int)scalar_index_list[axis] < tensor_shape_info_v2->shape[axis]) {
                    npu_data_buf_offset += scalar_index_list[axis] * tensor_shape_info_v2->stride_npu[axis];
                } else {
                    printf("get scalar int16 fail: invalid scalar index ...\n");
                    goto FUNC_OUT;
                }
            }

            if ((DRAM_FMT_8W1C16B == tensor->data_layout) ||
                (DRAM_FMT_RAW16B == tensor->data_layout)) {
                scalar_unsigned = ((((uint16_t *)tensor->base_pointer)[npu_data_buf_offset]) & 0xfffeu);
                scalar          = *(int16_t *)(&scalar_unsigned);
            } else if ((DRAM_FMT_1W16C8BHL == tensor->data_layout) ||
                       (DRAM_FMT_1W16C8BHL_CH_COMPACT == tensor->data_layout) ||
                       (DRAM_FMT_4W4C8BHL == tensor->data_layout) ||
                       (DRAM_FMT_16W1C8BHL == tensor->data_layout)) {
                if (DRAM_FMT_1W16C8BHL == tensor->data_layout) {
                    for (int axis = 0; axis < (int)tensor_shape_info_v2->shape_len; axis++) {
                        if (1 == tensor_shape_info_v2->stride_npu[axis]) {
                            channel_idx = axis;
                            continue;
                        }

                        npu_channel_group_stride_tmp = tensor_shape_info_v2->stride_npu[axis] * tensor_shape_info_v2->shape[axis];
                        if (npu_channel_group_stride_tmp > npu_channel_group_stride)
                            npu_channel_group_stride = npu_channel_group_stride_tmp;
                    }

                    npu_channel_group_stride -= 16;

                    /* npu_data_buf_offset += (scalar_index_list[channel_idx] / 16) * npu_channel_group_stride; */
                    npu_data_buf_offset += (scalar_index_list[channel_idx] >> 4) * npu_channel_group_stride;
                }

                /* npu_data_buf_offset = (npu_data_buf_offset / 16) * 32 + (npu_data_buf_offset % 16) */
                npu_data_buf_offset = ((npu_data_buf_offset >> 4) << 5) + (npu_data_buf_offset & 15u);

                scalar              = (int16_t)(((((uint16_t)(((uint8_t *)tensor->base_pointer)[npu_data_buf_offset])) & 0x007fu) +
                                                 (((uint16_t)(((uint8_t *)tensor->base_pointer)[npu_data_buf_offset + npu_data_high_bit_offset])) << 7)) << 1);
            }
        } else {
            printf("get scalar int16 fail: invalid source tensor shape version ...\n");
            goto FUNC_OUT;
        }
    } else {
        printf("get scalar int16 fail: invalid NPU data layout ...\n");
        goto FUNC_OUT;
    }

FUNC_OUT:
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

/**
 * @brief Convert ONNX data to NPU data with VMF_NNM_MODEL_TENSOR_DESCRIPTOR_T information
 */
int ex_convert_onnx_data_to_npu_data(VMF_NNM_MODEL_TENSOR_DESCRIPTOR_T *tensor_descriptor, float *onnx_data_buf, int32_t onnx_data_element_num, int8_t **npu_data_buf, int32_t *npu_data_buf_size)
{
    int status                                                              = KP_SUCCESS;

    VMF_NNM_MODEL_TENSOR_INFO_V2_T *tensor_shape_info                       = NULL;

    uint32_t npu_data_layout                                                = DRAM_FMT_UNKNOWN;

    VMF_NNM_MODEL_QUANTIZATION_PARAMETERS_V1_T *quantization_parameters_v1  = NULL;
    float quantization_max_value                                            = 0;
    float quantization_min_value                                            = 0;
    int32_t radix                                                           = 0;
    float scale                                                             = 0;
    float quantization_factor                                               = 0;
    int quantized_axis_stride                                               = 0;

    int channel_idx                                                         = 0;
    int npu_channel_group_stride_tmp                                        = 0;
    int npu_channel_group_stride                                            = -1;

    int32_t *onnx_data_shape_index                                          = NULL;
    int32_t onnx_data_buf_offset                                            = 0;
    int32_t npu_data_buf_offset                                             = 0;

    int max_npu_stride_axis                                                 = 0;
    uint32_t max_npu_stride                                                 = 0;
    uint16_t npu_data_element_u16b                                          = 0;
    uint16_t npu_data_element_i16b                                          = 0;
    uint16_t npu_data_high_bit_offset                                       = 16;

    if (NULL == tensor_descriptor ||
        NULL == onnx_data_buf ||
        0 == onnx_data_element_num ||
        NULL == npu_data_buf ||
        NULL == npu_data_buf_size) {
        printf("convert onnx data to npu data fail: NULL pointer input parameters ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    if (KP_MODEL_TENSOR_SHAPE_INFO_VERSION_2 != tensor_descriptor->tensor_shape_info.version) {
        printf("convert onnx data to npu data fail: only support KP_MODEL_TENSOR_SHAPE_INFO_VERSION_2 tensor shape ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    quantization_parameters_v1  = &(tensor_descriptor->quantization_parameters.quantization_parameters_data.v1);
    tensor_shape_info           = &(tensor_descriptor->tensor_shape_info.tensor_shape_info_data.v2);
    npu_data_layout             = tensor_descriptor->data_layout;

    /* input data quantization */
    {
        // get clip value by data layout
        switch (npu_data_layout)
        {
        case DRAM_FMT_4W4C8B:
        case DRAM_FMT_1W16C8B:
        case DRAM_FMT_1W16C8B_CH_COMPACT:
        case DRAM_FMT_16W1C8B:
        case DRAM_FMT_RAW8B:
        case DRAM_FMT_HW4C8B_KEEP_A:
        case DRAM_FMT_HW4C8B_DROP_A:
        case DRAM_FMT_HW1C8B:
            quantization_max_value = 127.0f;
            quantization_min_value = -128.0f;
            break;
        case DRAM_FMT_8W1C16B:
        case DRAM_FMT_RAW16B:
        case DRAM_FMT_4W4C8BHL:
        case DRAM_FMT_1W16C8BHL:
        case DRAM_FMT_1W16C8BHL_CH_COMPACT:
        case DRAM_FMT_16W1C8BHL:
        case DRAM_FMT_HW1C16B_LE:
        case DRAM_FMT_HW1C16B_BE:
            quantization_max_value = 32766.0f;
            quantization_min_value = -32768.0f;
            break;
        case DRAM_FMT_RAW_FLOAT:
            quantization_max_value = FLT_MAX;
            quantization_min_value = FLT_MIN;
            break;
        default:
            printf("error: get invalide data layout ...\n");
            status = KP_ERROR_INVALID_MODEL_21;
            goto FUNC_OUT;
        }

        // toolchain calculate the radix value from input data (after normalization), and set it into NEF model.
        // NPU will divide input data "2^radix" automatically, so, we have to scaling the input data here due to this reason.
        if (1 == quantization_parameters_v1->quantized_fixed_point_descriptor_num) {
            // layer-wise quantize
            radix = 0;
            scale = 0;
            if (KP_SUCCESS != _get_quantization_parameters_v1_information(quantization_parameters_v1, 0, &radix, &scale)) {
                printf("error: get invalide KneronKNE_DataType_enum_t ...\n");
                status = KP_ERROR_INVALID_MODEL_21;
                goto FUNC_OUT;
            }

            quantization_factor = ex_pow2(radix) * scale;

            for (onnx_data_buf_offset = 0; onnx_data_buf_offset < onnx_data_element_num; onnx_data_buf_offset++) {
                onnx_data_buf[onnx_data_buf_offset] = (float)_kneron_round(onnx_data_buf[onnx_data_buf_offset] * quantization_factor);
                onnx_data_buf[onnx_data_buf_offset] = MAX(quantization_min_value, MIN(onnx_data_buf[onnx_data_buf_offset], quantization_max_value));
            }
        } else if (1 < quantization_parameters_v1->quantized_fixed_point_descriptor_num) {
            // channel-wise quantize
            quantized_axis_stride = 1;
            for (uint32_t axis = 0; axis < tensor_descriptor->tensor_shape_info.tensor_shape_info_data.v2.shape_len; axis++) {
                if (axis != quantization_parameters_v1->quantized_axis) {
                    quantized_axis_stride *= tensor_descriptor->tensor_shape_info.tensor_shape_info_data.v2.shape[axis];
                }
            }

            onnx_data_buf_offset = 0;
            for (uint32_t quantized_fixed_point_descriptor_idx = 0; quantized_fixed_point_descriptor_idx < quantization_parameters_v1->quantized_fixed_point_descriptor_num; quantized_fixed_point_descriptor_idx++) {
                radix = 0;
                scale = 0;

                if (KP_SUCCESS != _get_quantization_parameters_v1_information(quantization_parameters_v1, quantized_fixed_point_descriptor_idx, &radix, &scale)) {
                    printf("error: get invalide KneronKNE_DataType_enum_t ...\n");
                    status = KP_ERROR_INVALID_MODEL_21;
                    goto FUNC_OUT;
                }

                quantization_factor = ex_pow2(radix) * scale;

                for (int quantized_axis_offset = 0; quantized_axis_offset < quantized_axis_stride; quantized_axis_offset++) {
                    onnx_data_buf[onnx_data_buf_offset] = (float)_kneron_round(onnx_data_buf[onnx_data_buf_offset] * quantization_factor);
                    onnx_data_buf[onnx_data_buf_offset] = MAX(quantization_min_value, MIN(onnx_data_buf[onnx_data_buf_offset], quantization_max_value));
                    onnx_data_buf_offset += 1;
                }
            }
        } else {
            printf("error: get invalide number of quantized_fixed_point_descriptor ...\n");
            status = KP_ERROR_INVALID_MODEL_21;
            goto FUNC_OUT;
        }
    }

    /* re-layout the data to fit NPU data layout format */
    {
        onnx_data_shape_index = calloc(tensor_shape_info->shape_len, sizeof(int32_t));
        if (NULL == onnx_data_shape_index) {
            printf("error: malloc working buffer onnx_data_shape_index fail ...\n");
            status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
            goto FUNC_OUT;
        }

        switch (npu_data_layout)
        {
        case DRAM_FMT_1W16C8B:
        case DRAM_FMT_1W16C8BHL:
            for (int axis = 0; axis < (int)tensor_shape_info->shape_len; axis++) {
                if (1 == tensor_shape_info->stride_npu[axis]) {
                    channel_idx = axis;
                    continue;
                }

                npu_channel_group_stride_tmp = tensor_shape_info->stride_npu[axis] * tensor_shape_info->shape[axis];
                if (npu_channel_group_stride_tmp > npu_channel_group_stride)
                    npu_channel_group_stride = npu_channel_group_stride_tmp;
            }

            *npu_data_buf_size = (tensor_shape_info->shape[channel_idx] >> 4) * npu_channel_group_stride;

            if (0 != (tensor_shape_info->shape[channel_idx] % 16)) {
                *npu_data_buf_size += npu_channel_group_stride;
            }

            /* reuse for npu_data_buf_offset calculate (without reset the ) */
            npu_channel_group_stride -= 16;
            break;
        default:
            for (int axis = 0; axis < (int)tensor_shape_info->shape_len; axis++) {
                if (tensor_shape_info->stride_npu[axis] > max_npu_stride) {
                    max_npu_stride      = tensor_shape_info->stride_npu[axis];
                    max_npu_stride_axis = axis;
                }
            }

            *npu_data_buf_size = tensor_shape_info->shape[max_npu_stride_axis] * max_npu_stride;
            break;
        }

        switch (npu_data_layout)
        {
        case DRAM_FMT_4W4C8B:
        case DRAM_FMT_1W16C8B:
        case DRAM_FMT_1W16C8B_CH_COMPACT:
        case DRAM_FMT_16W1C8B:
        case DRAM_FMT_RAW8B:
        case DRAM_FMT_HW4C8B_KEEP_A:
        case DRAM_FMT_HW4C8B_DROP_A:
        case DRAM_FMT_HW1C8B:
            if (NULL == *npu_data_buf) {
                printf("error: malloc working buffer npu_data_buf fail ...\n");
                status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
                goto FUNC_OUT;
            }

            memset(*npu_data_buf, 0, *npu_data_buf_size);

            while (true) {
                onnx_data_buf_offset    = 0;
                npu_data_buf_offset     = 0;

                for (uint32_t axis = 0; axis < tensor_shape_info->shape_len; axis++) {
                    onnx_data_buf_offset    += onnx_data_shape_index[axis] * tensor_shape_info->stride_onnx[axis];
                    npu_data_buf_offset     += onnx_data_shape_index[axis] * tensor_shape_info->stride_npu[axis];
                }

                if (DRAM_FMT_1W16C8B == npu_data_layout) {
                    /* npu_data_buf_offset += (onnx_data_shape_index[channel_idx] / 16) * npu_channel_group_stride; */
                    npu_data_buf_offset += (onnx_data_shape_index[channel_idx] >> 4) * npu_channel_group_stride;
                }

                ((int8_t *)*npu_data_buf)[npu_data_buf_offset] = (int8_t)onnx_data_buf[onnx_data_buf_offset];

                for (int32_t axis = (int32_t)tensor_shape_info->shape_len - 1; axis >= 0; axis--) {
                    onnx_data_shape_index[axis]++;
                    if (onnx_data_shape_index[axis] == tensor_shape_info->shape[axis]) {
                        if (axis == 0)
                            break;

                        onnx_data_shape_index[axis] = 0;
                        continue;
                    } else {
                        break;
                    }
                }

                if (onnx_data_shape_index[0] == tensor_shape_info->shape[0]) {
                    break;
                }
            }
            break;
        case DRAM_FMT_8W1C16B:
        case DRAM_FMT_RAW16B:
        case DRAM_FMT_HW1C16B_LE:
            if (NULL == *npu_data_buf) {
                printf("error: malloc working buffer npu_data_buf fail ...\n");
                status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
                goto FUNC_OUT;
            }

            memset(*npu_data_buf, 0, *npu_data_buf_size);

            while (true) {
                onnx_data_buf_offset    = 0;
                npu_data_buf_offset     = 0;

                for (uint32_t axis = 0; axis < tensor_shape_info->shape_len; axis++) {
                    onnx_data_buf_offset    += onnx_data_shape_index[axis] * tensor_shape_info->stride_onnx[axis];
                    npu_data_buf_offset     += onnx_data_shape_index[axis] * tensor_shape_info->stride_npu[axis];
                }

                npu_data_element_i16b                               = (int16_t)onnx_data_buf[onnx_data_buf_offset];
                npu_data_element_u16b                               = *((uint16_t *)(&npu_data_element_i16b));
                ((uint16_t *)*npu_data_buf)[npu_data_buf_offset]    = (npu_data_element_u16b & 0xfffeu);

                for (int32_t axis = (int32_t)tensor_shape_info->shape_len - 1; axis >= 0; axis--) {
                    onnx_data_shape_index[axis]++;
                    if (onnx_data_shape_index[axis] == tensor_shape_info->shape[axis]) {
                        if (axis == 0)
                            break;

                        onnx_data_shape_index[axis] = 0;
                        continue;
                    } else {
                        break;
                    }
                }

                if (onnx_data_shape_index[0] == tensor_shape_info->shape[0])
                    break;
            }
            break;
        case DRAM_FMT_HW1C16B_BE:
            if (NULL == *npu_data_buf) {
                printf("error: malloc working buffer npu_data_buf fail ...\n");
                status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
                goto FUNC_OUT;
            }

            memset(*npu_data_buf, 0, *npu_data_buf_size);

            while (true) {
                onnx_data_buf_offset    = 0;
                npu_data_buf_offset     = 0;

                for (uint32_t axis = 0; axis < tensor_shape_info->shape_len; axis++) {
                    onnx_data_buf_offset    += onnx_data_shape_index[axis] * tensor_shape_info->stride_onnx[axis];
                    npu_data_buf_offset     += onnx_data_shape_index[axis] * tensor_shape_info->stride_npu[axis];
                }

                npu_data_element_i16b                               = (int16_t)onnx_data_buf[onnx_data_buf_offset];
                npu_data_element_u16b                               = *((uint16_t *)(&npu_data_element_i16b));
                npu_data_element_u16b                               = (npu_data_element_u16b & 0xfffeu);
                ((uint16_t *)*npu_data_buf)[npu_data_buf_offset]    = ((npu_data_element_u16b >> 8) | (npu_data_element_u16b << 8));

                for (int32_t axis = (int32_t)tensor_shape_info->shape_len - 1; axis >= 0; axis--) {
                    onnx_data_shape_index[axis]++;
                    if (onnx_data_shape_index[axis] == tensor_shape_info->shape[axis]) {
                        if (axis == 0)
                            break;

                        onnx_data_shape_index[axis] = 0;
                        continue;
                    } else {
                        break;
                    }
                }

                if (onnx_data_shape_index[0] == tensor_shape_info->shape[0])
                    break;
            }
            break;
        case DRAM_FMT_4W4C8BHL:
        case DRAM_FMT_1W16C8BHL:
        case DRAM_FMT_1W16C8BHL_CH_COMPACT:
        case DRAM_FMT_16W1C8BHL:
            if (NULL == *npu_data_buf) {
                printf("error: malloc working buffer npu_data_buf fail ...\n");
                status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
                goto FUNC_OUT;
            }

            memset(*npu_data_buf, 0, *npu_data_buf_size);

            while (true) {
                onnx_data_buf_offset    = 0;
                npu_data_buf_offset     = 0;

                for (uint32_t axis = 0; axis < tensor_shape_info->shape_len; axis++) {
                    onnx_data_buf_offset    += onnx_data_shape_index[axis] * tensor_shape_info->stride_onnx[axis];
                    npu_data_buf_offset     += onnx_data_shape_index[axis] * tensor_shape_info->stride_npu[axis];
                }

                if (DRAM_FMT_1W16C8BHL == npu_data_layout) {
                    /* npu_data_buf_offset += (onnx_data_shape_index[channel_idx] / 16) * npu_channel_group_stride; */
                    npu_data_buf_offset += (onnx_data_shape_index[channel_idx] >> 4) * npu_channel_group_stride;
                }

                /* npu_data_buf_offset = (npu_data_buf_offset / 16) * 32 + (npu_data_buf_offset % 16) */
                npu_data_buf_offset = ((npu_data_buf_offset >> 4) << 5) + (npu_data_buf_offset & 15u);

                npu_data_element_i16b                                                       = (int16_t)onnx_data_buf[onnx_data_buf_offset];
                npu_data_element_u16b                                                       = ((*((uint16_t *)(&npu_data_element_i16b))) >> 1);
                ((uint8_t *)*npu_data_buf)[npu_data_buf_offset]                             = (uint8_t)(npu_data_element_u16b & 0x007fu);
                ((uint8_t *)*npu_data_buf)[npu_data_buf_offset + npu_data_high_bit_offset]  = (uint8_t)((npu_data_element_u16b >> 7) & 0x00ffu);

                for (int32_t axis = (int32_t)tensor_shape_info->shape_len - 1; axis >= 0; axis--) {
                    onnx_data_shape_index[axis]++;
                    if (onnx_data_shape_index[axis] == tensor_shape_info->shape[axis]) {
                        if (axis == 0)
                            break;

                        onnx_data_shape_index[axis] = 0;
                        continue;
                    } else {
                        break;
                    }
                }

                if (onnx_data_shape_index[0] == tensor_shape_info->shape[0])
                    break;
            }
            break;
        case DRAM_FMT_RAW_FLOAT:
            if (NULL == *npu_data_buf) {
                printf("error: malloc working buffer npu_data_buf fail ...\n");
                status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
                goto FUNC_OUT;
            }

            while (true) {
                onnx_data_buf_offset    = 0;
                npu_data_buf_offset     = 0;

                for (uint32_t axis = 0; axis < tensor_shape_info->shape_len; axis++) {
                    onnx_data_buf_offset    += onnx_data_shape_index[axis] * tensor_shape_info->stride_onnx[axis];
                    npu_data_buf_offset     += onnx_data_shape_index[axis] * tensor_shape_info->stride_npu[axis];
                }

                ((float *)*npu_data_buf)[npu_data_buf_offset] = (float)onnx_data_buf[onnx_data_buf_offset];

                for (int32_t axis = (int32_t)tensor_shape_info->shape_len - 1; axis >= 0; axis--) {
                    onnx_data_shape_index[axis]++;
                    if (onnx_data_shape_index[axis] == tensor_shape_info->shape[axis]) {
                        if (axis == 0)
                            break;

                        onnx_data_shape_index[axis] = 0;
                        continue;
                    } else {
                        break;
                    }
                }

                if (onnx_data_shape_index[0] == tensor_shape_info->shape[0]) {
                    break;
                }
            }
            break;
        }
    }

FUNC_OUT:
    if (NULL != onnx_data_shape_index)
        free(onnx_data_shape_index);

    return status;
}
