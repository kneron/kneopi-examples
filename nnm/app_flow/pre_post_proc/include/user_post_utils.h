/**
 * Utility function headers for the postprocess functions.
 *
 * Copyright (C) 2021 Kneron, Inc. All rights reserved.
 *
 */
#ifndef USER_POST_UTILS_H
#define USER_POST_UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

/******************************************************************
 * enum defined
*******************************************************************/
/**
 * @brief enum for NMS mode
 */
typedef enum
{
    EX_NMS_MODE_ALL_CLASS       = 0,
    EX_NMS_MODE_GROUP_CLASS     = 1,
    EX_NMS_MODE_SINGLE_CLASS    = 2,
    EX_NMS_MODE_END
} ex_nms_mode_t;

/******************************************************************
 * struct defined
*******************************************************************/
/**
 * @brief describe a bounding box information
 */
struct ex_bounding_box_s {
    float x1;           /**< top-left x corner */
    float y1;           /**< top-left y corner */
    float x2;           /**< bottom-right x corner */
    float y2;           /**< bottom-right y corner */
    float score;        /**< probability score */
    int32_t class_num;  /**< class number (of many) with highest probability */
};

/******************************************************************
 * public function
*******************************************************************/
/**
 * @brief optimized dequantization function.
 */
float ex_do_div_scale_optim(float v, float scale);

/**
 * @brief get the output node information
 */
int ex_get_output_tensor(struct kdp_image_s *image_p, int tensor_index, ngs_tensor_t **output_tensor);

/** 
 * @brief get the data pointer corresponding to given channel, row, and column indices
 */
int8_t *ex_get_scalar_int8(ngs_tensor_t* tensor, uint32_t* scalar_index_list, uint32_t scalar_index_list_len);

/** 
 * @brief get the tensor shape.
 */
int *ex_get_tensor_shape(ngs_tensor_t *tensor);

/** 
 * @brief get the tensor quantization parameters - version 1.
 */
ngs_quantization_parameters_v1_t *ex_get_tensor_quantization_parameters_v1(ngs_tensor_t *tensor);

/**
 * @brief power of two function.
 */
float ex_pow2(int exp);

/**
 * @brief calculate the IoU of two bounding box.
 */
float ex_box_iou(struct ex_bounding_box_s *a, struct ex_bounding_box_s *b);

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
                float nms_mode);

/**
 * @brief update candidate bbox list, reserve top max_candidate_num candidate bbox.
 */
int ex_update_candidate_bbox_list(struct ex_bounding_box_s *new_candidate_bbox,
                                  int max_candidate_num,
                                  struct ex_bounding_box_s *candidate_bbox_list,
                                  int *candidate_bbox_num,
                                  int *max_candidate_idx,
                                  int *min_candidate_idx);
/**
 * @brief remap one bounding box to original image coordinates.
 */
void ex_remap_bbox(struct kdp_image_s *image_p, struct ex_bounding_box_s *box, int need_scale);

#endif
