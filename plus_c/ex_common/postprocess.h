/**
 * @file        postprocess.h
 * @brief       Kneron PLUS post process APIs
 *
 * Post process functions for application examples are provided
 *
 * @version     0.1
 * @date        2021-03-22
 *
 * @copyright   Copyright (c) 2021 Kneron Inc. All rights reserved.
 */

#pragma once

#include <stdint.h>
#include "kp_struct.h"

/**
 * @brief YOLO V3 post-processing function for KL520.
 *
 * @param[in] node_output floating-point output node arrays, it should come from kp_generic_inference_retrieve_node().
 * @param[in] num_output_node total number of output node.
 * @param[in] pre_proc_info hardware pre-process related info.
 * @param[in] thresh_value range from 0 ~ 1
 * @param[out] yoloResult this is the yolo result output, users need to prepare a buffer of 'kp_yolo_result_t' for this.
 *
 * @return return 0 means sucessful, otherwise failed.
 */
int post_process_yolo_v3(kp_inf_float_node_output_t *node_output[], int num_output_node,
                         kp_hw_pre_proc_info_t *pre_proc_info, float thresh_value, kp_yolo_result_t *yoloResult);

/**
 * @brief YOLO V5 post-processing function (with sigmoid) for KL520.
 *
 * @param[in] node_output floating-point output node arrays, it should come from kp_generic_inference_retrieve_node().
 * @param[in] num_output_node total number of output node.
 * @param[in] pre_proc_info hardware pre-process related info.
 * @param[in] thresh_value range from 0 ~ 1
 * @param[out] yoloResult this is the yolo result output, users need to prepare a buffer of 'kp_yolo_result_t' for this.
 *
 * @return return 0 means sucessful, otherwise failed.
 */
int post_process_yolo_v5_520(kp_inf_float_node_output_t *node_output[], int num_output_node,
                             kp_hw_pre_proc_info_t *pre_proc_info, float thresh_value, kp_yolo_result_t *yoloResult);

/**
 * @brief YOLO V5 post-processing function (without sigmoid) for KL720.
 *
 * @param[in] node_output floating-point output node arrays, it should come from kp_generic_inference_retrieve_node().
 * @param[in] num_output_node total number of output node.
 * @param[in] pre_proc_info hardware pre-process related info.
 * @param[in] thresh_value range from 0 ~ 1
 * @param[out] yoloResult this is the yolo result output, users need to prepare a buffer of 'kp_yolo_result_t' for this.
 *
 * @return return 0 means sucessful, otherwise failed.
 */
int post_process_yolo_v5_720(kp_inf_float_node_output_t *node_output[], int num_output_node,
                             kp_hw_pre_proc_info_t *pre_proc_info, float thresh_value, kp_yolo_result_t *yoloResult);
