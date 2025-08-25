#ifndef DEMO_CUSTOMIZE_INF_SINGLE_MODEL_WITH_SW_NPU_FORMAT_CONVERT_H
#define DEMO_CUSTOMIZE_INF_SINGLE_MODEL_WITH_SW_NPU_FORMAT_CONVERT_H

#define DEMO_KL730_CUSTOMIZE_INF_SINGLE_MODEL_WITH_SW_NPU_FORMAT_CONVERT_JOB_ID 4002
#define YOLO_BOX_MAX                                                            100     /**< maximum number of bounding boxes for Yolo models */

#include "kp_struct.h"

/**
 * @brief In this customized inference example, we will focus 
 * on converting source data into NPU format. Some specific model 
 * inputs in NPU format that cannot utilize the Kneron-supported 
 * hardware conversion will use this example to learn how to perform 
 * the conversion manually. For more information about the NPU format 
 * and the Inproc support list, please refer to the KneronPLUS documentation.
 */

/**
 * @brief describe a yolo output result after post-processing
 */
typedef struct
{
    uint32_t class_count;                  /**< total class count */
    uint32_t box_count;                    /**< boxes of all classes */
    kp_bounding_box_t boxes[YOLO_BOX_MAX]; /**< box information */
} __attribute__((aligned(4))) kp_custom_with_sw_npu_format_convert_yolo_result_t;

typedef struct
{
    /* header stamp is necessary for data transfer between host and device */
    kp_inference_header_stamp_t header_stamp;
    uint32_t width;
    uint32_t height;
} __attribute__((aligned(4))) demo_customize_inf_single_model_with_sw_npu_format_convert_header_t;

// result (header + data) for 'Kneron APP Yolo Inference'
typedef struct
{
    /* header stamp is necessary for data transfer between host and device */
    kp_inference_header_stamp_t header_stamp;
    kp_custom_with_sw_npu_format_convert_yolo_result_t yolo_result;
} __attribute__((aligned(4))) demo_customize_inf_single_model_with_sw_npu_format_convert_result_t;

void demo_customize_inf_single_model_with_sw_npu_format_convert(int job_id, int num_input_buf, void **inf_input_buf_list);
void demo_customize_inf_single_model_with_sw_npu_format_convert_deinit();

#endif // DEMO_CUSTOMIZE_INF_SINGLE_MODEL_WITH_SW_NPU_FORMAT_CONVERT_H
