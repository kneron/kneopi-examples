#ifndef KDP2_INF_APP_YOLO_H
#define KDP2_INF_APP_YOLO_H

#include <stdint.h>
#include "kp_struct.h"

#define KDP2_INF_ID_APP_YOLO                    11
#define KDP2_JOB_ID_APP_YOLO_CONFIG_POST_PROC   100 // handle set or get

typedef struct
{
    uint32_t model_id;              // specify model id
    kp_normalize_mode_t model_norm; // specify model normalization
} __attribute__((aligned(4))) kp_app_yolo_config_t;

/**
 * @brief describe a yolo post-process configurations for yolo series
 */
typedef struct
{
    float prob_thresh;
    float nms_thresh;
    uint32_t max_detection_per_class;
    uint16_t anchor_row;
    uint16_t anchor_col;
    uint16_t stride_size;
    uint16_t reserved_size;
    uint32_t data[40];
} __attribute__((aligned(4))) kp_app_yolo_post_proc_config_t;

// align to YOLO_V5_MAX_BOX_NUM in post_utils.h
#define YOLO_GOOD_BOX_MAX_SCPU 500 /**< maximum number of bounding boxes for Yolo models */

/**
 * @brief describe a yolo output result after post-processing
 * The object name corresponding to class_num in each boxex are listed in the document in Kneron Document Center
 * Please visit https://doc.kneron.com/ -> Kneron PLUS - C -> Appendix -> Yolo Object Name Mapping
 */
typedef struct
{
    uint32_t class_count;                       /**< total class count */
    uint32_t box_count;                         /**< boxes of all classes */
    kp_bounding_box_t boxes[YOLO_GOOD_BOX_MAX_SCPU]; /**< box information */
} __attribute__((aligned(4))) kp_app_yolo_result_t;

/********** KDP2_INF_ID_APP_YOLO **********/

// post-proc config data struct shared for setting or getting
typedef struct
{
    /* header stamp is necessary for data transfer between host and device */
    kp_inference_header_stamp_t header_stamp;
    uint32_t set_or_get; // get = 0, set = 1
    uint32_t model_id;
    uint32_t param_size;
    uint8_t param_data[200]; // contains kp_app_yolo_post_proc_config_*** body

} __attribute__((aligned(4))) kdp2_ipc_app_yolo_post_proc_config_t;

// input header for 'Kneron APP Yolo Inference'
typedef struct
{
    /* header stamp is necessary for data transfer between host and device */
    kp_inference_header_stamp_t header_stamp;

    uint32_t inf_number;
    uint32_t width;
    uint32_t height;
    uint32_t channel;
    uint32_t model_id;
    uint32_t image_format;    // kp_image_format_t
    uint32_t model_normalize; // kp_normalize_mode_t

} __attribute__((aligned(4))) kdp2_ipc_app_yolo_inf_header_t;

// result (header + data) for 'Kneron APP Yolo Inference'
typedef struct
{
    /* header stamp is necessary for data transfer between host and device */
    kp_inference_header_stamp_t header_stamp;
    uint32_t inf_number;
    kp_app_yolo_result_t yolo_data;

} __attribute__((aligned(4))) kdp2_ipc_app_yolo_result_t;

void kdp2_app_yolo_config_post_process_parameters(int job_id, int num_input_buf, void **inf_input_buf_list);
void kdp2_app_yolo_inference(int job_id, int num_input_buf, void **inf_input_buf_list);
void kdp2_app_yolo_inference_deinit();

#endif
