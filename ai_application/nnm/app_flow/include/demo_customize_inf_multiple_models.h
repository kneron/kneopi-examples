#ifndef DEMO_CUSTOMIZE_INF_MULTIPLE_MODELS_H
#define DEMO_CUSTOMIZE_INF_MULTIPLE_MODELS_H

#include "kp_struct.h"

#define DEMO_KL730_CUSTOMIZE_INF_MULTIPLE_MODEL_JOB_ID  4001
#define PD_BOX_MAX                                      80

/**
 * @brief describe a pedestrian detect classification result of one detected person
 */
typedef struct
{
    float pd_class_score;   /**< a pedestrian classification score */
    kp_bounding_box_t pd;   /**< a pedestrian box information */
} __attribute__((aligned(4))) one_pd_classification_result_t;

/**
 * @brief describe a pedestrian detect classification output result
 */
typedef struct
{
    uint32_t box_count;                                     /**< boxes of all classes */
    one_pd_classification_result_t pds[PD_BOX_MAX];     /**< pedestrian detect information */
} __attribute__((aligned(4))) pd_classification_result_t;

typedef struct
{
    /* header stamp is necessary for data transfer between host and device */
    kp_inference_header_stamp_t header_stamp;
    uint32_t width;
    uint32_t height;
} __attribute__((aligned(4))) demo_customize_inf_multiple_models_header_t;

// result (header + data) for 'Customize Inference Multiple Models'
typedef struct
{
    /* header stamp is necessary for data transfer between host and device */
    kp_inference_header_stamp_t header_stamp;
    pd_classification_result_t pd_classification_result;
} __attribute__((aligned(4))) demo_customize_inf_multiple_models_result_t;

void demo_customize_inf_multiple_model(int job_id, int num_input_buf, void **inf_input_buf_list);
void demo_customize_inf_multiple_model_deinit();

#endif // DEMO_CUSTOMIZE_INF_MULTIPLE_MODELS_H
