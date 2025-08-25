/*
 * Kneron Application general functions
 *
 * Copyright (C) 2022 Kneron, Inc. All rights reserved.
 *
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "model_type.h"
#include "vmf_nnm_inference_app.h"
#include "vmf_nnm_fifoq_manager.h"

#include "demo_customize_inf_multiple_models.h"
#include "user_post_process_classifier.h"
#include "user_post_process_yolov5.h"

static bool is_init                                         = false;
static struct ex_object_detection_result_s *yolo_pd_result  = NULL;
static struct ex_classifier_top_n_result_s *imagenet_result = NULL;

/**
 * @brief describe class labels of pedestrian detection results.
 */
typedef enum {
    KP_APP_PD_CLASS_PERSON      = 0,
    KP_APP_PD_CLASS_BICYCLE     = 1,
    KP_APP_PD_CLASS_CAR         = 2,
    KP_APP_PD_CLASS_MOTORCYCLE  = 3,
    KP_APP_PD_CLASS_BUS         = 4,
    KP_APP_PD_CLASS_TRUCK       = 5,
    KP_APP_PD_CLASS_CAT         = 6,
    KP_APP_PD_CLASS_DOG         = 7
} kp_app_pd_class_t;

static ex_yolo_post_proc_config_t post_proc_params_v5s_480_256_3 = {
    .prob_thresh                = 0.3,
    .nms_thresh                 = 0.65,
    .max_detection              = 20,
    .max_detection_per_class    = PD_BOX_MAX,
    .nms_mode                   = EX_NMS_MODE_SINGLE_CLASS,
    .anchor_layer_num           = 3,
    .anchor_cell_num_per_layer  = 3,
    .data                       = {{{10, 13}, {16, 30}, {33, 23}},
                                   {{30, 61}, {62, 45}, {59, 119}},
                                   {{116, 90}, {156, 198}, {373, 326}},
                                   {{0, 0}, {0, 0}, {0, 0}},
                                   {{0, 0}, {0, 0}, {0, 0}}},
};

static bool init_temp_buffer()
{
    if (false == is_init) {
        /* allocate DDR memory for ncpu/npu output result */
        yolo_pd_result = (struct ex_object_detection_result_s *)malloc(sizeof(struct ex_object_detection_result_s));
        if (NULL == yolo_pd_result)
            return false;

        imagenet_result = (struct ex_classifier_top_n_result_s *)malloc(sizeof(struct ex_classifier_top_n_result_s));
        if (NULL == imagenet_result)
            return false;

        is_init = true;
    }

    return true;
}

static bool deinit_temp_buffer()
{
    if(is_init == true)
    {
        // free DDR memory for ncpu/npu output restult
        if(NULL != yolo_pd_result)
            free(yolo_pd_result);

        if(NULL != imagenet_result)
            free(imagenet_result);

        is_init = false;
    }


    return true;
}

static int inference_pedestrian_detection(demo_customize_inf_multiple_models_header_t *_input_header,
                                          struct ex_object_detection_result_s *_pd_result /* output */)
{
    // config image preprocessing and model settings
    VMF_NNM_INFERENCE_APP_CONFIG_T inf_config;
    memset(&inf_config, 0, sizeof(VMF_NNM_INFERENCE_APP_CONFIG_T)); // for safety let default 'bool' to 'false'

    // image buffer address should be just after the header
    inf_config.num_image                    = 1;
    inf_config.image_list[0].image_buf      = (void *)((uintptr_t)_input_header + sizeof(demo_customize_inf_multiple_models_header_t));
    inf_config.image_list[0].image_width    = _input_header->width;
    inf_config.image_list[0].image_height   = _input_header->height;
    inf_config.image_list[0].image_channel  = 3;                                                                    // assume RGB565
    inf_config.image_list[0].image_format   = KP_IMAGE_FORMAT_RGB565;                                               // assume RGB565
    inf_config.image_list[0].image_norm     = KP_NORMALIZE_KNERON;                                                  // this depends on model
    inf_config.image_list[0].image_resize   = KP_RESIZE_ENABLE;                                                     // enable resize
    inf_config.image_list[0].image_padding  = KP_PADDING_CORNER;                                                    // enable padding on corner
    inf_config.model_id                     = KNERON_YOLOV5S_PersonBicycleCarMotorcycleBusTruckCatDog8_256_480_3;   // this depends on model

    // setting pre/post-proc configuration
    inf_config.pre_proc_config              = NULL;
    inf_config.post_proc_config             = (void *)&post_proc_params_v5s_480_256_3;                              // yolo post-process configurations for yolo v5 series
    inf_config.post_proc_func               = user_post_yolov5_no_sigmoid;

    // set up pd result output buffer for ncpu/npu
    inf_config.ncpu_result_buf              = (void *)_pd_result;

    return VMF_NNM_Inference_App_Execute(&inf_config);
}

static int inference_pedestrian_classification(demo_customize_inf_multiple_models_header_t *_input_header,
                                               struct ex_bounding_box_s *_box,
                                               struct ex_classifier_top_n_result_s * _imagenet_result/* output */)
{
    // config image preprocessing and model settings
    VMF_NNM_INFERENCE_APP_CONFIG_T inf_config;
    memset(&inf_config, 0, sizeof(VMF_NNM_INFERENCE_APP_CONFIG_T)); // for safety let default 'bool' to 'false'

    int32_t left    = (int32_t)(_box->x1);
    int32_t top     = (int32_t)(_box->y1);
    int32_t right   = (int32_t)(_box->x2);
    int32_t bottom  = (int32_t)(_box->y2);

    // image buffer address should be just after the header
    inf_config.num_image                            = 1;
    inf_config.image_list[0].image_buf              = (void *)((uintptr_t)_input_header + sizeof(demo_customize_inf_multiple_models_header_t));
    inf_config.image_list[0].image_width            = _input_header->width;
    inf_config.image_list[0].image_height           = _input_header->height;
    inf_config.image_list[0].image_channel          = 3;                                    // assume RGB565
    inf_config.image_list[0].image_format           = KP_IMAGE_FORMAT_RGB565;               // assume RGB565
    inf_config.image_list[0].image_norm             = KP_NORMALIZE_KNERON;                  // this depends on model
    inf_config.image_list[0].image_resize           = KP_RESIZE_ENABLE;                     // default: enable resize
    inf_config.image_list[0].image_padding          = KP_PADDING_DISABLE;                   // default: disable padding
    inf_config.model_id                             = KNERON_PERSONCLASSIFIER_MB_56_48_3;   // this depends on model

    // set crop box
    inf_config.image_list[0].enable_crop            = true;                                 // enable crop image in ncpu/npu
    inf_config.image_list[0].crop_area.crop_number  = 0;
    inf_config.image_list[0].crop_area.x1           = left;
    inf_config.image_list[0].crop_area.y1           = top;
    inf_config.image_list[0].crop_area.width        = right - left;
    inf_config.image_list[0].crop_area.height       = bottom - top;

    // setting pre/post-proc configuration
    inf_config.pre_proc_config                      = NULL;
    inf_config.post_proc_config                     = NULL;
    inf_config.post_proc_func                       = user_post_classifier_top_n;

    // set up imagenet result output buffer for ncpu/npu
    inf_config.ncpu_result_buf                      = (void *)_imagenet_result;

    return VMF_NNM_Inference_App_Execute(&inf_config);
}

void demo_customize_inf_multiple_model(int job_id, int num_input_buf, void **inf_input_buf_list)
{
    // 'inf_input_buf' and 'inf_result_buf' are provided by kdp2 middleware
    // the content of 'inf_input_buf' is transmitted from host SW = header + image
    // 'inf_result_buf' is used to carry inference result back to host SW = header + inferernce result (from ncpu/npu)

    // now get an available free result buffer
    // normally the begin part of result buffer should contain app-defined result header
    // and the rest is for ncpu/npu inference output data

    // verify that the input data number meets the requirements of the model
    if (1 != num_input_buf) {
        VMF_NNM_Fifoq_Manager_Status_Code_Enqueue(job_id, KP_FW_WRONG_INPUT_BUFFER_COUNT_110);
        return;
    }

    int inf_status;
    int result_buf_size;
    uintptr_t inf_result_buf;
    uintptr_t inf_result_phy_addr;

    if (KP_SUCCESS != VMF_NNM_Fifoq_Manager_Result_Get_Free_Buffer(&inf_result_buf, &inf_result_phy_addr, &result_buf_size, -1)) {
        printf("[%s] get result free buffer failed\n", __FUNCTION__);
        return;
    }

    demo_customize_inf_multiple_models_header_t *input_header   = (demo_customize_inf_multiple_models_header_t *)inf_input_buf_list[0];
    demo_customize_inf_multiple_models_result_t *output_result  = (demo_customize_inf_multiple_models_result_t *)inf_result_buf;

    // pre set up result header stuff
    // header_stamp is a must to correctly transfer result data back to host SW
    output_result->header_stamp.magic_type  = KDP2_MAGIC_TYPE_INFERENCE;
    output_result->header_stamp.total_size  = sizeof(demo_customize_inf_multiple_models_result_t);
    output_result->header_stamp.job_id      = job_id;

    // this app needs extra DDR buffers for ncpu result
    int status = init_temp_buffer();
    if (!status) {
        // notify host error !
        output_result->header_stamp.status_code = KP_FW_DDR_MALLOC_FAILED_102;
        VMF_NNM_Fifoq_Manager_Result_Enqueue(inf_result_buf, inf_result_phy_addr, result_buf_size, -1, false);
        return;
    }

    // do pedestrian detection
    inf_status = inference_pedestrian_detection(input_header, yolo_pd_result);
    if (inf_status != KP_SUCCESS) {
        // notify host error !
        output_result->header_stamp.status_code = inf_status;
        VMF_NNM_Fifoq_Manager_Result_Enqueue(inf_result_buf, inf_result_phy_addr, result_buf_size, -1, false);
        return;
    }

    int box_count = 0;
    pd_classification_result_t *pd_result = &output_result->pd_classification_result;
    for (int i = 0; i < (int)yolo_pd_result->box_count; i++) {
        struct ex_bounding_box_s *box = &yolo_pd_result->boxes[i];

        if (KP_APP_PD_CLASS_PERSON == box->class_num) {
            // do face landmark for each faces
            inf_status = inference_pedestrian_classification(input_header, box, imagenet_result);

            if (KP_SUCCESS != inf_status) {
                // notify host error !
                output_result->header_stamp.status_code = inf_status;
                VMF_NNM_Fifoq_Manager_Result_Enqueue(inf_result_buf, inf_result_phy_addr, result_buf_size, -1, false);
                return;
            }

            // pedestrian_imagenet_classification result (class 0 : background, class 1: person)
            if (1 == imagenet_result->top_n_results[0].class_num)
                pd_result->pds[box_count].pd_class_score = imagenet_result->top_n_results[0].score;
            else
                pd_result->pds[box_count].pd_class_score = imagenet_result->top_n_results[1].score;
        }
        else{
            pd_result->pds[box_count].pd_class_score = 0;
        }

        memcpy(&pd_result->pds[box_count].pd, box, sizeof(kp_bounding_box_t));
        box_count++;
    }

    pd_result->box_count                    = box_count;

    output_result->header_stamp.status_code = KP_SUCCESS;
    output_result->header_stamp.total_size  = sizeof(demo_customize_inf_multiple_models_result_t) - sizeof(pd_classification_result_t) +
                                              sizeof(pd_result->box_count) + (box_count * sizeof(one_pd_classification_result_t));

    // send output result buffer back to host SW
    VMF_NNM_Fifoq_Manager_Result_Enqueue(inf_result_buf, inf_result_phy_addr, result_buf_size, -1, false);
}

void demo_customize_inf_multiple_model_deinit()
{
    deinit_temp_buffer();
}