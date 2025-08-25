/*
 * Kneron Application general functions
 *
 * Copyright (C) 2021 Kneron, Inc. All rights reserved.
 *
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "model_type.h"
#include "vmf_nnm_inference_app.h"
#include "vmf_nnm_fifoq_manager.h"

#include "demo_customize_inf_single_model.h"
#include "user_post_process_yolov5.h"
#include "user_pre_process_yolov5.h"

static ex_yolo_post_proc_config_t post_proc_params_v5s = {
    .prob_thresh                = 0.15,
    .nms_thresh                 = 0.5,
    .max_detection              = YOLO_BOX_MAX,
    .max_detection_per_class    = YOLO_BOX_MAX,
    .nms_mode                   = EX_NMS_MODE_SINGLE_CLASS,
    .anchor_layer_num           = 3,
    .anchor_cell_num_per_layer  = 3,
    .data                       = {{{10, 13}, {16, 30}, {33, 23}},
                                   {{30, 61}, {62, 45}, {59, 119}},
                                   {{116, 90}, {156, 198}, {373, 326}},
                                   {{0, 0}, {0, 0}, {0, 0}},
                                   {{0, 0}, {0, 0}, {0, 0}}},
};

void demo_customize_inf_single_model(int job_id, int num_input_buf, void **inf_input_buf_list)
{
    // 'inf_input_buf' and 'inf_result_buf' are provided by kdp2 middleware
    // the content of 'inf_input_buf' is transmitted from host SW = header + image
    // 'inf_result_buf' is used to carry inference result back to host SW = header + inference result (from ncpu/npu)

    // now get an available free result buffer
    // normally the begin part of result buffer should contain app-defined result header
    // and the rest is for ncpu/npu inference output data

    // verify that the input data number meets the requirements of the model
    if (1 != num_input_buf) {
        VMF_NNM_Fifoq_Manager_Status_Code_Enqueue(job_id, KP_FW_WRONG_INPUT_BUFFER_COUNT_110);
        return;
    }

    int result_buf_size;
    uintptr_t inf_result_buf;
    uintptr_t inf_result_phy_addr;

    if (KP_SUCCESS != VMF_NNM_Fifoq_Manager_Result_Get_Free_Buffer(&inf_result_buf, &inf_result_phy_addr, &result_buf_size, -1)) {
        printf("[%s] get result free buffer failed\n", __FUNCTION__);
        return;
    }

    demo_customize_inf_single_model_header_t *input_header  = (demo_customize_inf_single_model_header_t *)inf_input_buf_list[0];
    demo_customize_inf_single_model_result_t *output_result = (demo_customize_inf_single_model_result_t *)inf_result_buf;

    // config image preprocessing and model settings
    VMF_NNM_INFERENCE_APP_CONFIG_T inf_config;
    memset(&inf_config, 0, sizeof(VMF_NNM_INFERENCE_APP_CONFIG_T)); // for safety let default 'bool' to 'false'

    // image buffer address should be just after the header
    inf_config.num_image                    = 1;
    inf_config.image_list[0].image_buf      = (void *)((uintptr_t)input_header + sizeof(demo_customize_inf_single_model_header_t));
    inf_config.image_list[0].image_width    = input_header->width;
    inf_config.image_list[0].image_height   = input_header->height;
    inf_config.image_list[0].image_channel  = 3;                                        // assume RGB565
    inf_config.image_list[0].image_format   = KP_IMAGE_FORMAT_RGB565;                   // assume RGB565
    inf_config.image_list[0].image_norm     = KP_NORMALIZE_KNERON;                      // this depends on model
    inf_config.image_list[0].image_resize   = KP_RESIZE_ENABLE;                         // enable resize
    inf_config.image_list[0].image_padding  = KP_PADDING_CORNER;                        // enable padding on corner
    inf_config.model_id                     = KNERON_YOLOV5S_COCO80_640_640_3;          // this depends on model

    // setting pre/post-proc configuration
    inf_config.pre_proc_config              = NULL;
    inf_config.pre_proc_func                = user_pre_process_yolov5;
    inf_config.post_proc_config             = (void *)&post_proc_params_v5s;            // yolo post-process configurations for yolo v5 series
    inf_config.post_proc_func               = user_post_yolov5_no_sigmoid;

    // set up pd result output buffer for ncpu/npu
    inf_config.ncpu_result_buf              = (void *)&(output_result->yolo_result);    // give result buffer for ncpu/npu, callback will carry it

    // run preprocessing and inference, trigger ncpu/npu to do the work
    // if enable_parallel=true (works only for single model), result callback is needed
    // however if inference error then no callback will be invoked
    int inf_status = VMF_NNM_Inference_App_Execute(&inf_config);

    // header_stamp is a must to correctly transfer result data back to host SW
    output_result->header_stamp.magic_type  = KDP2_MAGIC_TYPE_INFERENCE;
    output_result->header_stamp.total_size  = sizeof(demo_customize_inf_single_model_result_t);
    output_result->header_stamp.job_id      = job_id;
    output_result->header_stamp.status_code = inf_status;

    // send output result buffer back to host SW
    VMF_NNM_Fifoq_Manager_Result_Enqueue(inf_result_buf, inf_result_phy_addr, result_buf_size, -1, false);
}

void demo_customize_inf_single_model_deinit()
{
    //there is no temp buffer need to release in this model
}