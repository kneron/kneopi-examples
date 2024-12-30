
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <getopt.h>
#include <unistd.h>

extern "C" {
#include <vmf_nnm_inference_app.h>
#include <vmf_nnm_fifoq_manager.h>
#include "kdp2_inf_app_yolo.h"
}

// inference app
#include "application_init.h"

static void _app_func(int num_input_buf, void** inf_input_buf_list);

void _app_func(int num_input_buf, void** inf_input_buf_list)
{
    if (0 >= num_input_buf) {
        printf("No input buffer for app function\n");
        return;
    }

    kp_inference_header_stamp_t *header_stamp = (kp_inference_header_stamp_t *)inf_input_buf_list[0];
    unsigned int job_id = header_stamp->job_id;

    switch (job_id)
    {
    case KDP2_INF_ID_APP_YOLO:
        kdp2_app_yolo_inference(job_id, num_input_buf, (void**)inf_input_buf_list);
        break;
    case KDP2_JOB_ID_APP_YOLO_CONFIG_POST_PROC:
        kdp2_app_yolo_config_post_process_parameters(job_id, num_input_buf, (void**)inf_input_buf_list);
        break;
    default:
        VMF_NNM_Fifoq_Manager_Status_Code_Enqueue(job_id, KP_FW_ERROR_UNKNOWN_APP);
        printf("%s, unsupported job_id %d \n", __func__, job_id);
        break;
    }
}

void app_initialize(void)
{
    printf(">> Start running KL730 KDP2 HOST_SENSOR mode ...\n");

    /* initialize inference app */
    /* register APP functions */
    /* specify depth of inference queues */
    VMF_NNM_Inference_App_Init(_app_func);
    VMF_NNM_Fifoq_Manager_Init();

    return;
}

static void _app_func_deinit(unsigned int job_id);

void _app_func_deinit(unsigned int job_id)
{
    switch (job_id)
    {
    case KDP2_INF_ID_APP_YOLO:
        kdp2_app_yolo_inference_deinit();
        break;
    default:
        printf("%s, unsupported job_id %d \n", __func__, job_id);
        break;
    }
}

void app_destroy(void)
{
    _app_func_deinit(KDP2_INF_ID_APP_YOLO);

    VMF_NNM_Inference_App_Destroy();
    VMF_NNM_Fifoq_Manager_Destroy();
}
