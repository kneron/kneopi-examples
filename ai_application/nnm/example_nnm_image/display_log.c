#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>
#include <signal.h>
#include <pthread.h>
#include <sys/time.h>

#include "kdp2_inf_app_yolo.h"
#include "demo_customize_inf_single_model.h"
#include "demo_customize_inf_multiple_models.h"
#include "demo_customize_inf_single_model_with_sw_npu_format_convert.h"

#include "example_shared_struct.h"
#include "kp_struct.h"

volatile extern NNM_SHARED_INPUT_T _input_data;
extern pthread_mutex_t _mutex_image;

volatile extern NNM_SHARED_RESULT_T _inf_result;
extern pthread_mutex_t _mutex_result;

extern unsigned int _image_count;
extern unsigned int _result_count;

extern bool _blDispatchRunning;
extern bool _blFifoqManagerRunning;

extern bool _blImageRunning;
extern bool _blSendInfRunning;
extern bool _blResultRunning;

volatile bool _blDisplayRunning = true;

extern unsigned int _loop_time;

int print_result_on_log()
{
    int ret = KP_SUCCESS;
    kp_inference_header_stamp_t *header_stamp = (kp_inference_header_stamp_t *)_inf_result.result_buffer;

    if (KDP2_INF_ID_APP_YOLO == header_stamp->job_id)
    {
        kdp2_ipc_app_yolo_result_t *app_yolo_result = (kdp2_ipc_app_yolo_result_t *)header_stamp;
        kp_app_yolo_result_t *yolo_result = (kp_app_yolo_result_t *)&app_yolo_result->yolo_data;

        printf("[Result] Inf Number %u, Box Count = %u\n", app_yolo_result->inf_number, yolo_result->box_count);
        for (uint32_t i = 0; i < yolo_result->box_count; i++) {
            printf("    [%d] x1 = %f, y1 = %f, x2 = %f, y2 = %f, score = %f, class_num = %d\n", i,
                                                                                                yolo_result->boxes[i].x1,
                                                                                                yolo_result->boxes[i].y1,
                                                                                                yolo_result->boxes[i].x2,
                                                                                                yolo_result->boxes[i].y2,
                                                                                                yolo_result->boxes[i].score,
                                                                                                yolo_result->boxes[i].class_num);
        }
    }
    else if (DEMO_KL730_CUSTOMIZE_INF_SINGLE_MODEL_JOB_ID == header_stamp->job_id)
    {
        demo_customize_inf_single_model_result_t *app_customize_result = (demo_customize_inf_single_model_result_t *)header_stamp;
        kp_custom_yolo_result_t *customize_yolo_result = (kp_custom_yolo_result_t *)&app_customize_result->yolo_result;

        printf("[Customize Result] Box Count = %u\n", customize_yolo_result->box_count);
        for (uint32_t i = 0; i < customize_yolo_result->box_count; i++) {
            printf("    [%d] x1 = %f, y1 = %f, x2 = %f, y2 = %f, score = %f, class_num = %d\n", i,
                                                                                                customize_yolo_result->boxes[i].x1,
                                                                                                customize_yolo_result->boxes[i].y1,
                                                                                                customize_yolo_result->boxes[i].x2,
                                                                                                customize_yolo_result->boxes[i].y2,
                                                                                                customize_yolo_result->boxes[i].score,
                                                                                                customize_yolo_result->boxes[i].class_num);
        }
    }
    else if (DEMO_KL730_CUSTOMIZE_INF_MULTIPLE_MODEL_JOB_ID == header_stamp->job_id)
    {
        demo_customize_inf_multiple_models_result_t *app_customize_result = (demo_customize_inf_multiple_models_result_t *)header_stamp;
        pd_classification_result_t *customize_pd_result = (pd_classification_result_t *)&app_customize_result->pd_classification_result;

        printf("[Customize Result] PD Count = %u\n", customize_pd_result->box_count);
        for (uint32_t i = 0; i < customize_pd_result->box_count; i++) {
            printf("    [%d] x1 = %f, y1 = %f, x2 = %f, y2 = %f, " \
                   "score = %f, class_num = %d, pd_class_score = %lf\n", i,
                                                                         customize_pd_result->pds[i].pd.x1,
                                                                         customize_pd_result->pds[i].pd.y1,
                                                                         customize_pd_result->pds[i].pd.x2,
                                                                         customize_pd_result->pds[i].pd.y2,
                                                                         customize_pd_result->pds[i].pd.score,
                                                                         customize_pd_result->pds[i].pd.class_num,
                                                                         customize_pd_result->pds[i].pd_class_score);
        }
    }
    else if (DEMO_KL730_CUSTOMIZE_INF_SINGLE_MODEL_WITH_SW_NPU_FORMAT_CONVERT_JOB_ID == header_stamp->job_id)
    {
        demo_customize_inf_single_model_with_sw_npu_format_convert_result_t *app_customize_result = (demo_customize_inf_single_model_with_sw_npu_format_convert_result_t *)header_stamp;
        kp_custom_with_sw_npu_format_convert_yolo_result_t *customize_yolo_result = (kp_custom_with_sw_npu_format_convert_yolo_result_t *)&app_customize_result->yolo_result;

        printf("[Customize Result] Box Count = %u\n", customize_yolo_result->box_count);
        for (uint32_t i = 0; i < customize_yolo_result->box_count; i++) {
            printf("    [%d] x1 = %f, y1 = %f, x2 = %f, y2 = %f, score = %f, class_num = %d\n", i,
                                                                                                customize_yolo_result->boxes[i].x1,
                                                                                                customize_yolo_result->boxes[i].y1,
                                                                                                customize_yolo_result->boxes[i].x2,
                                                                                                customize_yolo_result->boxes[i].y2,
                                                                                                customize_yolo_result->boxes[i].score,
                                                                                                customize_yolo_result->boxes[i].class_num);
        }
    }
    else
    {
        ret = KP_FW_ERROR_UNKNOWN_APP;
    }

    return ret;
}

void *example_display_log_thread(void *)
{
    while (true == _blDisplayRunning) {
        pthread_mutex_lock(&_mutex_result);

        if (true == _inf_result.result_ready_display) {
            print_result_on_log();
            _inf_result.result_ready_display = false;
        }

        pthread_mutex_unlock(&_mutex_result);
    }

    _blImageRunning = false;
    _blSendInfRunning = false;
    _blResultRunning = false;

    _blDispatchRunning = false;
    _blFifoqManagerRunning = false;

    return NULL;
}