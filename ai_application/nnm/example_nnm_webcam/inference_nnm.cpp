#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>
#include <signal.h>
#include <pthread.h>
#include <sys/time.h>

extern "C" {
#include "vmf_nnm_inference_app.h"
#include "vmf_nnm_fifoq_manager.h"
#include "vmf_nnm_ipc_cmd.h"
#include "kdp2_inf_app_yolo.h"
}

#include "example_shared_struct.h"
#include "kp_struct.h"
#include "model_type.h"

volatile extern NNM_SHARED_INPUT_T _input_data;
extern pthread_mutex_t _mutex_image;

extern bool _blDispatchRunning;
extern bool _blFifoqManagerRunning;

extern bool _blImageRunning;
extern bool _blDisplayRunning;

volatile bool _blSendInfRunning = true;
volatile bool _blResultRunning = true;

bool _enable_inf_droppable = true;

unsigned int _image_count = 0;
unsigned int _result_count = 0;

NNM_SHARED_RESULT_T _inf_result = {0};
pthread_mutex_t _mutex_result = PTHREAD_MUTEX_INITIALIZER;

bool init_config_yolo_params = false;
kp_app_yolo_post_proc_config_t post_proc_params_v5s = {
    .prob_thresh = 0.15,
    .nms_thresh = 0.5,
    .max_detection_per_class = 100,
    .anchor_row = 3,
    .anchor_col = 6,
    .stride_size = 3,
    .reserved_size = 0,
    .data = {
        // anchors[3][6]
        10, 13, 16, 30, 33, 23,
        30, 61, 62, 45, 59, 119,
        116, 90, 156, 198, 373, 326,
        // strides[3]
        8, 16, 32,
    },
};

int prepare_inference_header(uintptr_t buf_addr, int job_id)
{
    if (KDP2_INF_ID_APP_YOLO == job_id)
    {
        if (false == init_config_yolo_params) {
            /****************************************************************
             * configure application pre/post-processing params
             ****************************************************************/
            init_config_yolo_params = true;

            kdp2_ipc_app_yolo_post_proc_config_t *app_yolo_post_proc_config = (kdp2_ipc_app_yolo_post_proc_config_t *)buf_addr;
            kp_inference_header_stamp_t *header_stamp = &app_yolo_post_proc_config->header_stamp;

            header_stamp->magic_type = KDP2_MAGIC_TYPE_INFERENCE;
            header_stamp->total_size = sizeof(kdp2_ipc_app_yolo_post_proc_config_t);
            header_stamp->total_image = 1;
            header_stamp->image_index = 0;
            header_stamp->job_id = KDP2_JOB_ID_APP_YOLO_CONFIG_POST_PROC;

            app_yolo_post_proc_config->model_id = KNERON_YOLOV5S_COCO80_640_640_3;
            app_yolo_post_proc_config->param_size = sizeof(kp_app_yolo_post_proc_config_t);
            app_yolo_post_proc_config->set_or_get = 1;

            post_proc_params_v5s.prob_thresh = 0.5;
            memcpy((void *)app_yolo_post_proc_config->param_data, (void *)&post_proc_params_v5s, sizeof(kp_app_yolo_post_proc_config_t));

            return KP_SUCCESS;
        }

        pthread_mutex_lock(&_mutex_image);
        kdp2_ipc_app_yolo_inf_header_t *app_yolo_header = (kdp2_ipc_app_yolo_inf_header_t *)buf_addr;
        kp_inference_header_stamp_t *header_stamp = &app_yolo_header->header_stamp;
        int image_size = _input_data.input_buf_size;

        static uint32_t inf_number = 0;
        inf_number = inf_number + 1;
        inf_number %= 0xFFFFFFFF;   //To avoid overflow

        header_stamp->magic_type = KDP2_MAGIC_TYPE_INFERENCE;
        header_stamp->total_size = sizeof(kdp2_ipc_app_yolo_inf_header_t) + (uint32_t)image_size;
        header_stamp->total_image = 1;
        header_stamp->image_index = 0;
        header_stamp->job_id = KDP2_INF_ID_APP_YOLO;

        app_yolo_header->inf_number = inf_number;
        app_yolo_header->model_id = KNERON_YOLOV5S_COCO80_640_640_3;
        app_yolo_header->width = _input_data.input_image_width;
        app_yolo_header->height = _input_data.input_image_height;

        app_yolo_header->image_format = _input_data.input_image_format;
        app_yolo_header->model_normalize = KP_NORMALIZE_KNERON;

        memcpy((void *)(buf_addr + sizeof(kdp2_ipc_app_yolo_inf_header_t)), (void *)_input_data.input_buf_address, image_size);

        _input_data.input_ready_inf = false;
        pthread_mutex_unlock(&_mutex_image);
    }
    else
    {
        printf("[%s] Error: Job ID %u\n", __FUNCTION__, job_id);
        return KP_FW_ERROR_UNKNOWN_APP;
    }

    return KP_SUCCESS;
}

void *example_send_inf_thread(void *arg)
{
    int *job_id = (int *)arg;

    kp_inference_header_stamp_t *header_stamp = NULL;
    uintptr_t buf_addr = 0;      // contains a inference image or a command
    uintptr_t phy_buf_addr = 0;  // contains a inference image or a command
    int buf_size = 0;           // buffer size should bigger than inference image size
    int sts = 0;

    while (true == _blSendInfRunning)
    {
        if (0 == _input_data.input_ready_inf) {
            continue;
        }

        // take a free buffer to receive a inf image or a command
        if (true == VMF_NNM_Fifoq_Manager_Get_Fifoq_Allocated()) {
            sts = VMF_NNM_Fifoq_Manager_Image_Get_Free_Buffer(&buf_addr, &phy_buf_addr, &buf_size, -1, _enable_inf_droppable);

            if (KP_FW_FIFOQ_ACCESS_FAILED_125 == sts) {
                continue;
            } else if (KP_SUCCESS != sts) {
                printf("[%s] Error: FIFO queue error %d.\n", __FUNCTION__, sts);
                goto EXIT_FREAD_IMAGE_THREAD;
            }
        }
        else {
            printf("[%s] Error: FIFO queue is not allocated.\n", __FUNCTION__);
            goto EXIT_FREAD_IMAGE_THREAD;
        }

        sts = prepare_inference_header(buf_addr, *job_id);
        if (KP_SUCCESS != sts) {
            printf("[%s] Error: prepare_inference_header failed (%d)\n", __FUNCTION__, sts);
            goto EXIT_FREAD_IMAGE_THREAD_PUT_FREE_QUEUE;
        }

        header_stamp = (kp_inference_header_stamp_t *)buf_addr;

        if (header_stamp->magic_type == KDP2_MAGIC_TYPE_INFERENCE)
        {
            if (header_stamp->total_size > (uint32_t)buf_size) {
                printf("[%s] Error: Inference image size (%d) is bigger than buffer size (%d)\n", __FUNCTION__, header_stamp->total_size, buf_size);
                goto EXIT_FREAD_IMAGE_THREAD_PUT_FREE_QUEUE;
            }

            VMF_NNM_Fifoq_Manager_Image_Enqueue(header_stamp->total_image, header_stamp->image_index, buf_addr, phy_buf_addr, buf_size, 0, false);
        }
        else
        {
            printf("[%s] Error: Invalid magic type %u\n", __FUNCTION__, header_stamp->magic_type);
            VMF_NNM_Fifoq_Manager_Image_Put_Free_Buffer(buf_addr, phy_buf_addr, buf_size, 0);
        }

        _image_count++;
    }

EXIT_FREAD_IMAGE_THREAD_PUT_FREE_QUEUE:
    VMF_NNM_Fifoq_Manager_Image_Put_Free_Buffer(buf_addr, phy_buf_addr, buf_size, 0);

EXIT_FREAD_IMAGE_THREAD:

    _blImageRunning = false;
    _blResultRunning = false;
    _blDisplayRunning = false;

    _blDispatchRunning = false;
    _blFifoqManagerRunning = false;

    return NULL;
}

void *example_recv_result_thread(void *)
{
    kp_inference_header_stamp_t *header_stamp;
    uintptr_t buf_addr = 0;
    uintptr_t phy_buf_addr = 0;
    int buf_size = 0;
    int sts = 0;

    while (true == _blResultRunning) {
        // get result data from queue blocking wait
        int ret = VMF_NNM_Fifoq_Manager_Result_Dequeue(&buf_addr, &phy_buf_addr, &buf_size, -1);

        if (KP_FW_FIFOQ_ACCESS_FAILED_125 == ret) {
            continue;
        } else if (KP_SUCCESS != ret) {
            printf("[%s] Error: FIFO queue error %d.\n", __FUNCTION__, sts);
            goto EXIT_UPDATE_RESULT_THREAD;
        } else if (_inf_result.result_buffer_size < buf_size) {
            _inf_result.result_buffer = (uintptr_t)realloc((void *)_inf_result.result_buffer, buf_size);
            _inf_result.result_buffer_size = buf_size;
        }

        _result_count++;

        header_stamp = (kp_inference_header_stamp_t *)buf_addr;

        if (KP_SUCCESS != header_stamp->status_code) {
            printf("[%s] Error: status_code %d\n", __FUNCTION__, header_stamp->status_code);
            goto EXIT_UPDATE_RESULT_THREAD_PUT_FREE_QUEUE;
        }

        pthread_mutex_lock(&_mutex_result);

        memcpy((void *)_inf_result.result_buffer, (void *)buf_addr, buf_size);
        _inf_result.result_ready_display = true;

        pthread_mutex_unlock(&_mutex_result);

        // return free buf back to queue
        ret = VMF_NNM_Fifoq_Manager_Result_Put_Free_Buffer(buf_addr, phy_buf_addr, buf_size, -1);
        if (KP_FW_FIFOQ_ACCESS_FAILED_125 == ret) {
            continue;
        } else if (KP_SUCCESS != ret) {
            goto EXIT_UPDATE_RESULT_THREAD;
        }
    }

EXIT_UPDATE_RESULT_THREAD_PUT_FREE_QUEUE:
    VMF_NNM_Fifoq_Manager_Result_Put_Free_Buffer(buf_addr, phy_buf_addr, buf_size, -1);

EXIT_UPDATE_RESULT_THREAD:

    if (0 != _inf_result.result_buffer) {
        free((void *)_inf_result.result_buffer);
    }

    _blImageRunning = false;
    _blSendInfRunning = false;
    _blDisplayRunning = false;

    _blDispatchRunning = false;
    _blFifoqManagerRunning = false;

    return NULL;
}
