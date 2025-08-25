/**
 * @file        kl520_demo_cam_user_define_api_drop_frame.cpp
 * @brief       main code of user define api with camera application
 * @version     1.0
 * @date        2021-06-13
 *
 * @copyright   Copyright (c) 2021 Kneron Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

extern "C"
{
#include "kp_core.h"
#include "kp_inference.h"
#include "helper_functions.h"
#include "postprocess.h"
}

#include <opencv2/opencv.hpp>
#include <mutex>

static char _scpu_fw_path[128] = "../res/firmware/KL730/kp_firmware.tar";
static char _model_file_path[128] = "../res/models/KL730/YoloV5s_640_640_3/models_730.nef";
static char _webcam_path[128] = "/dev/video0";

static bool _receive_running = true;
static std::mutex _mutex_result;
static kp_device_group_t _device;
static kp_model_nef_descriptor_t _model_desc;
static kp_yolo_result_t _yolo_result_latest = {0};
static kp_generic_image_inference_desc_t _input_data;
static kp_generic_image_inference_result_header_t _output_desc;
static int _image_width;
static int _image_height;
static int _cur_image_index = 0;
static int _cur_result_index = 0;

static cv::VideoCapture _cv_camera_cap;
static cv::Mat _cv_img_show;

void *image_send_function(void *data)
{
    cv::Mat cv_img_rgb565;

    /******* set up the input descriptor *******/
    _input_data.model_id                                = _model_desc.models[0].id; // first model ID
    _input_data.inference_number                        = 0;                        // inference number, used to verify with output result
    _input_data.num_input_node_image                    = 1;                        // number of image

    _input_data.input_node_image_list[0].resize_mode    = KP_RESIZE_ENABLE;         // enable resize in pre-process
    _input_data.input_node_image_list[0].padding_mode   = KP_PADDING_CORNER;        // enable corner padding in pre-process
    _input_data.input_node_image_list[0].normalize_mode = KP_NORMALIZE_KNERON;      // this depends on models
    _input_data.input_node_image_list[0].image_format   = KP_IMAGE_FORMAT_RGB565;   // image format
    _input_data.input_node_image_list[0].width          = _image_width;             // image width
    _input_data.input_node_image_list[0].height         = _image_height;            // image height
    _input_data.input_node_image_list[0].crop_count     = 0;                        // number of crop area, 0 means no cropping

    while (true == _receive_running)
    {
        /******* Get one frame from camera *******/
        _cv_camera_cap.read(_cv_img_show);

        cv::cvtColor(_cv_img_show, cv_img_rgb565, cv::COLOR_BGR2BGR565); // convert image color fomart to Kneron-specified

        /******* Send buffer to generic inference *******/
        _input_data.input_node_image_list[0].image_buffer = (uint8_t *)cv_img_rgb565.data;    // buffer of image data
        int ret = kp_generic_image_inference_send(_device, &_input_data);
        if ((ret != KP_SUCCESS) && (true == _receive_running))
        {
            printf("kp_generic_raw_inference_send() error = %d (%s)\n", ret, kp_error_string(ret));
            break;
        }

        _cur_image_index++;
    }

    return NULL;
}

void *result_receive_function(void *data)
{
    uint32_t inference_number;
    kp_inf_float_node_output_t *output_nodes[3] = {NULL};
    kp_yolo_result_t yolo_result                = {0};

    /******* allocate memory for raw output *******/
    uint32_t raw_buf_size = _model_desc.models[0].max_raw_out_size;
    uint8_t *raw_output_buf = (uint8_t *)malloc(raw_buf_size);

    while (_receive_running)
    {
        /******* Receive one result of generic inference */
        int ret = kp_generic_image_inference_receive(_device, &_output_desc, raw_output_buf, raw_buf_size);

        if (KP_ERROR_USB_TIMEOUT_N7 == ret) {
            continue;
        } else if (KP_SUCCESS != ret) {
            printf("kp_generic_raw_inference_receive() error = %d (%s)\n", ret, kp_error_string(ret));
            break;
        }

        /******* retrieve output nodes in floating point format */
        output_nodes[0] = kp_generic_inference_retrieve_float_node(0, raw_output_buf, KP_CHANNEL_ORDERING_DEFAULT);
        output_nodes[1] = kp_generic_inference_retrieve_float_node(1, raw_output_buf, KP_CHANNEL_ORDERING_DEFAULT);
        output_nodes[2] = kp_generic_inference_retrieve_float_node(2, raw_output_buf, KP_CHANNEL_ORDERING_DEFAULT);

        /******* post-process yolo v5 output nodes to class/bounding boxes */
        post_process_yolo_v5_720(output_nodes, _output_desc.num_output_node, &_output_desc.pre_proc_info[0], 0.2, &yolo_result);

        free(output_nodes[0]);
        free(output_nodes[1]);
        free(output_nodes[2]);

        ++_cur_result_index;

        _mutex_result.lock();
        memcpy((void *)&_yolo_result_latest, (void *)&yolo_result, sizeof(kp_yolo_result_t));
        _mutex_result.unlock();
    }

    free(raw_output_buf);

    return NULL;
}

void init_camera() {
    /******* Open USB camera *******/
    bool opened = _cv_camera_cap.open(_webcam_path, cv::CAP_V4L2);
    printf("open camera ... %s\n", (opened) ? "OK" : "failed");

    /******* Setting frame size may failed in OpenCV *******/
    _image_width = static_cast<int>(_cv_camera_cap.get(cv::CAP_PROP_FRAME_WIDTH));
    _image_height = static_cast<int>(_cv_camera_cap.get(cv::CAP_PROP_FRAME_HEIGHT));
}

void dispalay() {
    char imgFPS[60]     = "image FPS: ";
    char infFPS[60]     = "inference FPS: ";
    char strImgRes[60];
    char strModelRes[60];
    char box_info[128];
    int image_count     = 0;
    int result_count    = 0;

    /******* record last result *******/
    static kp_yolo_result_t _yolo_result_last = {0};

    /******* record stabilized bounding boxes result *******/
    static uint32_t box_count_stabilized;
    static kp_bounding_box_t boxes_stabilized[YOLO_GOOD_BOX_MAX];

    /******* initialize cv mat for display *******/
    cv::Mat cv_img_show;

    /******* prepare display window *******/
    cv::namedWindow("Generic Image Inference", cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);

    /******* prepare image and model resolution *******/
    sprintf(strImgRes, "image: %d x %d", _image_width, _image_height);
    sprintf(strModelRes, "model: %d x %d", _model_desc.models[0].input_nodes[0].tensor_shape_info.tensor_shape_info_data.v2.shape[3], _model_desc.models[0].input_nodes[0].tensor_shape_info.tensor_shape_info_data.v2.shape[2]);

    /******* initial FPS calculate variables *******/
    image_count     = _cur_image_index;
    result_count    = _cur_result_index;
    helper_measure_time_begin();

    while (true) {
        if (_cv_img_show.empty())
            continue;

        cv_img_show = _cv_img_show.clone();

        _mutex_result.lock();

        /******* This function is optional to avoid box "jumping" *******/
        helper_bounding_box_stabilization(_yolo_result_last.box_count, _yolo_result_last.boxes, _yolo_result_latest.box_count, _yolo_result_latest.boxes,
                                          &box_count_stabilized, boxes_stabilized, 20, 0.3);

        /******* Draw all bounding boxes *******/
        for (uint32_t i = 0; i < box_count_stabilized; i++)
        {
            int x1 = boxes_stabilized[i].x1;
            int y1 = boxes_stabilized[i].y1;
            int x2 = boxes_stabilized[i].x2;
            int y2 = boxes_stabilized[i].y2;

            int textX = x1;
            int textY = y1 - 10;
            if (textY < 0)
                textY = y1 + 5;

            sprintf(box_info, "class %d (%.2f)", boxes_stabilized[i].class_num, boxes_stabilized[i].score);

            cv::rectangle(cv_img_show, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(50, 255, 50), 2);
            cv::putText(cv_img_show, box_info, cv::Point(textX, textY), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(50, 50, 255), 1);
        }

        memcpy((void *)&_yolo_result_last, (void *)&_yolo_result_latest, sizeof(kp_yolo_result_t));

        _mutex_result.unlock();

        /******* calculate FPS every 60 frames *******/
        if ((_cur_image_index - image_count) > 60)
        {
            double time_spent;
            helper_measure_time_end(&time_spent);
            sprintf(imgFPS, "image FPS: %.2lf", (_cur_image_index - image_count) / time_spent);
            sprintf(infFPS, "inference FPS: %.2lf", (_cur_result_index - result_count) / time_spent);

            image_count = _cur_image_index;
            result_count = _cur_result_index;
            helper_measure_time_begin();
        }

        cv::putText(cv_img_show, imgFPS, cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(50, 50, 255), 1);
        cv::putText(cv_img_show, infFPS, cv::Point(5, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(50, 50, 255), 1);
        cv::putText(cv_img_show, strImgRes, cv::Point(5, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(50, 50, 255), 1);
        cv::putText(cv_img_show, strModelRes, cv::Point(5, 80), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(50, 50, 255), 1);
        cv::putText(cv_img_show, "Press 'ESC' to exit", cv::Point(10, cv_img_show.rows - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 2);

        cv::imshow("Generic Image Inference", cv_img_show);

        /******* Press 'ESC' to exit *******/
        if (27 == cv::waitKey(10))
        {
            break;
        }
    }

    _receive_running = false;
    cv::destroyAllWindows();
}

int main(int argc, char *argv[])
{
    /******* each device has a unique port ID, 0 for auto-search *******/
    int port_id = (argc > 1) ? atoi(argv[1]) : 0;
    int ret;

    /******* reboot the device *******/
    _device = kp_connect_devices(1, &port_id, NULL);
    printf("connect device ... %s\n", (_device) ? "OK" : "failed");

    kp_set_timeout(_device, 5000); // 5 secs timeout

    /******* upload firmware to device *******/
    ret = kp_load_firmware_from_file(_device, _scpu_fw_path, NULL);
    printf("upload firmware ... %s\n", (ret == KP_SUCCESS) ? "OK" : "failed");

    /******* upload model to device *******/
    ret = kp_load_model_from_file(_device, _model_file_path, &_model_desc);
    printf("upload model ... %s\n", (ret == KP_SUCCESS) ? "OK" : "failed");
    fflush(stdout);

    /******* configure inference settings (make it frame-droppabe for real-time purpose) *******/
    kp_inf_configuration_t infConf = {.enable_frame_drop = true};
    ret = kp_inference_configure(_device, &infConf);
    printf("configure inference frame-droppable ... %s\n", (ret == KP_SUCCESS) ? "OK" : "failed");
    fflush(stdout);

    /******* initial camera resource *******/
    init_camera();

    /******* create send image thread and receive result thread *******/
    pthread_t image_send_thd, result_recv_thd;

    pthread_create(&image_send_thd, NULL, image_send_function, NULL);
    pthread_create(&result_recv_thd, NULL, result_receive_function, NULL);

    /******* display result *******/
    dispalay();

    /******* join send image thread and receive result thread *******/
    pthread_join(image_send_thd, NULL);
    pthread_join(result_recv_thd, NULL);

    printf("\ndisconnecting device ...\n");

    /******* release plus resource *******/
    kp_release_model_nef_descriptor(&_model_desc);
    kp_disconnect_devices(_device);

    return 0;
}
