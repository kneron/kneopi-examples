#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>
#include <signal.h>
#include <pthread.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>

extern "C" {
#include "kdp2_inf_app_yolo.h"
}

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

extern void sig_kill(int signo);

int draw_display_image(cv::Mat *cv_img_display, const char *strImgFPS, const char *strInfFPS)
{
    int ret = KP_SUCCESS;
    kp_inference_header_stamp_t *header_stamp = (kp_inference_header_stamp_t *)_inf_result.result_buffer;

    if (KDP2_INF_ID_APP_YOLO == header_stamp->job_id)
    {
        pthread_mutex_lock(&_mutex_result);
        kdp2_ipc_app_yolo_result_t *app_yolo_result = (kdp2_ipc_app_yolo_result_t *)header_stamp;
        kp_app_yolo_result_t *yolo_result = (kp_app_yolo_result_t *)&app_yolo_result->yolo_data;

        for (uint32_t i = 0; i < yolo_result->box_count; i++) {
            cv::rectangle(*cv_img_display, cv::Point(yolo_result->boxes[i].x1, yolo_result->boxes[i].y1),
                            cv::Point(yolo_result->boxes[i].x2, yolo_result->boxes[i].y2), cv::Scalar(50, 255, 50), 2);
        }
        pthread_mutex_unlock(&_mutex_result);

        cv::putText(*cv_img_display, strImgFPS, cv::Point(5, 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(50, 50, 255), 1);
        cv::putText(*cv_img_display, strInfFPS, cv::Point(5, 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(50, 50, 255), 1);
        cv::putText(*cv_img_display, "Press 'ESC' to exit", cv::Point(10, cv_img_display->rows - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255), 2);
    }
    else
    {
        ret = KP_FW_ERROR_UNKNOWN_APP;
    }

    return ret;
}

void *example_display_liveview_thread(void *)
{
    struct timeval time_begin;
    struct timeval time_end;
    float time_spent = 0.0;
    char strImgFPS[50] = "Image FPS: ";
    char strInfFPS[50] = "Inference FPS: ";
    cv::Mat cv_image_source;
    cv::Mat cv_image_display;

    cv::namedWindow("Inference Display", cv::WINDOW_AUTOSIZE | cv::WINDOW_GUI_NORMAL);
    gettimeofday(&time_begin, NULL);

    while (true == _blDisplayRunning) {

        if (_result_count >= 60)
        {
            gettimeofday(&time_end, NULL);
            time_spent = (float)(time_end.tv_sec - time_begin.tv_sec) + (float)(time_end.tv_usec - time_begin.tv_usec) * .000001;
            sprintf(strImgFPS, "Image FPS: %.2lf", _image_count / time_spent);
            sprintf(strInfFPS, "Inference FPS: %.2lf", _result_count / time_spent);
            _image_count = 0;
            _result_count = 0;

            gettimeofday(&time_begin, NULL);
        }

        pthread_mutex_lock(&_mutex_image);
        switch (_input_data.input_image_format) {
        case KP_IMAGE_FORMAT_RGB565:
            cv_image_source = cv::Mat(_input_data.input_image_height, _input_data.input_image_width, CV_8UC2, (void *)_input_data.input_buf_address);
            cv::cvtColor(cv_image_source, cv_image_display, cv::COLOR_BGR5652BGR);
            break;
        case KP_IMAGE_FORMAT_RGBA8888:
            cv_image_source = cv::Mat(_input_data.input_image_height, _input_data.input_image_width, CV_8UC4, (void *)_input_data.input_buf_address);
            cv::cvtColor(cv_image_source, cv_image_display, cv::COLOR_RGBA2BGR);
            break;
        case KP_IMAGE_FORMAT_YUV420:
            cv_image_source = cv::Mat(_input_data.input_image_height * 1.5, _input_data.input_image_width, CV_8UC1, (void *)_input_data.input_buf_address);
            cv::cvtColor(cv_image_source, cv_image_display, cv::COLOR_YUV2BGR_I420);
            break;
        default:
            cv_image_display = cv::Mat();
            break;
        }

        pthread_mutex_unlock(&_mutex_image);

        /* Display image */
        if (false == cv_image_display.empty()) {
            if (true == _inf_result.result_ready_display) {
                draw_display_image(&cv_image_display, strImgFPS, strInfFPS);
            }

            cv::imshow("Inference Display", cv_image_display);
        }

        /* Press 'ESC' to exit */
        if (27 == cv::waitKey(10)) {
            sig_kill(0);
            break;
        }
    }

    _blImageRunning = false;
    _blSendInfRunning = false;
    _blResultRunning = false;

    _blDispatchRunning = false;
    _blFifoqManagerRunning = false;

    return NULL;
}