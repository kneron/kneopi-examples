#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>
#include <signal.h>
#include <pthread.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>

#include "example_shared_struct.h"
#include "kp_struct.h"

extern bool _blDispatchRunning;
extern bool _blFifoqManagerRunning;

extern bool _blSendInfRunning;
extern bool _blResultRunning;
extern bool _blDisplayRunning;

bool _blImageRunning = true;

NNM_SHARED_INPUT_T _input_data = {0};
pthread_mutex_t _mutex_image = PTHREAD_MUTEX_INITIALIZER;

void *example_webcam_input_thread(void *arg)
{
    EXAMPLE_WEBCAM_INIT_OPT_T* pInitOpt=(EXAMPLE_WEBCAM_INIT_OPT_T*)arg;

    cv::VideoCapture cv_camera_cap;
    cv::Mat cv_read_camera, cv_img_to_be_sent;

    if (false == cv_camera_cap.open(pInitOpt->pszCameraPath, cv::CAP_V4L2)) {
        printf("[%s] open camera failed %s\n", __FUNCTION__, pInitOpt->pszCameraPath);
        goto EXIT_FREAD_IMAGE_THREAD;
    }

    /* Setting frame size may failed in OpenCV */
    cv_camera_cap.set(cv::CAP_PROP_FRAME_WIDTH, pInitOpt->dwImageWidth);
    cv_camera_cap.set(cv::CAP_PROP_FRAME_HEIGHT, pInitOpt->dwImageHeight);
    cv_camera_cap.set(cv::CAP_PROP_FPS, pInitOpt->dwFps);

    pInitOpt->dwImageWidth = (unsigned int)cv_camera_cap.get(cv::CAP_PROP_FRAME_WIDTH);
    pInitOpt->dwImageHeight = (unsigned int)cv_camera_cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    while (true == _blImageRunning)
    {
        cv_camera_cap.read(cv_read_camera);

        pthread_mutex_lock(&_mutex_image);
        cv::cvtColor(cv_read_camera, cv_img_to_be_sent, cv::COLOR_BGR2RGBA);

        _input_data.input_buf_address = (uintptr_t)cv_img_to_be_sent.data;
        _input_data.input_image_width = cv_img_to_be_sent.cols;
        _input_data.input_image_height = cv_img_to_be_sent.rows;
        _input_data.input_image_format = KP_IMAGE_FORMAT_RGBA8888;

        _input_data.input_buf_size = _input_data.input_image_width * _input_data.input_image_height * 4;
        _input_data.input_ready_inf = true;
        pthread_mutex_unlock(&_mutex_image);
    }

EXIT_FREAD_IMAGE_THREAD:

    _blSendInfRunning = false;
    _blResultRunning = false;
    _blDisplayRunning = false;

    _blDispatchRunning = false;
    _blFifoqManagerRunning = false;

    return NULL;
}