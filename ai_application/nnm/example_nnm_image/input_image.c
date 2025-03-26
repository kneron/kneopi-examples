#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>
#include <signal.h>
#include <pthread.h>
#include <sys/time.h>

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

unsigned int _loop_time = 0;

kp_image_format_t string_to_kp_image_format(char *image_format_string)
{
    kp_image_format_t kp_format = KP_IMAGE_FORMAT_UNKNOWN;

    if (NULL == image_format_string)
        goto FUNC_OUT;

    if (0 == strcmp("RGB565", image_format_string)) {
        kp_format = KP_IMAGE_FORMAT_RGB565;
    }
    else if (0 == strcmp("YUYV", image_format_string)) {
        kp_format = KP_IMAGE_FORMAT_YUYV;
    }
    else if (0 == strcmp("YUV420", image_format_string)) {
        kp_format = KP_IMAGE_FORMAT_YUV420;
    }
    else if (0 == strcmp("RAW8", image_format_string)) {
        kp_format = KP_IMAGE_FORMAT_RAW8;
    }
    else if (0 == strcmp("RGBA8888", image_format_string)) {
        kp_format = KP_IMAGE_FORMAT_RGBA8888;
    }
    else {
        kp_format = KP_IMAGE_FORMAT_UNKNOWN;
    }

FUNC_OUT:

    return kp_format;
}

static unsigned char* read_file_to_raw_buffer(void *arg, kp_image_format_t *p_image_format, uint32_t* p_image_buffer_size)
{
    EXAMPLE_IMAGE_INIT_OPT_T *pInitOpt = (EXAMPLE_IMAGE_INIT_OPT_T *)arg;
    struct stat tInputState;
    unsigned char* image_buffer = NULL; //image buffer index

    int ret = 0;
    uint32_t image_buffer_size = 0;

    *p_image_format = string_to_kp_image_format(pInitOpt->pszImageFormat);

    if((KP_IMAGE_FORMAT_RGB565 == *p_image_format) || (KP_IMAGE_FORMAT_YUYV == *p_image_format)) {
        image_buffer_size = pInitOpt->dwImageWidth * pInitOpt->dwImageHeight * 2;
    }
    else if(KP_IMAGE_FORMAT_YUV420 == *p_image_format) {
        image_buffer_size = pInitOpt->dwImageWidth * pInitOpt->dwImageHeight * 1.5;
    }
    else if(KP_IMAGE_FORMAT_RAW8 == *p_image_format) {
        image_buffer_size = pInitOpt->dwImageWidth * pInitOpt->dwImageHeight;
    }
    else if (KP_IMAGE_FORMAT_RGBA8888 == *p_image_format) {
        image_buffer_size = pInitOpt->dwImageWidth * pInitOpt->dwImageHeight * 4;
    }


    printf("fopen %s \n", pInitOpt->pszImageName);
    FILE *pFile = fopen(pInitOpt->pszImageName, "rb");

    if (pFile == NULL) {
        printf("%s() fopen failed.\n", __func__);
        goto FUNC_OUT;
    }

    ret = fstat(fileno(pFile), &tInputState);

    if (ret) {
        printf("%s() fstat failed.\n", __func__);
        goto FUNC_OUT;
    }

    if (tInputState.st_size != (int)image_buffer_size) {
        printf("%s() image format mismatch: file size: %d, buf size by image width, height: %d\n", __func__, (int)tInputState.st_size, image_buffer_size);
        goto FUNC_OUT;
    }

    image_buffer = (unsigned char *)malloc(sizeof(unsigned char) * image_buffer_size);
    if(!image_buffer) {
        printf("%s() image_buffer failed.\n", __func__);
        goto FUNC_OUT;
    }
    if(fread((void *)image_buffer, image_buffer_size, 1, pFile) != 1) {
        printf("[%s] File read size is not equal to file size\n",  pInitOpt->pszImageName);
        goto FUNC_OUT;
    }
    printf("image_buffer addr = 0x%p \n", image_buffer);

    *p_image_buffer_size = image_buffer_size;

FUNC_OUT:
    if(pFile)
        fclose(pFile);

    return image_buffer;
}

void *example_fread_image_thread(void *arg)
{
    EXAMPLE_IMAGE_INIT_OPT_T *pInitOpt = (EXAMPLE_IMAGE_INIT_OPT_T *)arg;

    struct timeval tGetTime1, tGetTime2;
    double fpsTime = 1000000 / pInitOpt->fFps;
    unsigned int ImgSendCount = 0;
    float spend = 0.0, sleepTime = 0.0;

    uint32_t image_buffer_size = 0;
    kp_image_format_t image_format;
    unsigned char* image_buffer = NULL;
    int image_pixel_size = 0;

    _loop_time = pInitOpt->dwLoopTime;
    image_buffer = (unsigned char *)read_file_to_raw_buffer(pInitOpt, &image_format, &image_buffer_size);

    while (true == _blImageRunning)
    {
        gettimeofday(&tGetTime1, NULL);

        pthread_mutex_lock(&_mutex_image);
        _input_data.input_buf_address = (uintptr_t)image_buffer;
        _input_data.input_image_width = pInitOpt->dwImageWidth;
        _input_data.input_image_height = pInitOpt->dwImageHeight;
        _input_data.input_image_format = image_format;

        switch(image_format) {
        case KP_IMAGE_FORMAT_RAW8:
            image_pixel_size = 1;
            break;
        case KP_IMAGE_FORMAT_RGBA8888:
            image_pixel_size = 4;
            break;
        case KP_IMAGE_FORMAT_RGB565:
        case KP_IMAGE_FORMAT_YUYV:
            image_pixel_size = 2;
            break;
        case KP_IMAGE_FORMAT_YUV420:
            image_pixel_size = 1.5;
            break;
        default:
            image_pixel_size = 0;
            break;
        }

        _input_data.input_buf_size = _input_data.input_image_width * _input_data.input_image_height * image_pixel_size;
        _input_data.input_ready_inf = true;

        pthread_mutex_unlock(&_mutex_image);

        gettimeofday(&tGetTime2, NULL);
        spend = tGetTime2.tv_sec * 1000000 + tGetTime2.tv_usec - tGetTime1.tv_sec * 1000000 - tGetTime1.tv_usec;
        sleepTime = fpsTime - spend;
        if((sleepTime < 0.0) || (sleepTime > 1000000))
            sleepTime = 0.0;

        usleep(sleepTime);

        ImgSendCount++;
        if (_loop_time == ImgSendCount) {
            _blImageRunning = false;
            usleep(3000 * 1000);
        }
    }

EXIT_FREAD_IMAGE_THREAD:

    if(NULL != image_buffer)
        free(image_buffer);

    _blSendInfRunning = false;
    _blResultRunning = false;
    _blDisplayRunning = false;

    _blDispatchRunning = false;
    _blFifoqManagerRunning = false;

    return NULL;
}
