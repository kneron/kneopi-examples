/*
 *******************************************************************************
 *  Copyright (c) 2010-2022 VATICS(KNERON) Inc. All rights reserved.
 *
 *  +-----------------------------------------------------------------+
 *  | THIS SOFTWARE IS FURNISHED UNDER A LICENSE AND MAY ONLY BE USED |
 *  | AND COPIED IN ACCORDANCE WITH THE TERMS AND CONDITIONS OF SUCH  |
 *  | A LICENSE AND WITH THE INCLUSION OF THE THIS COPY RIGHT NOTICE. |
 *  | THIS SOFTWARE OR ANY OTHER COPIES OF THIS SOFTWARE MAY NOT BE   |
 *  | PROVIDED OR OTHERWISE MADE AVAILABLE TO ANY OTHER PERSON. THE   |
 *  | OWNERSHIP AND TITLE OF THIS SOFTWARE IS NOT TRANSFERRED.        |
 *  |                                                                 |
 *  | THE INFORMATION IN THIS SOFTWARE IS SUBJECT TO CHANGE WITHOUT   |
 *  | ANY PRIOR NOTICE AND SHOULD NOT BE CONSTRUED AS A COMMITMENT BY |
 *  | VATICS(KNERON) INC.                                             |
 *  +-----------------------------------------------------------------+
 *
 *******************************************************************************
 */

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <getopt.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/resource.h>
#include <linux/videodev2.h>

#include <iniparser/iniparser.h>

extern "C" {
#include <vmf_nnm_inference_app.h>
#include <vmf_nnm_fifoq_manager.h>
}

#include "application_init.h"
#include "example_shared_struct.h"

//fifo queue buffer setting
#define IMAGE_BUFFER_COUNT      3
#define IMAGE_BUFFER_SIZE       1024 + (3840 * 2160 * 1.5) // header + YUV420

#define RESULT_BUFFER_COUNT     3
#define RESULT_BUFFER_SIZE      (1024 * 1024)


#define EXAMPLE_RTSP_CONFIG_PATH "./ini/example_rtsp.ini"

extern void *example_rtsp_input_thread(void *arg);
extern void *example_send_inf_thread(void *arg);
extern void *example_recv_result_thread(void *arg);
extern void *example_display_liveview_thread(void *arg);

bool _blDispatchRunning = true;
bool _blFifoqManagerRunning = true;
extern bool _blImageRunning;
extern bool _blSendInfRunning;
extern bool _blResultRunning;
extern bool _blDisplayRunning;

int loadConfig(const char* HostFfmpegConfigPath, EXAMPLE_RTSP_INIT_OPT_T* pExampleRtspInit)
{
    dictionary* ini = NULL;
    const char* tmp = NULL;
    struct stat info;
    char search_str[30] = {0};

    //! check file
    if (0 != stat(HostFfmpegConfigPath, &info))
        return -1;

    if (!(info.st_mode & S_IFREG))
        return -1;

    ini = iniparser_load(HostFfmpegConfigPath);

    pExampleRtspInit->pszModelPath = strdup(iniparser_getstring(ini, "nnm:ModelPath", "model.nef"));
    pExampleRtspInit->dwJobId = iniparser_getint(ini, "nnm:JobId", 11);

    pExampleRtspInit->pszRtspURL = strdup(iniparser_getstring(ini, "nnm:RtspURL", "rtsp://stream.strba.sk:1935/strba/VYHLAD_JAZERO.stream"));

    printf("[NNM] RTSP URL: %s\n", pExampleRtspInit->pszRtspURL);
    printf("[NNM] Model: %s dwJobId: %u\n", pExampleRtspInit->pszModelPath, pExampleRtspInit->dwJobId);
    iniparser_freedict(ini);

    return 0;
}

void sig_kill(int signo)
{
    _blDispatchRunning = false;
    _blFifoqManagerRunning = false;

    _blImageRunning = false;
    _blSendInfRunning = false;
    _blResultRunning = false;
    _blDisplayRunning = false;

    VMF_NNM_Fifoq_Manager_Wakeup();
}

void print_usage(char* argv[])
{
    printf("Usage 1, setting by ini: %s, default auto load [%s] \r\n", argv[0], EXAMPLE_RTSP_CONFIG_PATH);
}

char* parse_argument(int argc, char* argv[])
{
    int ch;
    char* pszConfigPath = NULL;

    while ((ch = getopt(argc, argv, "c:")) != -1) {
        switch (ch) {
            case 'c':
                if(pszConfigPath == NULL) {
                    pszConfigPath = strdup(optarg);
                    printf("pszHostFfmpegConfigPath = %s\n", pszConfigPath);
                }
                break;

            default:
                print_usage(argv);
        }
    }

    if (NULL == pszConfigPath ) {
        print_usage(argv);

        return NULL;
    }

    return pszConfigPath;
}

int main (int argc, char* argv[])
{
    uint32_t major, minor, patch, build;
    char* pszHostFfmpegConfigPath = NULL;

    VMF_NNM_Get_Version(&major, &minor, &patch, &build);

    printf("\n\n**********************************************************\n");
    printf("Kneron Firmware\n");
    printf("Ver. %d.%d.%d.%d\n", major, minor, patch, build);
    printf("Build Time: %s %s\n", __DATE__, __TIME__);
    printf("**********************************************************\n");

    pthread_t task_inf_data_handle; //VMF_NNM_Inference_Image_Dispatcher_Thread;
    pthread_t task_buf_mgr_handle;  //VMF_NNM_Fifoq_Manager_Enqueue_Image_Thread;

    pthread_t task_webcam_image_handle;
    pthread_t task_send_inf_handle;
    pthread_t task_recv_result_handle;
    pthread_t task_display_handle;

    int ret = KP_SUCCESS;
    EXAMPLE_RTSP_INIT_OPT_T ExampleRtspInit; //for host rtsp mode/thread

    memset(&ExampleRtspInit, 0, sizeof(EXAMPLE_RTSP_INIT_OPT_T));

    if (argc > 1) {
        pszHostFfmpegConfigPath = parse_argument(argc, argv);
        if(pszHostFfmpegConfigPath == NULL)   //parse argument
            goto EXIT;

        if(0 != loadConfig(pszHostFfmpegConfigPath, &ExampleRtspInit))                  // load ini file
            goto EXIT;

    } else {
        if(0 != loadConfig(EXAMPLE_RTSP_CONFIG_PATH, &ExampleRtspInit))                  // load ini file
            goto EXIT;
    }

    //! register signal
    signal(SIGTERM, sig_kill);
    signal(SIGKILL, sig_kill);
    signal(SIGINT, sig_kill);

    //SIGSEGV
    struct sigaction sa;
    memset(&sa, 0, sizeof(struct sigaction));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler  = sig_kill;
    sa.sa_flags = SA_SIGINFO|SA_RESETHAND;  // Reset signal handler to system default after signal triggered
    sigaction(SIGSEGV, &sa, NULL);

    printf("[%s] app_initialize \n", __func__);
    app_initialize();

    VMF_NNM_Load_Model_From_File(ExampleRtspInit.pszModelPath);

    VMF_NNM_Fifoq_Manager_Allocate_Buffer(IMAGE_BUFFER_COUNT, IMAGE_BUFFER_SIZE, RESULT_BUFFER_COUNT, RESULT_BUFFER_SIZE);

    pthread_create(&task_webcam_image_handle, NULL, example_rtsp_input_thread, &ExampleRtspInit);
    pthread_create(&task_send_inf_handle, NULL, example_send_inf_thread, &ExampleRtspInit.dwJobId);
    pthread_create(&task_recv_result_handle, NULL, example_recv_result_thread, NULL);
    pthread_create(&task_display_handle, NULL, example_display_liveview_thread, NULL);

    pthread_create(&task_buf_mgr_handle, NULL, VMF_NNM_Fifoq_Manager_Enqueue_Image_Thread, &_blFifoqManagerRunning);
    pthread_create(&task_inf_data_handle, NULL, VMF_NNM_Inference_Image_Dispatcher_Thread, &_blDispatchRunning);

    pthread_join(task_webcam_image_handle, NULL);
    pthread_join(task_send_inf_handle, NULL);
    pthread_join(task_recv_result_handle, NULL);
    pthread_join(task_display_handle, NULL);

    pthread_join(task_buf_mgr_handle, NULL);
    pthread_join(task_inf_data_handle, NULL);

    app_destroy();
    VMF_NNM_Fifoq_Manager_Release_All_Buffer();
    ret = 0;

EXIT:
    printf("%s free\n", __func__);

    if(pszHostFfmpegConfigPath) {
        free(pszHostFfmpegConfigPath);
        pszHostFfmpegConfigPath = NULL;
    }

    if (ExampleRtspInit.pszModelPath) {
        free(ExampleRtspInit.pszModelPath);
        ExampleRtspInit.pszModelPath = NULL;
    }

    if (ExampleRtspInit.pszRtspURL) {
        free(ExampleRtspInit.pszRtspURL);
        ExampleRtspInit.pszRtspURL = NULL;
    }

    printf("%s end\n", __func__);
    return ret;
}
