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

#include <iniparser/iniparser.h>

#include <vmf_nnm_inference_app.h>
#include <vmf_nnm_fifoq_manager.h>

#include "application_init.h"
#include "example_shared_struct.h"

#define IMAGE_BUFFER_COUNT      3
#define IMAGE_BUFFER_SIZE       (2 * 1920 * 1080 + 1024)
#define RESULT_BUFFER_COUNT     3
#define RESULT_BUFFER_SIZE      (1024 * 1024)

#define EXAMPLE_IMAGE_CONFIG_PATH "./ini/example_image.ini"

extern void *example_fread_image_thread(void *arg);
extern void *example_send_inf_thread(void *arg);
extern void *example_recv_result_thread(void *arg);
extern void *example_display_log_thread(void *arg);

bool _blDispatchRunning = true;
bool _blFifoqManagerRunning = true;
extern bool _blImageRunning;
extern bool _blSendInfRunning;
extern bool _blResultRunning;
extern bool _blDisplayRunning;


int loadConfig(char* HostVerifyConfigPath, EXAMPLE_IMAGE_INIT_OPT_T* pExampleImageInit)
{
	dictionary* ini = NULL;
	struct stat info;

	//! check file
	if (0 != stat(HostVerifyConfigPath, &info))
		return -1;

	if (!(info.st_mode & S_IFREG))
		return -1;

	printf("iniparser_load %s \n", HostVerifyConfigPath);
	ini = iniparser_load(HostVerifyConfigPath);


	pExampleImageInit->pszModelPath = strdup(iniparser_getstring(ini, "nnm:ModelPath", "model.nef"));
	pExampleImageInit->dwImageWidth = iniparser_getint(ini, "nnm:ImageWidth", 640);
	pExampleImageInit->dwImageHeight = iniparser_getint(ini, "nnm:ImageHeight", 640);
	pExampleImageInit->dwJobId = iniparser_getint(ini, "nnm:JobId", 11);
	pExampleImageInit->fFps = (float)iniparser_getdouble(ini, "nnm:Fps", 1);
    if (pExampleImageInit->fFps == 0) {
        pExampleImageInit->fFps = 1;
        printf("[Fps = 0], set a default fps %f.\n", pExampleImageInit->fFps);
    }
    pExampleImageInit->pszImageName = strdup(iniparser_getstring(ini, "nnm:ImageName", "a_man_640x480.yuv"));
    pExampleImageInit->pszImageFormat = strdup(iniparser_getstring(ini, "nnm:ImageFormat", "YUV420"));

    pExampleImageInit->dwLoopTime = iniparser_getint(ini, "nnm:InfLoopTime", 10);
    //eGetImageBufMode = iniparser_getint(ini, "nnm:GetImageBufMode", 0);
    //eNniProcessMode = iniparser_getint(ini, "nnm:NniProcessMode", 0);
    if ((0 == strcmp("RGB565", pExampleImageInit->pszImageFormat)) ||
       (0 == strcmp("YUYV", pExampleImageInit->pszImageFormat))   ||
       (0 == strcmp("RAW8", pExampleImageInit->pszImageFormat))   ||
       (0 == strcmp("YUV420", pExampleImageInit->pszImageFormat)))
    {
        printf("[%s], image format = %s \n", __func__, pExampleImageInit->pszImageFormat);
    }
    else
        printf("[%s], image format not support \n", __func__);

    printf("[NNM] Model: %s pszImageName: %s \n", pExampleImageInit->pszModelPath, pExampleImageInit->pszImageName);
	printf("[NNM] Model: %s ImageWidth: %d ImageHeight: %d Fps: %f \n", pExampleImageInit->pszModelPath, pExampleImageInit->dwImageWidth, pExampleImageInit->dwImageHeight, pExampleImageInit->fFps);
	printf("[NNM] Model: %s dwJobId: %d \n", pExampleImageInit->pszModelPath, pExampleImageInit->dwJobId);
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
}

void print_usage(char* argv[])
{
    printf("Usage: %s\n" \
           "                -c <config_file_path>\n\n", argv[0]);
    printf("Example: %s -c ./ini/example_image.ini\n", argv[0]);
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
                    printf("pszHostVerifyConfigPath = %s\n", pszConfigPath);
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
    char* pszHostVerifyConfigPath = NULL;

    VMF_NNM_Get_Version(&major, &minor, &patch, &build);

    printf("\n\n**********************************************************\n");
    printf("Kneron Firmware\n");
    printf("Ver. %d.%d.%d.%d\n", major, minor, patch, build);
    printf("Build Time: %s %s\n", __DATE__, __TIME__);
    printf("**********************************************************\n");

    pthread_t task_inf_data_handle; //VMF_NNM_Inference_Image_Dispatcher_Thread;
    pthread_t task_buf_mgr_handle;  //VMF_NNM_Fifoq_Manager_Enqueue_Image_Thread;

    pthread_t task_fread_image_handle;
    pthread_t task_send_inf_handle;
    pthread_t task_recv_result_handle;
    pthread_t task_display_handle;

    int ret = KP_SUCCESS;
    EXAMPLE_IMAGE_INIT_OPT_T ExampleImageInit; //for host verify mode/thread

    memset(&ExampleImageInit, 0, sizeof(EXAMPLE_IMAGE_INIT_OPT_T));

    if (argc > 1) {
        pszHostVerifyConfigPath = parse_argument(argc, argv);
        if(pszHostVerifyConfigPath == NULL)   //parse argument
            goto EXIT;

        if(0 != loadConfig(pszHostVerifyConfigPath, &ExampleImageInit))                  // load ini file
            goto EXIT;

    }
    else {
        if(0 != loadConfig(EXAMPLE_IMAGE_CONFIG_PATH, &ExampleImageInit))                // load ini file
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
    app_initialize();   //VMF_NNM_Inference_App_Init(_app_func);

    VMF_NNM_Load_Model_From_File(ExampleImageInit.pszModelPath);

    VMF_NNM_Fifoq_Manager_Allocate_Buffer(IMAGE_BUFFER_COUNT, IMAGE_BUFFER_SIZE, RESULT_BUFFER_COUNT, RESULT_BUFFER_SIZE);

    pthread_create(&task_fread_image_handle, NULL, example_fread_image_thread, &ExampleImageInit);
    pthread_create(&task_send_inf_handle, NULL, example_send_inf_thread, &ExampleImageInit.dwJobId);
    pthread_create(&task_recv_result_handle, NULL, example_recv_result_thread, NULL);
    pthread_create(&task_display_handle, NULL, example_display_log_thread, NULL);

    pthread_create(&task_buf_mgr_handle, NULL, VMF_NNM_Fifoq_Manager_Enqueue_Image_Thread, &_blFifoqManagerRunning);
    pthread_create(&task_inf_data_handle, NULL, VMF_NNM_Inference_Image_Dispatcher_Thread, &_blDispatchRunning);

    pthread_join(task_fread_image_handle, NULL);
    VMF_NNM_Fifoq_Manager_Wakeup();

    pthread_join(task_send_inf_handle, NULL);
    pthread_join(task_recv_result_handle, NULL);
    pthread_join(task_display_handle, NULL);

    pthread_join(task_buf_mgr_handle, NULL);
    pthread_join(task_inf_data_handle, NULL);

    app_destroy();  //VMF_NNM_Inference_App_Destroy();
    VMF_NNM_Fifoq_Manager_Release_All_Buffer();
EXIT:

    if (ExampleImageInit.pszModelPath)
        free(ExampleImageInit.pszModelPath);

    if (ExampleImageInit.pszImageName)
        free(ExampleImageInit.pszImageName);

    if (ExampleImageInit.pszImageFormat)
        free(ExampleImageInit.pszImageFormat);

    if (pszHostVerifyConfigPath)
        free(pszHostVerifyConfigPath);

    printf("%s end\n", __func__);
    return ret;
}
