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
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <signal.h>
#include "video_buf.h"
#include "ssm_info.h"
#include "sync_shared_memory.h"

static int g_bTerminate = 0;
static char *g_szPin = NULL;
static char *g_szOutput = NULL;
static unsigned int g_dwSaveNum = 1;
static SSM_HANDLE_T* g_ptSsmHandle = NULL;

static void signal_handler(int signo)
{
    (void) signo;
    g_bTerminate = 1;

    if (g_ptSsmHandle)
        SSM_Reader_Wakeup(g_ptSsmHandle);
}

static void *ssm_reader(void *param)
{
    (void)param;

    SSM_BUFFER_T tReaderSsmBuffer;
    VMF_VSRC_SSM_OUTPUT_INFO_T tSsmInfo;
    FILE* fp = NULL;

    fp = fopen(g_szOutput, "wb");
    if (!fp) {
        printf("open write file failed\n");
        return NULL;
    }

    g_ptSsmHandle = SSM_Reader_Init(g_szPin);

    if (!g_ptSsmHandle) {
        printf("SSM_Reader_Init failed\n");
        fclose(fp);
        return NULL;
    }

    memset(&tReaderSsmBuffer, 0, sizeof(SSM_BUFFER_T));

    while(!g_bTerminate) {
        if (g_dwSaveNum == 0)
            break;

        if (SSM_Reader_ReturnReceiveBuff(g_ptSsmHandle, &tReaderSsmBuffer)) {
            printf("[%s:%d] SSM_Reader_ReturnReceiveBuff Fail!\n", __func__, __LINE__);
            break;
        }
        VMF_VSRC_SSM_GetInfo(tReaderSsmBuffer.buffer, &tSsmInfo);

        fwrite(tReaderSsmBuffer.buffer + tSsmInfo.dwOffset[0] , 1, tSsmInfo.dwYSize ,fp);
        if (0 < tSsmInfo.dwUVSize) {
            fwrite(tReaderSsmBuffer.buffer + tSsmInfo.dwOffset[1] , 1, tSsmInfo.dwUVSize ,fp);
            fwrite(tReaderSsmBuffer.buffer + tSsmInfo.dwOffset[2] , 1, tSsmInfo.dwUVSize ,fp);
        }
        fflush(fp);

        printf("[%s] index %d buffer %p, YUV420 planar data frame\n",__FUNCTION__, tReaderSsmBuffer.idx, (void*)tReaderSsmBuffer.buffer_phys_addr);
        g_dwSaveNum -= 1;
    }

    printf("[%s] Release\n",__FUNCTION__);

    SSM_Reader_ReturnBuff(g_ptSsmHandle, &tReaderSsmBuffer);
    SSM_Release(g_ptSsmHandle);
    g_ptSsmHandle = NULL;
    fclose(fp);

    return NULL;
}

int main(int argc, char **argv)
{
    int ch = 0;
    pthread_t tNormalPid;

    signal(SIGTERM, signal_handler);
    signal(SIGINT, signal_handler);

    while ((ch = getopt(argc, argv, "p:n:o:")) != -1) {
        switch (ch) {
            case 'p':
                g_szPin = strdup(optarg);
                break;
            case 'n':
                g_dwSaveNum = atoi(optarg);
                break;
            case 'o':
                g_szOutput = strdup(optarg);
                break;
            default:
                printf("Usage: %s [-p ssm_pin] [-o output file name] [-n number (default is 1)]\n", argv[0]);
                printf("example: \r\n");
                printf("\t %s -p vsrc_ssm_0_640_480 -o ssm_reader.yuv \n", argv[0]);
                return -1;
        }
    }

    if ((g_szPin == NULL) || (g_szOutput == NULL)) {
        printf("[Error] Need ssm pin and output file name.\n.");
        return -1;
    }

    pthread_create(&tNormalPid, NULL, ssm_reader, NULL);
    pthread_join(tNormalPid, NULL);

    if (g_szPin)
        free(g_szPin);

    if (g_szOutput)
        free(g_szOutput);
    return 0;
}