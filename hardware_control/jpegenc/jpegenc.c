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
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <iniparser.h>
#include <mem_broker.h>
#include <video_buf.h>
#include <video_snapshot_mechanism.h>
#include <sync_shared_memory.h>


char *g_InputPath = NULL;
char *g_OutputPath = NULL;
unsigned int g_dwImgWidth = 0;
unsigned int g_dwImgHeight = 0;
unsigned int g_dwImgQP = 0;

static void print_usage(const char *name)
{
    if (name) {
        printf("Usage:\t %s \n"
                "\t-c ConfigFile\n"
                "\t-h this help\n" , name);
    }
}

static int loadConfig(char *configPath)
{
    if (!configPath) {
        printf("%s failed \n", __func__);
        return -1;
    }

    dictionary *ini;
    const char *tmp = NULL;
    ini = iniparser_load(configPath);
    if (ini == NULL) {
        printf("can't parse file: %s\n", configPath);
        return -1;
    }

    int ret = -1;

    do {
        if ((tmp = iniparser_getstring(ini, "config:inputpath", NULL)) == NULL) {
            printf("No inputpath\n");
            break;
        }

        if(tmp)
            g_InputPath = strdup(tmp);

        if ((tmp = iniparser_getstring(ini, "config:outputpath", NULL)) == NULL) {
            printf("No outputpath\n");
            break;
        }

        if(tmp)
            g_OutputPath = strdup(tmp);

        g_dwImgWidth = iniparser_getint(ini, "config:img_width", 0);
        g_dwImgHeight = iniparser_getint(ini, "config:img_height", 0);
        g_dwImgQP = iniparser_getint(ini, "config:img_qp", 100);

        ret = 0;
    } while(0);

    iniparser_freedict(ini);
    return ret;
}

int main (int argc, char **argv)
{
    FILE *pfOutput = NULL;
    FILE *pfInput = NULL;
    SSM_BUFFER_T tSsmBuff;
    VMF_SNAP_CROP_PARAMS_T tCropInfo;
    VMF_VSRC_SSM_OUTPUT_INFO_T tVsrcSsmInfo;

    int opt;
    char *configPath = NULL;
    unsigned char *pbyOutBuf = NULL;
    unsigned char *pbyInbuf = NULL;
    unsigned int dwReadCount;
    unsigned int dwOutputBufSize = 0;
    unsigned int dwYSize = 0;
    unsigned int dwInBufSize = 0;
    unsigned int dwYOffset = 0;
    int ret;

    while (-1 != (opt = getopt(argc, argv, "c:h:"))) {
        switch(opt)
        {
        case 'c':
            configPath = strdup(optarg);
            break;
        case 'h':
        default:
            print_usage(argv[0]);
            exit(1);
        }
    }


    if(0 != loadConfig(configPath) || (g_dwImgWidth == 0)|| (g_dwImgHeight == 0) || !g_InputPath || !g_OutputPath) {
        print_usage(argv[0]);
        return -1;
    }

    memset(&tSsmBuff, 0, sizeof(SSM_BUFFER_T));
    memset(&tCropInfo, 0, sizeof(VMF_SNAP_CROP_PARAMS_T));
    memset(&tVsrcSsmInfo, 0, sizeof(VMF_VSRC_SSM_OUTPUT_INFO_T));

    if ((pfInput=fopen(g_InputPath, "rb")) == NULL) {
        printf("Open input bitstream file fail !!\n");
        goto EXIT;
    }

    //out buffer    
    unsigned int dwImgWStride = VMF_16_ALIGN(g_dwImgWidth);
    unsigned int dwImgHStride = VMF_16_ALIGN(g_dwImgHeight);
    
    dwYSize = dwOutputBufSize = g_dwImgWidth * g_dwImgHeight;    
    dwYOffset = dwImgWStride * dwImgHStride;
    dwInBufSize = (dwYOffset * 3 >> 1) + VMF_MAX_SSM_HEADER_SIZE;
    
    pbyOutBuf = (unsigned char *)MemBroker_GetMemory(dwOutputBufSize , VMF_ALIGN_TYPE_DEFAULT);
    if (!pbyOutBuf) {
        printf("Allocate out buffer fail !! Size = %d\n", dwOutputBufSize);
        goto EXIT;
    }
    memset(pbyOutBuf, 0, dwOutputBufSize);
    
    pbyInbuf = (unsigned char *)MemBroker_GetMemory(dwInBufSize, VMF_ALIGN_TYPE_DEFAULT);
    if (!pbyInbuf) {
        printf("Allocate bit stream buffer fail \n");
        goto EXIT;
    }
    memset(pbyInbuf, 0, dwInBufSize);
    
    if(dwImgWStride == g_dwImgWidth && dwImgHStride == g_dwImgHeight) {
        unsigned int dwYuvSize = dwYSize*3/2;
		dwReadCount = fread(pbyInbuf + VMF_MAX_SSM_HEADER_SIZE, sizeof(unsigned char), dwYuvSize, pfInput);
        if (dwReadCount != dwYuvSize) {
            printf("Read yuv file failed, read total %d, should be %d.\n", dwReadCount, dwYuvSize);
            goto EXIT;
        } else
            printf("Read yuv total size of %d\n", dwReadCount);
	} else {
        //! Y
        for (unsigned int i = 0; i < g_dwImgHeight; i++) {
            fread(pbyInbuf + VMF_MAX_SSM_HEADER_SIZE + i*dwImgWStride, g_dwImgWidth, 1, pfInput);	
        }
        //! U
        for (unsigned int i = 0; i < g_dwImgHeight>>1; i++) {
            fread((char*)pbyInbuf + VMF_MAX_SSM_HEADER_SIZE + dwYOffset + (i*dwImgWStride>>1), g_dwImgWidth>>1, 1, pfInput);
	    }	
	    //! V
        for (unsigned int i = 0; i < g_dwImgHeight>>1; i++) {
            fread((char*)pbyInbuf + VMF_MAX_SSM_HEADER_SIZE + dwYOffset + (dwYOffset>>2) + (i*dwImgWStride>>1), g_dwImgWidth>>1, 1, pfInput);
        }
	}
  
    MemBroker_CacheFlush(pbyInbuf, dwInBufSize);

    tSsmBuff.buffer = pbyInbuf;
    tSsmBuff.buffer_phys_addr = MemBroker_GetPhysAddr(pbyInbuf);
    tVsrcSsmInfo.dwYStride = dwImgWStride;
    tVsrcSsmInfo.dwYSize = dwYOffset;
    tVsrcSsmInfo.dwUVSize = dwYOffset >> 2;
    tVsrcSsmInfo.dwOffset[0] = VMF_MAX_SSM_HEADER_SIZE;
    tVsrcSsmInfo.dwOffset[1] = tVsrcSsmInfo.dwOffset[0] + dwYOffset;
    tVsrcSsmInfo.dwOffset[2] = tVsrcSsmInfo.dwOffset[1] + (dwYOffset >> 2);
    tVsrcSsmInfo.dwWidth = g_dwImgWidth;
    tVsrcSsmInfo.dwHeight = g_dwImgHeight;
    tVsrcSsmInfo.dwType = 0;
    VMF_VSRC_SSM_SetInfo(tSsmBuff.buffer, &tVsrcSsmInfo);

    tCropInfo.dwStartX = 0;
    tCropInfo.dwStartY = 0;
    tCropInfo.dwCropWidth = g_dwImgWidth;
    tCropInfo.dwCropHeight = g_dwImgHeight;
    tCropInfo.dwBufSize = dwOutputBufSize;
    tCropInfo.pOutBuffer = (void *)pbyOutBuf;
    tCropInfo.dwQp = g_dwImgQP;

    ret = VMF_SNAP_ProcessOneFrame_YUV(&tSsmBuff, &tCropInfo);
    if (ret == -1) {
        printf("Jpeg encode failed\n");
        goto EXIT;
    }

    if ((pfOutput=fopen(g_OutputPath, "wb")) == NULL) {
        printf("Open output YUV file fail !!\n");
        goto EXIT;
    }

    printf("Jpeg Encode done, size is %d, (buffer size is %d)\n",ret , dwOutputBufSize);
    fwrite(tCropInfo.pOutBuffer, sizeof(unsigned char), ret, pfOutput);

EXIT:


    if (pbyOutBuf)
        MemBroker_FreeMemory(pbyOutBuf);

    if (pbyInbuf) {
        MemBroker_FreeMemory(pbyInbuf);
    }

    if (pfOutput) {
        fclose(pfOutput);
    }

    if (pfInput) {
        fclose(pfInput);
    }

    if (configPath)
        free(configPath);

    if (g_InputPath)
        free(g_InputPath);

    if (g_OutputPath)
        free(g_OutputPath);

    return 0;
}