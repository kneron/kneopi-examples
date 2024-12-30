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
#include <video_decoder.h>
#define GB_BUFFER_SIZE		(1024*1024*16) // power of 2
#define MAX_WIDTH			3840 //1920
#define MAX_HEIGHT			3840 //1088
unsigned int g_dwOutFrameNum = 0;
char *g_InputPath = NULL;
char *g_OutputPath = NULL;

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

	if ((tmp = iniparser_getstring(ini, "config:inputpath", NULL)) == NULL) {
		printf("No inputpath\n");
		return -1;
	}

	if(tmp)
		g_InputPath = strdup(tmp);

	if ((tmp = iniparser_getstring(ini, "config:outputpath", NULL)) == NULL) {
		printf("No outputpath\n");
		return -1;
	}

	if(tmp)
		g_OutputPath = strdup(tmp);

	g_dwOutFrameNum = iniparser_getint(ini, "config:framenum", -1);
	iniparser_freedict(ini);
	return 0;
}

int main (int argc, char **argv) 
{
	VMF_VDEC_HANDLE_T *ptVideoDecoder = NULL;
	VMF_JDEC_STATE_T  *ptVideoState = NULL;
	FILE *pfOutput = NULL;
	FILE *pfInput = NULL;
	VMF_VIDEO_BUF_T tFrameInfo;
	int opt;
	char *configPath = NULL;
	unsigned int dwFilePosition;
	unsigned int i = 0;
	unsigned char *pbyFrame[3] = {0}; 
	unsigned char *pbyNetbuf = NULL;
	unsigned int dwFrameSize = MAX_WIDTH * MAX_HEIGHT;
	unsigned int dwFrameCount;
	unsigned int dwReadCount;

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

	if(0 != loadConfig(configPath) || (g_dwOutFrameNum == 0) || !g_InputPath || !g_OutputPath) {
		print_usage(argv[0]);
	 	return -1;
	}
   
    memset(&tFrameInfo, 0, sizeof(VMF_VIDEO_BUF_T));
    if ((pfInput=fopen(g_InputPath, "rb")) == NULL) {
		printf("Open input bitstream file fail !!\n");
		goto EXIT;
	}


	if ((pfOutput=fopen(g_OutputPath, "wb")) == NULL) {
		printf("Open output YUV file fail !!\n");
		goto EXIT;
	}

	//out buffer 
	for (i = 0; i < 3; i++) {
		pbyFrame[i] = (unsigned char *)MemBroker_GetMemory(dwFrameSize , VMF_ALIGN_TYPE_8_BYTE);
		if (!pbyFrame[i]) {
			printf("Allocate out frame buffer fail !! Size = %d\n", dwFrameSize);
			goto EXIT;
		} 
	}

	pbyNetbuf = (unsigned char *)MemBroker_GetMemory(GB_BUFFER_SIZE, VMF_ALIGN_TYPE_8_BYTE);
	if (!pbyNetbuf) {
		printf("Allocate bit stream buffer fail \n");
		goto EXIT;
	}

	//frame buffer info
    tFrameInfo.apbyVirtAddr[0] = pbyFrame[0];
    tFrameInfo.apbyVirtAddr[1] = pbyFrame[1];
    tFrameInfo.apbyVirtAddr[2] = pbyFrame[2];
    tFrameInfo.apbyPhysAddr[0] = (unsigned char *)MemBroker_GetPhysAddr((unsigned char *)tFrameInfo.apbyVirtAddr[0]); 
    tFrameInfo.apbyPhysAddr[1] = (unsigned char *)MemBroker_GetPhysAddr((unsigned char *)tFrameInfo.apbyVirtAddr[1]); 
    tFrameInfo.apbyPhysAddr[2] = (unsigned char *)MemBroker_GetPhysAddr((unsigned char *)tFrameInfo.apbyVirtAddr[2]);
    tFrameInfo.adwHWInfo[1] = tFrameInfo.adwHWInfo[2] = 0;
	VMF_VDEC_INITOPT_T vdec_opt;
    vdec_opt.eCodecType = VMF_VDEC_CODEC_TYPE_JPEG;
	vdec_opt.dwMaxWidth = MAX_WIDTH;
	vdec_opt.dwMaxHeight = MAX_HEIGHT;
	ptVideoDecoder = VMF_VDEC_Init(&vdec_opt);
	if(!ptVideoDecoder) {
		printf("jpeg decoder init failed \n");
		goto EXIT;
	}

	dwFilePosition = 0;
	
	for (dwFrameCount = 0; dwFrameCount < g_dwOutFrameNum; dwFrameCount++)
	{
		if (fseek(pfInput, dwFilePosition, 0) < 0) {
			printf("fseek failed \n");
			goto EXIT;
		}
		dwReadCount = fread(pbyNetbuf, sizeof(unsigned char), GB_BUFFER_SIZE, pfInput);
		if (dwReadCount == 0) {
			goto EXIT;
		}

		ptVideoState = VMF_VDEC_GetState(ptVideoDecoder);
		ptVideoState->pbyInBuf = (unsigned char*)pbyNetbuf; 
		ptVideoState->dwInBufSize = dwReadCount;
		ptVideoState->ptOutBuf = &tFrameInfo;
		if (0 != VMF_VDEC_ProcessOneFrame(ptVideoDecoder)) {
			printf("something wrong \n");	
			goto EXIT;
		}

		dwFilePosition = dwFilePosition + ptVideoState->dwDecSize;
		dwFrameSize = ptVideoState->dwWinWidth * ptVideoState->dwWinHeight;
		//Y
	    fwrite(ptVideoState->ptOutBuf->apbyVirtAddr[0], sizeof(unsigned char), dwFrameSize, pfOutput); 
	    //U
	    fwrite(ptVideoState->ptOutBuf->apbyVirtAddr[1], sizeof(unsigned char), (dwFrameSize>>2), pfOutput); 
	    //V
	    fwrite(ptVideoState->ptOutBuf->apbyVirtAddr[2], sizeof(unsigned char), (dwFrameSize>>2), pfOutput); 
		
	}

EXIT:

	for (i = 0; i < 3; i++) {
		if (pbyFrame[i]) 
			MemBroker_FreeMemory(pbyFrame[i]);
	}

	if (pbyNetbuf) {
		MemBroker_FreeMemory(pbyNetbuf);
	}

	if (pfOutput) {
		fclose(pfOutput);
	}
	
	if (pfInput) {
		fclose(pfInput);
	}

	if (ptVideoDecoder)
		VMF_VDEC_Release(ptVideoDecoder);

	if (configPath) 
		free(configPath);

	if (g_InputPath)
		free(g_InputPath);

	if (g_OutputPath)
		free(g_OutputPath);

	return 0;
}