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
#include <string.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <getopt.h>
#include <unistd.h>
#include <stdarg.h>
#include <sys/time.h>

#include "mem_broker.h"
#include "msg_broker.h"
#include "sync_shared_memory.h"
#include "video_encoder_output_srb.h"
#include "video_encoder_output_scm.h"
#include "video_source.h"
#include "video_encoder.h"
#include "video_bind.h"
#include "frame_info.h"
#include "vmf_log.h"

#define MODULE_NAME	"venc1"
#define VENC_OUTPUT_BUF_NUM  3
#define VENC_VSRC_PIN     "vsrc_ssm"					//! VMF_VSRC Output pin
#define VENC_OUTPUT_PIN   "venc_srb_1"					//! VMF_VideoEnc Output pin
#define VENC_OUTPUT_PIN_RS   "venc_srb_2"					//! VMF_VideoEnc Output pin
#define VENC_CMD_FIFO     "/tmp/venc/c0/command.fifo" 	//! communicate with rtsps, vrec, etc.
#define VENC_RESOURCE_DIR  "./Resource/"                //! directory contains ISP, AE, AWB, AutoScene sub directory
#define VENC_ENCODE_BUF_SIZE (4*1024*1024)	            //! 4*1024*1024
#define VENC_ENCODE_BUF_SIZE_RS (1*1024*1024)	            //! 4*1024*1024
#define RESIZE_WIDTH 640
#define RESIZE_HEIGHT 480
#define release_string(X) do { if (X) free(X); } while (0)
#define METADATA_PIN      "mdat_srb"
#define METADATA_BUF_SIZE 1024

typedef int (*VMF_VENC_OUT_SETUP_FUNC)(VMF_VENC_CONFIG_T *, VMF_VENC_CODEC_TYPE, void *, void *);
typedef int (*VMF_VENC_OUT_RELEASE_FUNC)(void *);

VMF_LAYOUT_T g_tLayout;
static int g_bTerminate = 0;
static int g_bProcIfpOnly = 0;
unsigned int g_dwStreamingNum = 0;
unsigned int g_dwStreamingNumRs = 0;
char* g_szAutoSceneConfig = NULL;
char* g_szSensorConfig = NULL; 
VMF_VSRC_HANDLE_T *g_ptVsrcHandle = NULL;
VMF_BIND_CONTEXT_T *g_ptBind = NULL;
VMF_VENC_OUT_SRB_T *g_ptVencOutputSRB = NULL;
VMF_VENC_OUT_SRB_T *g_ptVencOutputRsSRB = NULL;
VMF_VENC_OUT_SCM_T *g_ptVencOutputSCM = NULL;
VMF_VENC_OUT_SCM_T *g_ptVencOutputRsSCM = NULL;
VMF_VENC_HANDLE_T *g_ptVencHandle = NULL;
VMF_VENC_HANDLE_T *g_ptVencRsHandle = NULL;

static void exit_handler(void);
static void sig_handler(int);
static void print_msg(const char *, ...);
static void vsrc_init_callback(unsigned int, unsigned int);
static void release_video_source(void);
static int init_video_source();
static void release_bind(void);
static int init_bind(void);
static int init_output_srb(const char *, unsigned int, unsigned int, VMF_VENC_OUT_SRB_T **);
static void release_video_encoder(VMF_VENC_HANDLE_T *, VMF_VENC_OUT_RELEASE_FUNC, void **); 
static int init_video_encoder(VMF_VENC_HANDLE_T **, VMF_VENC_CODEC_TYPE, unsigned int,  unsigned int, unsigned int, VMF_VENC_OUT_SETUP_FUNC, void *);
static void msg_callback(MsgContext *, void *);

void exit_handler(void)
{
	release_string(g_szAutoSceneConfig);
	release_string(g_szSensorConfig);
}

void print_msg(const char *fmt, ...)
{
	va_list ap;		
	va_start(ap, fmt);
	fprintf(stderr, "[%s] ", MODULE_NAME);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}

void vsrc_init_callback(unsigned int width, unsigned int height)
{
	memset(&g_tLayout, 0, sizeof g_tLayout);
	g_tLayout.dwCanvasWidth = width;
	g_tLayout.dwCanvasHeight = height;
	g_tLayout.dwVideoPosX = 0;
	g_tLayout.dwVideoPosY = 0;
	g_tLayout.dwVideoWidth = width;
	g_tLayout.dwVideoHeight = height;
	print_msg("[%s]: width:%d, height:%d \n",__func__, width, height);
}

void release_video_source(void)
{
	if (g_ptVsrcHandle) {
		VMF_VSRC_Stop(g_ptVsrcHandle);
		VMF_VSRC_Release(g_ptVsrcHandle);
	}
}

int init_video_source()
{
	VMF_VSRC_INITOPT_T tVsrcInitOpt;
	VMF_VSRC_FRONTEND_CONFIG_T tVsrcFrontendConfig;

	memset(&tVsrcInitOpt, 0, sizeof tVsrcInitOpt);
	memset(&tVsrcFrontendConfig, 0, sizeof tVsrcFrontendConfig);

	tVsrcFrontendConfig.dwSensorConfigCount = 1;
	tVsrcFrontendConfig.apszSensorConfig[0] = g_szSensorConfig;
	tVsrcInitOpt.dwFrontConfigCount = 1;
	tVsrcInitOpt.ptFrontConfig = &tVsrcFrontendConfig;
	tVsrcInitOpt.pszAutoSceneConfig = g_szAutoSceneConfig;
	tVsrcInitOpt.pszOutPinPrefix = VENC_VSRC_PIN;
	tVsrcInitOpt.fnInitCallback = vsrc_init_callback;
	tVsrcInitOpt.pszResourceDir = VENC_RESOURCE_DIR;
    tVsrcInitOpt.tEngConfig.dwIspMode = VMF_ISP_MODE_DISABLE;

	g_ptVsrcHandle =  VMF_VSRC_Init(&tVsrcInitOpt);
	if (!g_ptVsrcHandle) {
		print_msg("[%s] VMF_VSRC_Init failed!\n", __func__);
		return -1;
	}

	if (VMF_VSRC_Start(g_ptVsrcHandle, NULL) != 0) {
		print_msg("[%s] VMF_VSRC_Start failed!\n", __func__);
		return -1;
	}

	return 0;
}

void release_bind(void)
{
	if (g_ptBind)
		VMF_BIND_Release(g_ptBind);
}

int init_bind(void)
{
	VMF_BIND_INITOPT_T tBindOpt;
	memset(&tBindOpt, 0, sizeof tBindOpt);

	tBindOpt.dwSrcOutputIndex = 0;
	tBindOpt.ptSrcHandle = g_ptVsrcHandle;
	tBindOpt.pfnQueryFunc = (VMF_BIND_QUERY_FUNC) VMF_VSRC_GetInfo;
	tBindOpt.pfnIspFunc = (VMF_BIND_CONFIG_ISP_FUNC) VMF_VSRC_ConfigISP;

	g_ptBind = VMF_BIND_Init(&tBindOpt);
	if (!g_ptBind){
		print_msg("[%s] VMF_BIND_Init failed!!\n", __func__);
		return -1;
	}

	return 0;
}

int init_output_srb(const char* name, unsigned int buf_number, unsigned int buf_size, VMF_VENC_OUT_SRB_T** pptVencOutputSRB)
{
	VMF_VENC_OUT_SRB_INITOPT_T tSrbInitOpt;
	memset(&tSrbInitOpt, 0, sizeof(VMF_VENC_OUT_SRB_INITOPT_T));

	tSrbInitOpt.pszSrbName = name;
	tSrbInitOpt.dwSrbNum  = buf_number;
	tSrbInitOpt.dwSrbSize = buf_size;

	if (0 != VMF_VENC_OUT_SRB_Init(pptVencOutputSRB, &tSrbInitOpt)) {
		print_msg("[%s] VMF_VENC_OUT_SRB_Init failed!!\n", __func__);
		return -1;
	}

	return 0;
}

void release_video_encoder(VMF_VENC_HANDLE_T *ptVencHandle, VMF_VENC_OUT_RELEASE_FUNC fnVencOutReleaseFunc, void **ppVencOutput)
{
	if (ptVencHandle)
		VMF_VENC_Release(ptVencHandle);

	if (fnVencOutReleaseFunc && ppVencOutput && *ppVencOutput)
		fnVencOutReleaseFunc(ppVencOutput);
}

int init_video_encoder(VMF_VENC_HANDLE_T **pptVencHandle, VMF_VENC_CODEC_TYPE eCodecType, unsigned int dwWidth, unsigned int dwHeight,
		unsigned int dwBitrate, VMF_VENC_OUT_SETUP_FUNC fnVencOutSetupFunc, void *pVencOutput)
{
	VMF_VENC_CONFIG_T tVencConfig;
	void *pCodecConfig = NULL;
	VMF_H4E_CONFIG_T h4e_config = {
        VMF_H264ENC_HIGH,           // eProfile
		25,							// dwQp
		dwBitrate,					// dwBitrate
		15,							// fFps
		15,							// dwGop
        10,                         // dwMinIQp
        50,                         // dwMaxIQp
        10,                         // dwMinPQp
        50,                         // dwMaxPQp
		0,
	};

	VMF_H5E_CONFIG_T h5e_config = {
        VMF_H265ENC_MAIN,           // eProfile
		25,							// dwQp
		dwBitrate,					// dwBitrate
		15,							// fFps
		15,							// dwGop
        10,                         // dwMinIQp
        50,                         // dwMaxIQp
        10,                         // dwMinPQp
        50,                         // dwMaxPQp
		0,
	};

	VMF_JE_CONFIG_T je_config = {
		50,							// dwQp
		0,							// bEnableThumbnail
		0,							// dwThumbnailQp
		0,							// bJfifHdr
		1024*1024,					// dwBitrate	        
		5		
	};

	memset(&tVencConfig, 0, sizeof tVencConfig);
	tVencConfig.dwEncWidth = dwWidth;
	tVencConfig.dwEncHeight = dwHeight;
	tVencConfig.bConnectIfp = g_bProcIfpOnly;
    tVencConfig.eProcessMode = VMF_VENC_ONE_FRAME;
    
	print_msg("[%s] enc :%d, %d\n", __func__, dwWidth, dwHeight);
	switch(eCodecType){
	case VMF_VENC_CODEC_TYPE_H264:
		eCodecType = VMF_VENC_CODEC_TYPE_H264;
		pCodecConfig = &h4e_config;
		break;
	case VMF_VENC_CODEC_TYPE_H265: 
		eCodecType = VMF_VENC_CODEC_TYPE_H265;
		pCodecConfig = &h5e_config;
		break;
	case VMF_VENC_CODEC_TYPE_MJPG: 
		eCodecType = VMF_VENC_CODEC_TYPE_MJPG;
		pCodecConfig = &je_config;
		break;
	default: 
		print_msg("[%s] Invalid Codec Type\n", __func__);
		return -1;
	}

	fnVencOutSetupFunc(&tVencConfig, eCodecType, pCodecConfig, pVencOutput);

	tVencConfig.fnSrcConnectFunc = (VMF_SRC_CONNECT_FUNC) VMF_BIND_Request;
	tVencConfig.pBind = g_ptBind;

	*pptVencHandle = VMF_VENC_Init(&tVencConfig);
	if (!*pptVencHandle) {
		print_msg("[%s] VMF_VENC_Init() failed\n", __func__);		
		return -1;
	}

	/* start the video encoder engine */
	VMF_VENC_ProduceStreamHdr(*pptVencHandle);
	return 0;
}

void msg_callback(MsgContext* msg_context, void* user_data)
{
	(void) user_data;
	print_msg("msg_context->pszHost=%s, msg_context->pszCmd=%s \n", 
			msg_context->pszHost, msg_context->pszCmd);

	if (!strcasecmp(msg_context->pszHost, "encoder0")) {
		if (!strcasecmp(msg_context->pszCmd, "start")) {
			if (++g_dwStreamingNum == 1) {
				VMF_VENC_Start(g_ptVencHandle);
			}
		} else if (!strcasecmp(msg_context->pszCmd, "stop")) {
			if (g_dwStreamingNum) {
				if (--g_dwStreamingNum == 0) {
					VMF_VENC_Stop(g_ptVencHandle);
				}
			}
		} else if (!strcasecmp(msg_context->pszCmd, "forceCI")) {
			VMF_VENC_ProduceStreamHdr(g_ptVencHandle);
		} else if (!strcasecmp(msg_context->pszCmd, "forceIntra")) {
			VMF_VENC_SetIntra(g_ptVencHandle);
		}
	} else if (!strcasecmp(msg_context->pszHost, "encoder1")) {
		if (!strcasecmp(msg_context->pszCmd, "start")) {
			if (++g_dwStreamingNumRs == 1) {
				VMF_VENC_Start(g_ptVencRsHandle);
			}
		} else if (!strcasecmp(msg_context->pszCmd, "stop")) {
			if (g_dwStreamingNumRs) {
				if (--g_dwStreamingNumRs == 0) {
					VMF_VENC_Stop(g_ptVencRsHandle);
				}
			}
		} else if (!strcasecmp(msg_context->pszCmd, "forceCI")) {
			VMF_VENC_ProduceStreamHdr(g_ptVencRsHandle);
		} else if (!strcasecmp(msg_context->pszCmd, "forceIntra")) {
			VMF_VENC_SetIntra(g_ptVencRsHandle);
		}
	} 

	if (msg_context->bHasResponse) {
		msg_context->dwDataSize = 0;
	}
}

void sig_handler(int signo)
{
	print_msg("[%s] receive SIGNAL: %d\n",__func__, signo);
	g_bTerminate = 1;
}

int main(int argc, char* argv[])
{
	int ch, ret = -1;
	VMF_VENC_CODEC_TYPE eCodecType = VMF_VENC_CODEC_TYPE_H264;
	VMF_VENC_OUT_SETUP_FUNC fnVencOutSetupFunc = NULL;
	VMF_VENC_OUT_RELEASE_FUNC fnVencOutReleaseFunc = NULL;
	void *pVencOutput = NULL, *pVencOutputRs = NULL;

	/* register exit handler */
	atexit(exit_handler);
	
	/* register signal handler */
	signal(SIGTERM, sig_handler);
	signal(SIGINT, sig_handler);

	while ((ch = getopt(argc, argv, "c:C:a:f")) != -1) {
		switch(ch) {
		case 'c':
			g_szSensorConfig = strdup(optarg);
			break;
		case 'C':
			eCodecType = (VMF_VENC_CODEC_TYPE) atoi(optarg);
			break;
		case 'a':
			g_szAutoSceneConfig = strdup(optarg);
			break;
		case 'f':
			g_bProcIfpOnly = atoi(optarg);
			break;
		default:
			print_msg("Usage: %s [-c<sensor_config_file>] [-C<codec_type>] "
						"[-a autosecne_config_file] [-f bIfpOnly] \r\n", argv[0]);
			goto RELEASE;
		}
	}

	/* check sensor config */
	if (!g_szSensorConfig){
		print_msg("[%s] Err: no sensor config\n", __func__);
		goto RELEASE;
	}

	/* initializing the video source */
	if (init_video_source()) {
		print_msg("[%s] init_video_source failed!\n", __func__);
		goto RELEASE;
	}
	
	/* initializing the binder associated with the video source */
	if (init_bind()) {
		print_msg("[%s] init_bind failed!\n", __func__);
		goto RELEASE;
	}

	/* initialize the SRB for primary encoder output buffer */
	if (init_output_srb(VENC_OUTPUT_PIN,  VENC_OUTPUT_BUF_NUM, VENC_ENCODE_BUF_SIZE, &g_ptVencOutputSRB)
		|| (!g_bProcIfpOnly && init_output_srb(VENC_OUTPUT_PIN_RS,  VENC_OUTPUT_BUF_NUM, VENC_ENCODE_BUF_SIZE_RS, &g_ptVencOutputRsSRB))) {
		print_msg("[%s] init_output_srb failed!\n", __func__);
		goto RELEASE;
	}
	pVencOutput   = g_ptVencOutputSRB;
	pVencOutputRs = g_ptVencOutputRsSRB;
	fnVencOutSetupFunc = VMF_VENC_OUT_SRB_Setup_Config;
	fnVencOutReleaseFunc = (VMF_VENC_OUT_RELEASE_FUNC) VMF_VENC_OUT_SRB_Release;

	/* initialize encoder for main stream */
	if (init_video_encoder(&g_ptVencHandle, eCodecType, g_tLayout.dwVideoWidth, g_tLayout.dwVideoHeight, 4000000, fnVencOutSetupFunc, pVencOutput)
		|| (!g_bProcIfpOnly && init_video_encoder(&g_ptVencRsHandle, eCodecType, RESIZE_WIDTH, RESIZE_HEIGHT, 1000000, fnVencOutSetupFunc, pVencOutputRs))) {
		print_msg("[%s] init_video_encoder failed!\n", __func__);
		goto RELEASE;
	}

	/* register the message communication pipe */
	MsgBroker_RegisterMsg(VENC_CMD_FIFO);
	MsgBroker_Run(VENC_CMD_FIFO, msg_callback, NULL, &g_bTerminate);
	MsgBroker_UnRegisterMsg();

	ret = 0;
	
RELEASE:
	release_video_encoder(g_ptVencHandle, fnVencOutReleaseFunc, &pVencOutput);
	if (!g_bProcIfpOnly)
		release_video_encoder(g_ptVencRsHandle, fnVencOutReleaseFunc, &pVencOutputRs);
	release_bind();
	release_video_source();
	
	return ret;
}
