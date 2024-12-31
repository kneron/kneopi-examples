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
#include <stdarg.h>
#include <stdbool.h>
#include <string.h>
#include <signal.h>
#include <fcntl.h>
#include <errno.h>
#include <getopt.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>

#include <mem_broker.h>
#include <sync_shared_memory.h>
#include <video_isp_process.h>
#include <ssm_info.h>
#include <frame_info.h>

#include "vmf_ie_engine.h"

#define MODULE_NAME			"usb_camera"
#define VENC_VSRC_C_PIN     "vsrc_ssm_c_0"              //! VMF_VSRC Customer Output pin
#define FPS 30
#define BUF_NUMBER 4

struct Device {
	int iFd;
	enum v4l2_buf_type	eBufType;
	enum v4l2_memory	eMemType;
	unsigned int dwWidth;
	unsigned int dwHeight;
};

struct MapBuf {
	unsigned int dwSize[VIDEO_MAX_PLANES];
	void *pbyMem[VIDEO_MAX_PLANES];
};

static int g_bTerminate = 0;
static unsigned int g_dwCameraWidth = 0;
static unsigned int g_dwCameraHeight = 0;

void print_msg(const char *fmt, ...)
{
	va_list ap;		
	va_start(ap, fmt);
	fprintf(stderr, "[%s] ", MODULE_NAME);
	vfprintf(stderr, fmt, ap);
	va_end(ap);
}

void ssm_clear_header(unsigned char* virt_addr, unsigned int buf_size, void* pUserData)
{
	VMF_VSRC_SSM_OUTPUT_INFO_T* vsrc_ssm_writer_info = (VMF_VSRC_SSM_OUTPUT_INFO_T*) pUserData;

	if (buf_size > VMF_MAX_SSM_HEADER_SIZE)
		memset(virt_addr, 0, VMF_MAX_SSM_HEADER_SIZE);

	VMF_VSRC_SSM_SetInfo(virt_addr, vsrc_ssm_writer_info);	
}

static void sig_kill(int signo)
{
	print_msg("[%s] receive SIGNAL: %d\n",__func__, signo);
	g_bTerminate = 1;
}

static int init_cam(struct Device *pDev, const char *pszDeviceName, unsigned int dwWidth, unsigned int dwHeight)
{
	pDev->eBufType = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	pDev->eMemType = V4L2_MEMORY_MMAP;//V4L2_MEMORY_USERPTR;//V4L2_MEMORY_MMAP;
	pDev->dwWidth = dwWidth;
	pDev->dwHeight = dwHeight;
	pDev->iFd = open(pszDeviceName, O_RDWR);
	if (pDev->iFd < 0) {
		pDev->iFd = 0;
		printf("Open device failed\n");
	}
	return -1;
}

static int close_cam(struct Device *pDev)
{
	if (pDev->iFd) {
		close(pDev->iFd);
		pDev->iFd = 0;
	}
	return pDev->iFd;
}

static int config_cam(struct Device *pDev)
{
	struct v4l2_format tFormat;
	struct v4l2_fmtdesc tFmtdesc;
	struct v4l2_streamparm tParm;
	unsigned int i = 0;
	//! enum format
	printf("Available formats:\n");
	memset(&tFmtdesc, 0, sizeof tFmtdesc);
	tFmtdesc.index = i;
	tFmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	while (ioctl(pDev->iFd, VIDIOC_ENUM_FMT, &tFmtdesc) != -1) {
		printf("\t%d.%s (0x%08x)\n",tFmtdesc.index+1,tFmtdesc.description, tFmtdesc.pixelformat);
		tFmtdesc.index++;
	}

	//! set format
	tFormat.type = pDev->eBufType;
	tFormat.fmt.pix.width = pDev->dwWidth;
	tFormat.fmt.pix.height = pDev->dwHeight;
	tFormat.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;//V4L2_PIX_FMT_MJPEG;
	tFormat.fmt.pix.field = V4L2_FIELD_NONE;
	tFormat.fmt.pix.bytesperline = 0; //stride
	tFormat.fmt.pix.sizeimage = 0;
	tFormat.fmt.pix.priv = V4L2_PIX_FMT_PRIV_MAGIC;
	tFormat.fmt.pix.flags = 0;

	if (ioctl(pDev->iFd, VIDIOC_S_FMT, &tFormat) < 0) {
		printf("Config cam failed\n");
		return -1;
	}
	printf("Config format: %s (0x%08x) %ux%u (stride %u) field %s buffer size %u\n", "YUYV", tFormat.fmt.pix.pixelformat, tFormat.fmt.pix.width, tFormat.fmt.pix.height, tFormat.fmt.pix.bytesperline, "none", tFormat.fmt.pix.sizeimage);

	//! set frame rate
	memset(&tParm, 0, sizeof tParm);
	tParm.type = pDev->eBufType;
	ioctl(pDev->iFd, VIDIOC_G_PARM, &tParm);
	printf("Current frame rate %u/%u\n", tParm.parm.capture.timeperframe.numerator, tParm.parm.capture.timeperframe.denominator);

	tParm.parm.capture.timeperframe.numerator = 1;
	tParm.parm.capture.timeperframe.denominator = FPS;//30;
	printf("Config frame rate %u/%u\n", tParm.parm.capture.timeperframe.numerator, tParm.parm.capture.timeperframe.denominator);

	ioctl(pDev->iFd, VIDIOC_S_PARM, &tParm);

	ioctl(pDev->iFd, VIDIOC_G_PARM, &tParm);
	printf("Frame rate %u/%u\n", tParm.parm.capture.timeperframe.numerator, tParm.parm.capture.timeperframe.denominator);

	return 0;
}

static int video_queue_buffer(struct Device *dev, int index)
{
	struct v4l2_buffer buf;
	struct v4l2_plane planes[VIDEO_MAX_PLANES];
	int ret;

	memset(&buf, 0, sizeof buf);
	memset(&planes, 0, sizeof planes);

	buf.index = index;
	buf.type = dev->eBufType;
	buf.memory = dev->eMemType;

	ret = ioctl(dev->iFd, VIDIOC_QBUF, &buf);
	if (ret < 0)
		printf("Unable to queue buffer: %s (%d).\n",
			strerror(errno), errno);

	return ret;
}

void* capture_video(void *pParam)
{
	int ret = -1;
	unsigned int i;

	unsigned int dwWidth = g_dwCameraWidth;
	unsigned int dwHeight = g_dwCameraHeight;
	void *pbyYUV422 = NULL;

	struct Device *pDev = pParam;
	struct v4l2_requestbuffers rb;
	struct v4l2_plane planes[VIDEO_MAX_PLANES];
	struct v4l2_buffer buf;
	struct MapBuf *pBufs = NULL;	
	struct MapBuf *pBuf = NULL;

    VMF_IE_HANDLE_T *ptIeHandle = NULL;
    VMF_IE_CONFIG_T tIeConfig;
    VMF_IE_STATE_T tIeState;

	SSM_WRITER_INIT_OPTION_T ssm_opt;
	SSM_HANDLE_T* ptSsmWriterHandle = NULL;
	SSM_BUFFER_T tWriterSsmBuffer;
	VMF_VSRC_SSM_OUTPUT_INFO_T vsrc_ssm_writer_info;

	//! allocate buffer
	memset(&rb, 0, sizeof rb);
	rb.count = BUF_NUMBER;
	rb.type = pDev->eBufType;
	rb.memory = pDev->eMemType;
	if (ioctl(pDev->iFd, VIDIOC_REQBUFS, &rb) < 0) {
		printf("Request buffer failed\n");
		goto CLOSE;
	}
	pBufs = malloc(rb.count * sizeof pBufs[0]);
	if (pBufs == NULL) {
		printf("Allocate buffer failed\n");
		goto CLOSE;
	}
	for (i = 0; i < rb.count; i++) {
		pBuf = pBufs + i;
		memset(&buf, 0, sizeof buf);
		memset(planes, 0, sizeof planes);
		buf.index = i;
		buf.type = pDev->eBufType;
		buf.memory = pDev->eMemType;
		buf.length = VIDEO_MAX_PLANES; //?
		buf.m.planes = planes;
		if (ioctl(pDev->iFd, VIDIOC_QUERYBUF, &buf)) {
			printf("Query buffer failed\n");
			goto CLOSE;
		}
		printf("buf[%d] length %d offset %d\n", i, buf.length, buf.m.offset);

		//! one plane
		pBuf->pbyMem[0] = mmap(0, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, pDev->iFd, buf.m.offset);
		if (pBuf->pbyMem[0] == MAP_FAILED) {
			printf("Map buffer failed\n");
			goto CLOSE;
		}
		pBuf->dwSize[0] = buf.length;
	}

	pbyYUV422 = MemBroker_GetMemory(dwWidth*dwHeight*2, VMF_ALIGN_TYPE_128_BYTE);

	memset(&ssm_opt, 0, sizeof ssm_opt);
	memset(&tWriterSsmBuffer, 0, sizeof(SSM_BUFFER_T));
	memset(&vsrc_ssm_writer_info, 0, sizeof vsrc_ssm_writer_info);

	vsrc_ssm_writer_info.dwYStride = dwWidth;
	vsrc_ssm_writer_info.dwYSize = vsrc_ssm_writer_info.dwYStride*dwHeight;
	vsrc_ssm_writer_info.dwUVSize = vsrc_ssm_writer_info.dwYSize>>2;
	vsrc_ssm_writer_info.dwOffset[0] = VMF_MAX_SSM_HEADER_SIZE;
	vsrc_ssm_writer_info.dwOffset[1] = vsrc_ssm_writer_info.dwOffset[0] + vsrc_ssm_writer_info.dwYSize;
	vsrc_ssm_writer_info.dwOffset[2] = vsrc_ssm_writer_info.dwOffset[1] + vsrc_ssm_writer_info.dwUVSize;
	vsrc_ssm_writer_info.dwWidth = dwWidth;
	vsrc_ssm_writer_info.dwHeight = dwHeight;
	ssm_opt.name      = VENC_VSRC_C_PIN;
	ssm_opt.buf_size  = dwWidth*dwHeight*3/2 + VMF_MAX_SSM_HEADER_SIZE;
	ssm_opt.alignment = VMF_ALIGN_TYPE_DEFAULT;
	ssm_opt.pUserData = &vsrc_ssm_writer_info;
	ssm_opt.fp_setup_buffer = ssm_clear_header;;
	ptSsmWriterHandle = SSM_Writer_Init(&ssm_opt);
	if (!ptSsmWriterHandle) {
		print_msg("%s() failed, SSM_Reader_Init failed!\n", __func__);
		goto CLOSE;
	}
	SSM_Writer_SendGetBuff(ptSsmWriterHandle, &tWriterSsmBuffer);

	if ((ptIeHandle = VMF_IE_Init()))
	{
		memset(&tIeConfig, 0, sizeof(tIeConfig));
		memset(&tIeState, 0, sizeof(tIeState));
		tIeConfig.dwSrcWidth = dwWidth;
		tIeConfig.dwSrcHeight = dwHeight;
		tIeConfig.eSrcImageFormat = VMF_IE_IMAGE_FORMAT_YUYV422;

		tIeConfig.dwDstWidth = dwWidth;
		tIeConfig.dwDstHeight = dwHeight;
		tIeConfig.eDstImageFormat = VMF_IE_IMAGE_FORMAT_YUV420;

		tIeState.dwSrcBuffSize = dwWidth*dwHeight*2;
		tIeState.dwDstBuffSize = dwWidth*dwHeight*3/2;
		tIeState.pbyDstPhyAddr = tWriterSsmBuffer.buffer_phys_addr + vsrc_ssm_writer_info.dwOffset[0];
		tIeState.pbyDstVirAddr = tWriterSsmBuffer.buffer + vsrc_ssm_writer_info.dwOffset[0];
		VMF_IE_Config(ptIeHandle, &tIeConfig);
	} else {
		printf("----- ie init failed\n");
		goto CLOSE;
	}

	ioctl(pDev->iFd, VIDIOC_STREAMON, &pDev->eBufType);

	//! queue buffer
	for (i=0; i < BUF_NUMBER; i++) {
		video_queue_buffer(pDev, i);
	}

	while (!g_bTerminate) {
		memset(&buf, 0, sizeof buf);
		memset(planes, 0, sizeof planes);
		buf.type = pDev->eBufType;
		buf.memory = pDev->eMemType;
		buf.length = VIDEO_MAX_PLANES;
		buf.m.planes = planes;
		if (ioctl(pDev->iFd, VIDIOC_DQBUF, &buf) < 0) {
			printf("Dequeue buffer failed\n");
			goto CLOSE;
		}

		// printf("buffer[%d] bytesused %d\n", buf.index, buf.bytesused);

		pBuf = pBufs + buf.index;

		VMF_FRAME_INFO_T *pInfo = (VMF_FRAME_INFO_T *)tWriterSsmBuffer.buffer;
		pInfo->dwSec = (unsigned int)buf.timestamp.tv_sec;
		pInfo->dwUSec = (unsigned int)buf.timestamp.tv_usec;

		memcpy(pbyYUV422, pBuf->pbyMem[0], dwWidth*dwHeight*2);
		MemBroker_CacheCopyBack(pbyYUV422, dwWidth*dwHeight*2);

        tIeState.pbySrcPhyAddr = (unsigned char *)MemBroker_GetPhysAddr(pbyYUV422);
        tIeState.pbySrcVirAddr = pbyYUV422;
		tIeState.pbyDstPhyAddr = tWriterSsmBuffer.buffer_phys_addr + vsrc_ssm_writer_info.dwOffset[0];
		tIeState.pbyDstVirAddr = tWriterSsmBuffer.buffer + vsrc_ssm_writer_info.dwOffset[0];
        VMF_IE_ProcessOneFrame(ptIeHandle, &tIeState);

		SSM_Writer_SendGetBuff(ptSsmWriterHandle, &tWriterSsmBuffer);

		ret = video_queue_buffer(pDev, buf.index);
		if (ret < 0)
			printf("Unable to queue buffer: %s (%d).\n", strerror(errno), errno);
	}
	ioctl(pDev->iFd, VIDIOC_STREAMOFF, &pDev->eBufType);

CLOSE:
	if (pbyYUV422)
		MemBroker_FreeMemory(pbyYUV422);
	if(ptIeHandle)
		VMF_IE_Release(ptIeHandle);
	if (ptSsmWriterHandle) {
		SSM_Release(ptSsmWriterHandle);
	}
	if (pBufs) {
		for (i=0; i < BUF_NUMBER; i++) {
			pBuf = pBufs + i;
			munmap(pBuf->pbyMem[0], pBuf->dwSize[0]);
		}
		memset(&rb, 0, sizeof rb);
		rb.count = 0;
		rb.type = pDev->eBufType;
		rb.memory = pDev->eMemType;
		if (ioctl(pDev->iFd, VIDIOC_REQBUFS, &rb) < 0) {
			printf("Release buffers failed\n");
		}
		free(pBufs);
	}
	return NULL;
}


int main(int argc, char* argv[])
{
	int ch = -1;
	struct Device tDev;
	pthread_t tcapture;

	/* register signal */
	signal(SIGTERM, sig_kill);
	signal(SIGINT, sig_kill);

	while ((ch = getopt(argc, argv, "hW:H:")) != -1) {
		switch(ch) {
		case 'W':
			g_dwCameraWidth = atoi(optarg);
			break;
		case 'H':
			g_dwCameraHeight = atoi(optarg);
			break;
		case 'h':
		default:
			print_msg("Usage: %s  [-W Camera Width] [-H Camera Height]\r\n", argv[0]);
			goto RELEASE;
		}
	}

	/* check camera */
	if ((0>=g_dwCameraWidth) || (0>=g_dwCameraHeight)){
		print_msg("[%s] Err: Camera width & height setting failed\n", __func__);
		goto RELEASE;
	}

	memset(&tDev, 0, sizeof tDev);
	init_cam(&tDev, "/dev/video0", g_dwCameraWidth, g_dwCameraHeight);
	if (config_cam(&tDev)) {
		close_cam(&tDev);
	}
	if (tDev.iFd) {
		pthread_create(&tcapture, NULL, capture_video, &tDev);
		pthread_setname_np(tcapture, "v_usb_cap");
	}

	pthread_join(tcapture, NULL);
	close_cam(&tDev);
RELEASE:
	print_msg("terminated successfully!\n");
	
	return 0;
}
