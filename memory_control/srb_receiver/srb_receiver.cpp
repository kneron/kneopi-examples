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
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <cstdio>
#include <cstring>
#include <signal.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <sys/types.h> 

#include <sync_ring_buffer.h>
#include <msg_broker.h>
#include <mem_util.h>
#include <video_encoder.h>

using namespace std;

#define MAKEFOURCC(ch0, ch1, ch2, ch3)  ((unsigned int)(unsigned char)(ch0) | ((unsigned int)(unsigned char)(ch1) << 8) | ((unsigned int)(unsigned char)(ch2) << 16) | ((unsigned int)(unsigned char)(ch3) << 24 ))

#define FOURCC_CONF (MAKEFOURCC('C','O','N','F'))	
#define FOURCC_H264 (MAKEFOURCC('H','2','6','4'))
#define FOURCC_H265 (MAKEFOURCC('H','2','6','5'))
#define FOURCC_JPEG (MAKEFOURCC('J','P','E','G'))

#define OUTPUT_BUFFER_HEADER 256

#define SRB_RCV_VENC_PIN1     "venc_srb_1"                //! VMF_VideoEnc Output pin
#define SRB_RCV_VENC_CMD_FIFO "/tmp/venc/c0/command.fifo" //! communicate with rtsps, vrec, etc.

bool g_bWriteFile = 0;
bool g_bRunning = 1;
char* g_pszOutputPath = NULL;
SRB_HANDLE_T* g_aptSrbHandle;

int cmdfifo_video_sender(const char* cmd, int ch)
{
	MsgContext msg_context;
	char host[64] = {0};		
	sprintf(host, "encoder%d", ch);
	
	msg_context.pszHost = host;
	msg_context.dwHostLen = strlen(host) + 1;
	msg_context.pszCmd = cmd;
	msg_context.dwCmdLen = strlen(msg_context.pszCmd) + 1;
	msg_context.dwDataSize = 0;
	msg_context.pbyData = NULL;					
	msg_context.bHasResponse = 0;		
	if (MsgBroker_SendMsg(SRB_RCV_VENC_CMD_FIFO, &msg_context)) {
		return 0;
	}

	return -1;
}

#define BUFFER_SIZE 1024*1024

static void double_buf(FILE* fp, unsigned int channel, unsigned char *pbyEncPtr, unsigned int dataSize) 
{
	static unsigned char tmpbuf[2][2][BUFFER_SIZE] = {0};
	static unsigned int buf_idx[2] = {0};
	static unsigned char bswitch[2] = {0};
	unsigned char* buf;
		
	if (buf_idx[channel] + dataSize < BUFFER_SIZE) {
		buf = tmpbuf[channel][bswitch[channel]];
		memcpy(buf+buf_idx[channel], pbyEncPtr, dataSize);
		buf_idx[channel] = buf_idx[channel] + dataSize; 		
	}
	else {
		if (fp) {
			fwrite(tmpbuf[channel][bswitch[channel]], buf_idx[channel], 1, fp);
			if (dataSize > BUFFER_SIZE) {
				fwrite(pbyEncPtr, dataSize, 1, fp);
				return;
			}
		}
			
		bswitch[channel] = !bswitch[channel];
		buf = tmpbuf[channel][bswitch[channel]];
		buf_idx[channel] = 0;
		memcpy(buf+buf_idx[channel], pbyEncPtr, dataSize);
		buf_idx[channel] = buf_idx[channel] + dataSize; 
	}
}

void h26x_output_data(unsigned int dwCodec, FILE* fp, unsigned int dwIndex, bool* pbIsKeyFrame, unsigned char *pbyEncPtr)
{
	VMF_VENC_STREAM_DATA_HDR* data_hdr = (VMF_VENC_STREAM_DATA_HDR*) pbyEncPtr;
	unsigned char* pdwPayload = pbyEncPtr + OUTPUT_BUFFER_HEADER;
				
	if (data_hdr->bIsKeyFrame) {
		*pbIsKeyFrame = 1;
	}

	if (!(*pbIsKeyFrame)) {
		return;
	}

    printf("[srb_receiver] %s() %s ch[%d] nal type(%x), data size(%d) \n", 
		    __func__, dwCodec == FOURCC_H264?"H264":"H265", dwIndex, dwCodec == FOURCC_H264 ? pdwPayload[4] & 0x1F : (pdwPayload[4] >> 1) & 0x3F, data_hdr->dwDataBytes);
    
	if (g_bWriteFile) {		
		double_buf(fp, dwIndex - 1, pdwPayload, data_hdr->dwDataBytes);
	}
}

void jpg_output_data(char* pszFolder, unsigned int dwIndex, unsigned char *pbyEncPtr)
{
	VMF_VENC_STREAM_DATA_HDR* data_hdr = (VMF_VENC_STREAM_DATA_HDR*) pbyEncPtr;
	printf("[srb_receiver] %s() ch[%d] data size(%d) \n", __func__, dwIndex, data_hdr->dwDataBytes);			
				
	if (g_bWriteFile) {
		FILE* fp = NULL;
		struct timeval tv;
		char szPath[128] = {0};
		gettimeofday(&tv,NULL);
		snprintf(szPath, 128, "%s/vtcs_srb_ch%d-%ld_%ld.jpg", pszFolder?pszFolder:".", dwIndex, tv.tv_sec, tv.tv_usec);
		fp = fopen(szPath, "wb");
		if (fp) {
			fwrite(pbyEncPtr + OUTPUT_BUFFER_HEADER, data_hdr->dwDataBytes, 1, fp);
			fclose(fp);
		}
	}	
}

static void gen_output_path(char* pszOutputPath, char* pszFolder, unsigned int dwChannelIdx, const char* pszFileName)
{
	if (pszFolder) {
		if (pszFolder[strlen(pszFolder) - 1] == '/') {
			sprintf(pszOutputPath, "%svtcs_srb_ch%d.%s", pszFolder, dwChannelIdx, pszFileName);
		} else {
			sprintf(pszOutputPath, "%s/vtcs_srb_ch%d.%s", pszFolder, dwChannelIdx, pszFileName);
		}
	} else {
		sprintf(pszOutputPath, "vtcs_srb_ch%d.%s", dwChannelIdx, pszFileName);
	}
}

void* srb_reader(void *pParam)
{
	int ret = 0;
	unsigned int dwChannelIdx = addr2uint(pParam);
	SRB_HANDLE_T* ptSrbHandle = NULL;	
	SRB_BUFFER_T srb_buf;	
	unsigned int* values;
	unsigned char *enc_ptr;
	bool bIsKeyframe = 0;
	unsigned char vps[64] = {0};
	unsigned char sps[64] = {0};
	unsigned char pps[64] = {0};  
	unsigned short vps_size = 0;
	unsigned short sps_size = 0;
	unsigned short pps_size = 0; 	
	FILE* fp = NULL;

	ptSrbHandle = g_aptSrbHandle = SRB_InitReader(SRB_RCV_VENC_PIN1);
	memset(&srb_buf, 0, sizeof(SRB_BUFFER_T));	
	cmdfifo_video_sender("start", dwChannelIdx-1);
	cmdfifo_video_sender("forceCI", dwChannelIdx-1); // conf	

	while (g_bRunning) {
		//! Receive video data 
		ret = SRB_ReturnReceiveReaderBuff(ptSrbHandle, &srb_buf);
		if (ret < 0)  { 
			usleep(1000); 
			printf("[srb_receiver] %s() SRB_ReturnReceiveReaderBuff Fail !\n", __func__); 
			continue; 
		}
		enc_ptr = srb_buf.buffer;		
		values = (unsigned int*) enc_ptr;

		if (values[0] == FOURCC_CONF) {
			if (values[2] == FOURCC_H264) {		
				//! SPS/PPS 
				unsigned char* dst_buf = enc_ptr + addr2uint(&(((VMF_VENC_H264_STREAM_HDR*)0)->abySpsData));  

                sps_size = values[6];
				pps_size = values[7];
				if (sps_size < sizeof(sps)) {
					memcpy(sps, dst_buf, sps_size);
				}
				if (pps_size < sizeof(pps)) {
					memcpy(pps, dst_buf + sps_size, pps_size);	
				}
                if (g_bWriteFile) {
					if (!fp) {
						char szPath[128] = {0};
						gen_output_path(szPath, g_pszOutputPath, dwChannelIdx, "h264");						
						fp = fopen(szPath, "wb+");
						fwrite(sps, sps_size, 1, fp);
						fwrite(pps, pps_size, 1, fp);
					}
				}
				printf("[srb_receiver] %s() H264 ch[%d] sps_size(%d) pps_size(%d) \n", __func__, dwChannelIdx, sps_size, pps_size);
			} else if (values[2] == FOURCC_H265) {
                //! VPS/SPS/PPS 
				unsigned char* dst_buf = enc_ptr + addr2uint(&(((VMF_VENC_H265_STREAM_HDR*)0)->abyVpsData));  

                vps_size = values[6];
				sps_size = values[7];
				pps_size = values[8];
				if (vps_size < sizeof(vps)) {
					memcpy(vps, dst_buf, vps_size);
				}
				
				if (sps_size < sizeof(sps)) {
					memcpy(sps, dst_buf + vps_size, sps_size);	
				}
				
				if (pps_size < sizeof(pps)) {
					memcpy(pps, dst_buf + vps_size + sps_size, pps_size);	
				}
				if (g_bWriteFile) {
					if (!fp) {
						char szPath[128] = {0};
						gen_output_path(szPath, g_pszOutputPath, dwChannelIdx, "h265");						
						fp = fopen(szPath, "wb+");
						fwrite(vps, vps_size, 1, fp);
						fwrite(sps, sps_size, 1, fp);
						fwrite(pps, pps_size, 1, fp);
					}
				}
				
				printf("[srb_receiver] %s() H265 ch[%d] vps_size(%d) sps_size(%d) pps_size(%d) \n", __func__, dwChannelIdx, vps_size, sps_size, pps_size);
			}
		} else {	
			switch (values[0]) {
				case FOURCC_H264: 
				case FOURCC_H265: {
					h26x_output_data(values[0], fp, dwChannelIdx, &bIsKeyframe, enc_ptr);
				} break;
				
				case FOURCC_JPEG: {
					jpg_output_data(g_pszOutputPath, dwChannelIdx, enc_ptr);					
				} break;
			}				
		}
	}

	SRB_ReturnReaderBuff(ptSrbHandle, &srb_buf);
	SRB_Release(ptSrbHandle);
	if (fp) {
		fclose(fp);
	}
	cmdfifo_video_sender("stop", dwChannelIdx - 1);
	
	return NULL;
}

static void sig_kill(int signo)
{
	printf("[srb_receiver] %s() receive SIGNAL: %d\n", __func__, signo);
	g_bRunning = 0;	
	if (g_aptSrbHandle) {
		SRB_WakeupReader(g_aptSrbHandle);
	}
}

int main(int argc, char* argv[])
{
	int cmd;
	pthread_t srb_pid1;
	while ((cmd = getopt(argc, argv, "w:h:p:")) != -1) {
		switch (cmd) {
			case 'w':
				g_bWriteFile = atoi(optarg);
				break;
				
			case 'p':
				g_pszOutputPath = strdup(optarg);
				break;

			case 'h':
			default:
				printf("Usage: %s [-w write_file(0:no,1:yes)] [-p output_path(output folder path)] [-h (this help)]\r\n", argv[0]);
				exit(1);
		}
	}
	
	signal(SIGTERM, sig_kill);
	signal(SIGINT, sig_kill);

	//! Create a SRB reader thread to receive bitstream 
	pthread_create(&srb_pid1, NULL, srb_reader, (void *) 1);

	pthread_join(srb_pid1, NULL);
	printf("[srb_receiver] Delete srb_reader thread \n");	
		
	free(g_pszOutputPath);

	return 0;
}
