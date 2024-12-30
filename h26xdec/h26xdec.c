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
#include <mem_util.h>
#include <video_decoder.h>
#include <vector_dma.h>

#define STREAM_FEEDING_SIZE     (1*1024*1024) // max bitstream size(HEVC:10MB,VP9:not specified)
#define FEEDING_SIZE     (256*1024)


static char* g_pszInputPath = NULL;
static char* g_pszOutputPath = NULL;
static unsigned int g_dwWidth = 0;
static unsigned int g_dwHeight = 0;
static unsigned int g_bWriteFile = 1;
static unsigned int g_eCodec = 1;


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
		g_pszInputPath = strdup(tmp);

	if ((tmp = iniparser_getstring(ini, "config:outputpath", NULL)) == NULL) {
		printf("No outputpath\n");
		return -1;
	}

	if(tmp)
		g_pszOutputPath = strdup(tmp);

	g_eCodec = iniparser_getint(ini, "config:codec", 1);	        
	g_dwWidth = iniparser_getint(ini, "config:width", 0);
	g_dwHeight = iniparser_getint(ini, "config:height", 0);
	g_bWriteFile = iniparser_getint(ini, "config:writefile", 1);
    
	iniparser_freedict(ini);
	return 0;
}

int main (int argc, char **argv)
{
	VMF_VDEC_HANDLE_T *ptH26xDecoder = NULL;
	VMF_H26XDEC_STATE_T *ptH26xState = NULL;
	unsigned int dwRet = 0;
	int opt;
	FILE *pfOutput = NULL;
	FILE *pfInput = NULL;
	unsigned char* pbyInBuf = NULL;
	unsigned int dwReadCount = 0;
	char *configPath = NULL;
	unsigned char* pbyOutYUV = NULL;
	unsigned int stride_w, stride_h;
	unsigned int bDoDMA = 0;
	VMF_DMA_DESCRIPTOR_T* pDesc = NULL;
	VMF_DMA_HANDLE_T* hDma = NULL;
    VMF_DMA_ADDR_T dma_addr;
	unsigned int bOutYuv = 0;
    unsigned int bFirstRead = 1;

    memset(&dma_addr, 0, sizeof(VMF_DMA_ADDR_T));
    
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

	if (0 != loadConfig(configPath)
		|| (g_bWriteFile && !g_pszOutputPath) || !g_pszInputPath) {
		goto EXIT;
	}

    if ((pfInput=fopen(g_pszInputPath, "rb")) == NULL) {
		printf("Open input bitstream file fail !!\n");
		goto EXIT;
	}

	if (g_bWriteFile && (pfOutput=fopen(g_pszOutputPath, "wb")) == NULL) {
		printf("Open output YUV file fail !!\n");
		goto EXIT;
	}
	
	//dwYSize = VMF_32_ALIGN(g_dwWidth)*VMF_16_ALIGN(g_dwHeight);
	//dwYuvSize = (dwYSize*3) >> 1;
	VMF_VDEC_INITOPT_T vdec_opt;
	memset(&vdec_opt, 0, sizeof(VMF_VDEC_INITOPT_T));

    vdec_opt.eCodecType = g_eCodec;
    vdec_opt.dwStreamSize = STREAM_FEEDING_SIZE;
	ptH26xDecoder = VMF_VDEC_Init(&vdec_opt);
	if(ptH26xDecoder == NULL) {
		printf("h26x decoder init failed \n");
		goto EXIT;
	}

	ptH26xState = VMF_VDEC_GetState(ptH26xDecoder);
	pbyInBuf = (unsigned char*)malloc(STREAM_FEEDING_SIZE);
	if (!pbyInBuf) {
		printf("Allocate bit stream buffer fail !! Size = %d\n", STREAM_FEEDING_SIZE);
		goto EXIT;
	}
	ptH26xState->tStreamBuf.ulVirtAddr = addr2uint(pbyInBuf);

	stride_w = VMF_32_ALIGN(g_dwWidth);
	stride_h = VMF_16_ALIGN(g_dwHeight); 
	if (g_dwWidth != stride_w || g_dwHeight != stride_h) {
		bDoDMA = 1;
	}

	if(bDoDMA) {
		pbyOutYUV = (unsigned char *)MemBroker_GetMemory((g_dwWidth * g_dwHeight * 3 / 2), VMF_ALIGN_TYPE_128_BYTE);
		if (!pbyOutYUV) {
			printf("Allocate output YUV buffer fail !!\n" );
			goto EXIT;
		}
        
		VMF_DMA_2DCF_INIT_T init;
		memset(&init, 0, sizeof(VMF_DMA_2DCF_INIT_T));
		init.dwProcessCbCr = 1;
		pDesc = VMF_DMA_Descriptor_Create(DMA_2D, &init);
	
		hDma = VMF_DMA_Init(1,128);
	}

	while (1) 
	{
        ptH26xState->tStreamBuf.dwSize = 0;            
        if(bFirstRead || VMF_DEC_EMPTY == ptH26xState->eResult)
        {
            bFirstRead = 0;
            dwReadCount = fread(pbyInBuf, sizeof(unsigned char), FEEDING_SIZE, pfInput);
            if (dwReadCount == 0) { 
                ptH26xState->bEndOfBitstream = 1;
            }   
            ptH26xState->tStreamBuf.dwSize = dwReadCount;
        }

		dwRet = VMF_VDEC_ProcessOneFrame(ptH26xDecoder);

        if (0 == dwRet)
		{
			//! Check result
			switch(ptH26xState->eResult) {
				case VMF_DEC_OK:
					bOutYuv = 1;
					break;
				case VMF_DEC_EMPTY: //! Need more input bitstream
					bOutYuv = 0;
					break;
				case VMF_DEC_ERROR:
				default: {
					printf("H26xDec OneFrame OK but result is error.(%d) \n", ptH26xState->eResult);
					goto EXIT;
				}
			}
		} else {
		    if (!ptH26xState->bEndOfBitstream) {
    			printf("H26xDec OneFrame failed Error result is %d \n", ptH26xState->eResult);
		    }
            goto EXIT;
		}
		
		if (bOutYuv && g_bWriteFile) {
			if (!bDoDMA) {
				fwrite((unsigned char*)ptH26xState->tFrameBuf.ulVirtYAddr , sizeof(unsigned char),  ptH26xState->tFrameBuf.dwSize, pfOutput);
			} else {
			    //update source and destination address		
        		dma_addr.dwCopyWidth = g_dwWidth;
        		dma_addr.dwCopyHeight = g_dwHeight;
        		dma_addr.pbySrcYPhysAddr = (unsigned char*)ptH26xState->tFrameBuf.ulPhysYAddr;
        		dma_addr.pbySrcCbPhysAddr = (unsigned char*)ptH26xState->tFrameBuf.ulPhysCbAddr;
        		dma_addr.pbySrcCrPhysAddr = (unsigned char*)ptH26xState->tFrameBuf.ulPhysCrAddr;
        		dma_addr.dwSrcStride = ptH26xState->tFrameBuf.dwStride;
        		
        		dma_addr.pbyDstYPhysAddr = (unsigned char*)(unsigned char *)MemBroker_GetPhysAddr(pbyOutYUV);
        		dma_addr.pbyDstCbPhysAddr = dma_addr.pbyDstYPhysAddr + g_dwWidth* g_dwHeight;
        		dma_addr.pbyDstCrPhysAddr = dma_addr.pbyDstCbPhysAddr + g_dwWidth* g_dwHeight/4;
        		dma_addr.dwDstStride = g_dwWidth;
	
		        VMF_DMA_Descriptor_Update_Addr(pDesc, &dma_addr);
		        VMF_DMA_Setup(hDma, &pDesc, 1);
				VMF_DMA_Process(hDma);
				fwrite(pbyOutYUV, sizeof(unsigned char), g_dwWidth*g_dwHeight*3/2, pfOutput);				
			}
			fflush(pfOutput);
			sync();
		}
	}

EXIT:

	if (pbyInBuf)
		free(pbyInBuf);

	if (pbyOutYUV) {
		MemBroker_FreeMemory(pbyOutYUV);
	}
	
	if (pfOutput) {
		fclose(pfOutput);
	}
	
	if (pfInput) {
		fclose(pfInput);
	}

	if (ptH26xDecoder) {
		VMF_VDEC_Release(ptH26xDecoder);
	}

	if (bDoDMA) {
		//step 5a: free DMA handle
		VMF_DMA_Release(hDma);
		//step 5b: free DMA descriptor
		VMF_DMA_Descriptor_Destroy(pDesc);
	}

	if (g_pszInputPath) {
		free(g_pszInputPath);
	}

	if(g_pszOutputPath) {
		free(g_pszOutputPath);
	}

	if (configPath) 
		free(configPath);

	return 0;
}

