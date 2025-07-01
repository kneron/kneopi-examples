#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>
#include <signal.h>
#include <pthread.h>
#include <sys/time.h>

#include <opencv2/opencv.hpp>

extern "C" {
#include <mem_broker.h>
#include <msgbroker/msg_broker.h>
#include <video_source.h>
#include <video_bind.h>
#include <video_buf.h>
#include <sync_shared_memory.h>
#include <ssm_info.h>
#include <vmf_log.h>
#include <vector_dma.h>
#include <iniparser/iniparser.h>

#include "fec_api.h"
}

#include "example_shared_struct.h"
#include "kp_struct.h"

#define VENC_VSRC_PIN       "vsrc_ssm"                  //! VMF_VSRC Output pin
#define VENC_VSRC_C_PIN     "vsrc_ssm_c_0"              //! VMF_VSRC Customer Output pin
#define VENC_VSRC_B_PIN     "vsrc_ssm_b_0"              //! VMF_VSRC Draw Box Customer Output pin
#define VENC_RESOURCE_DIR   "./Resource/"               //! directory contains ISP, AE, AWB, AutoScene sub directory
#define VENC_CMD_FIFO       "/tmp/venc/c0/command.fifo" //! communicate with rtsps, vrec, etc.
#define SRB_DEFAULT_PREFIX  "venc_srb"

extern FECDefValue gFecDefValue;

static VMF_BIND_CONTEXT_T* g_ptBind = NULL;

VMF_SRC_CONNECT_INFO_T connect_info;
VMF_LAYOUT_T g_tLayout;
VMF_VSRC_HANDLE_T* g_ptVsrcHandle = NULL;
ssm_handle_t *gptSsmHandle;

extern bool _blDispatchRunning;
extern bool _blFifoqManagerRunning;

extern bool _blSendInfRunning;
extern bool _blResultRunning;
extern bool _blDisplayRunning;

bool _blImageRunning = true;

NNM_SHARED_INPUT_T _input_data = {0};
pthread_mutex_t _mutex_image = PTHREAD_MUTEX_INITIALIZER;

/******************************* Local functions Implementation ************************************/
typedef struct
{
    VMF_DMA_HANDLE_T* ptDmaHandle;
    VMF_DMA_DESCRIPTOR_T* ptDmaDesc;
} DMA_INFO_T;

DMA_INFO_T* dma2d_init()
{
    DMA_INFO_T* dma_info;
    dma_info = (DMA_INFO_T*)MemBroker_GetMemory(sizeof(DMA_INFO_T), VMF_ALIGN_TYPE_DEFAULT);
    if (NULL == dma_info) {
        printf("dma2d_init MemBroker_GetMemory failed\n");
        return NULL;
    }
    dma_info->ptDmaHandle = VMF_DMA_Init(1,128);
    if (NULL == dma_info->ptDmaHandle) {
        if(dma_info)
            MemBroker_FreeMemory(dma_info);
        printf("dma2d_init failed\n");
        return NULL;
    }
    VMF_DMA_2DCF_INIT_T init;
    memset(&init, 0, sizeof(VMF_DMA_2DCF_INIT_T));
    init.dwProcessCbCr = 1;
    dma_info->ptDmaDesc = VMF_DMA_Descriptor_Create(DMA_2D, &init);
    if (NULL == dma_info->ptDmaDesc) {
        if(dma_info->ptDmaHandle)
            VMF_DMA_Release(dma_info->ptDmaHandle);
        if(dma_info)
            MemBroker_FreeMemory(dma_info);
        printf("dma_desc init failed\n");
    }
    return dma_info;
}

void dma2d_release(DMA_INFO_T* dma_info)
{
    VMF_DMA_Descriptor_Destroy(dma_info->ptDmaDesc);
    VMF_DMA_Release(dma_info->ptDmaHandle);
    MemBroker_FreeMemory(dma_info);
}

int dma2d_copy(VMF_DMA_HANDLE_T* dma_handle, VMF_DMA_DESCRIPTOR_T* dma_desc, void* dest, void* source, VMF_VSRC_SSM_OUTPUT_INFO_T* vsrc_ssm_info)
{
    int ret = 0;
    VMF_DMA_ADDR_T dma_addr;
    memset(&dma_addr, 0, sizeof(VMF_DMA_ADDR_T));
    //MemBroker_CacheCopyBack(source, vsrc_ssm_info->dwWidth * vsrc_ssm_info->dwHeight * 1.5);//Reading source data from ssm buffer doesn't need using CacheCopyBack.
    dma_addr.dwCopyWidth = vsrc_ssm_info->dwWidth;
    dma_addr.dwCopyHeight = vsrc_ssm_info->dwHeight;
    dma_addr.pbySrcYPhysAddr = (unsigned char*)MemBroker_GetPhysAddr(source) + vsrc_ssm_info->dwOffset[0];
    dma_addr.pbySrcCbPhysAddr = (unsigned char*)MemBroker_GetPhysAddr(source) + vsrc_ssm_info->dwOffset[1];
    dma_addr.pbySrcCrPhysAddr = (unsigned char*)MemBroker_GetPhysAddr(source) + vsrc_ssm_info->dwOffset[2];
    dma_addr.dwSrcStride = vsrc_ssm_info->dwYStride;

    dma_addr.pbyDstYPhysAddr = (unsigned char*)MemBroker_GetPhysAddr(dest);
    dma_addr.pbyDstCbPhysAddr = dma_addr.pbyDstYPhysAddr + vsrc_ssm_info->dwWidth* vsrc_ssm_info->dwHeight;
    dma_addr.pbyDstCrPhysAddr = dma_addr.pbyDstCbPhysAddr + vsrc_ssm_info->dwWidth* vsrc_ssm_info->dwHeight/4;
    dma_addr.dwDstStride = vsrc_ssm_info->dwWidth;
    MemBroker_CacheFlush(dest, vsrc_ssm_info->dwWidth * vsrc_ssm_info->dwHeight * 1.5);
    ret |= VMF_DMA_Descriptor_Update_Addr(dma_desc, &dma_addr);
    ret |= VMF_DMA_Setup(dma_handle, &dma_desc, 1);
    ret |= VMF_DMA_Process(dma_handle);
    return ret;
}

/* ========================= sensor related ================================ */

static void release_video_source(VMF_VSRC_HANDLE_T* ptVsrcHandle)
{
    VMF_VSRC_Stop(ptVsrcHandle);
    VMF_VSRC_Release(ptVsrcHandle);
}

/* This function is called after video source is initialized and first video frame is captured in buffer */
static void vsrc_init_callback(unsigned int width, unsigned int height)
{
    memset(&g_tLayout, 0, sizeof(VMF_LAYOUT_T));
    g_tLayout.dwCanvasWidth = width;
    g_tLayout.dwCanvasHeight = height;
    g_tLayout.dwVideoPosX = 0;
    g_tLayout.dwVideoPosY = 0;
    g_tLayout.dwVideoWidth = width;
    g_tLayout.dwVideoHeight = height;
}

void set_eis(VMF_EIS_INIT_T* ptEisInit)
{
    EIS_T* ptEis = &(gFecDefValue.tEis);

    if (!ptEisInit) {
        printf("[%s] Err: eis init is null.\n", __func__);
        return;
    }

    ptEisInit->pszLensCurveNodesPath = ptEis->pszCurveNodesPath;
    ptEisInit->pszLogPath = NULL;
    ptEisInit->fGyroDataGain = ptEis->fGyroDataGain;
    ptEisInit->dwGridSection = ptEis->dwGridSection;
    ptEisInit->dwMaxGridSection = (ptEis->dwMaxGridSection >= ptEis->dwGridSection)?ptEis->dwMaxGridSection:ptEis->dwGridSection;
    ptEisInit->fCropRatio = ptEis->fCropRatio;
    ptEisInit->dwImageType = ptEis->dwImageType;
    ptEisInit->dwProcessMode = ptEis->dwProcessMode;
    ptEisInit->dwCoordinateTransform[0] = ptEis->dwCoordinateTransform[0];
    ptEisInit->dwCoordinateTransform[1] = ptEis->dwCoordinateTransform[1];
    ptEisInit->dwCoordinateTransform[2] = ptEis->dwCoordinateTransform[2];
    ptEisInit->sqwTimeOffset = ptEis->sqwTimeOffset;
    ptEisInit->bImageRotate180 = ptEis->bImageRotate180;
    ptEisInit->sdwReadoutTimeOffset = ptEis->sdwReadoutTimeOffset;
    ptEisInit->fReadoutTimeRatio = ptEis->fReadoutTimeRatio;
    ptEisInit->bForceOriRs = ptEis->bForceOriRs;
    memcpy(&ptEisInit->tGyroConfig, &ptEis->tGyroConfig, sizeof(VMF_VSRC_GYRO_CONFIG_T));
}

static int init_video_source(EXAMPLE_SENSOR_INIT_OPT_T* pExampleSensorInit)
{
    VMF_VSRC_INITOPT_T tVsrcInitOpt;
    VMF_VSRC_FRONTEND_CONFIG_T tVsrcFrontendConfig;

    memset(&tVsrcInitOpt, 0, sizeof(VMF_VSRC_INITOPT_T));
    memset(&tVsrcFrontendConfig, 0, sizeof(VMF_VSRC_FRONTEND_CONFIG_T));

    if (pExampleSensorInit->tSensorConf.dwFecMode != 0) {
        if (loadCalibrateConfig(&pExampleSensorInit->tSensorConf, &tVsrcFrontendConfig) == -1) {
            printf("[%s] Err: no calibrate config\n", __func__);
            return -1;
        }
        if (loadFECConfig(&pExampleSensorInit->tSensorConf, &tVsrcFrontendConfig) == -1) {
            printf("[%s] Err: no fec config\n", __func__);
            return -1;
        }
    } else {
        if (loadFECConfig(&pExampleSensorInit->tSensorConf, &tVsrcFrontendConfig) == -1) {
            printf("[%s] Err: no fec config\n", __func__);
            return -1;
        }
    }
    tVsrcFrontendConfig.tFecInitConfig.eFecMethod = VMF_FEC_METHOD_GTR;
    //! Fusion or Normal mode
    if (pExampleSensorInit->tSensorConf.pszFusionConfigPath) {
        tVsrcFrontendConfig.dwSensorConfigCount = 2;
        tVsrcFrontendConfig.apszSensorConfig[0] = pExampleSensorInit->tSensorConf.pszSensorConfigPath;
        tVsrcFrontendConfig.apszSensorConfig[1] = pExampleSensorInit->tSensorConf.pszFusionConfigPath;
        tVsrcInitOpt.eAppMode = VMF_VSRC_APP_MODE_FUSION;
        printf("[%s] Fusion Mode\n", __func__);
    } else {
        tVsrcFrontendConfig.dwSensorConfigCount = 1;
        tVsrcFrontendConfig.apszSensorConfig[0] = pExampleSensorInit->tSensorConf.pszSensorConfigPath;
        tVsrcInitOpt.eAppMode = VMF_VSRC_APP_MODE_NORMAL;
        printf("[%s] Normal Mode\n", __func__);
    }

    tVsrcInitOpt.dwFrontConfigCount = 1;
    tVsrcInitOpt.ptFrontConfig = &tVsrcFrontendConfig;
    tVsrcInitOpt.pszAutoSceneConfig = pExampleSensorInit->tSensorConf.pszAutoSceneConfigPath;
    tVsrcInitOpt.pszOutPinPrefix = VENC_VSRC_PIN;

    tVsrcInitOpt.fnInitCallback = vsrc_init_callback;
    tVsrcInitOpt.pszResourceDir = VENC_RESOURCE_DIR;
    tVsrcInitOpt.tEngConfig.dwIfpOutDataType = 1;
    tVsrcInitOpt.tEngConfig.dwIspOutDataType = 1;

    //! Fill EIS configuration
    if (pExampleSensorInit->dwEisEnable == 1) {
        tVsrcFrontendConfig.tFecInitConfig.ptEisInit = (VMF_EIS_INIT_T *)calloc(1, sizeof(VMF_EIS_INIT_T));
        if (!tVsrcFrontendConfig.tFecInitConfig.ptEisInit) {
            printf("[%s] calloc VMF_EIS_INIT_T fail.\n", __func__);
            pExampleSensorInit->dwEisEnable = 0;
        } else {
            set_eis(tVsrcFrontendConfig.tFecInitConfig.ptEisInit);
        }
    }

    g_ptVsrcHandle = VMF_VSRC_Init(&tVsrcInitOpt);
    if (!g_ptVsrcHandle) {
        printf("[%s] VMF_VSRC_Init failed!\n", __func__);
        return -1;
    }

    if (VMF_VSRC_Start(g_ptVsrcHandle, NULL) != 0) {
        printf("[%s] VMF_VSRC_Start failed!\n", __func__);
        release_video_source(g_ptVsrcHandle);
        return -1;
    }

    return setup_fec_mode(g_ptVsrcHandle, &g_tLayout, (FEC_MODE)pExampleSensorInit->tSensorConf.dwFecMode, pExampleSensorInit->dwEisEnable);
}

static VMF_BIND_CONTEXT_T* init_bind(void* hVideoSource)
{
    //! init video bind
    VMF_BIND_INITOPT_T tBindOpt;
    memset(&tBindOpt, 0, sizeof(VMF_BIND_INITOPT_T));
    tBindOpt.dwSrcOutputIndex = 0;
    tBindOpt.ptSrcHandle = hVideoSource;
    tBindOpt.pfnQueryFunc = (VMF_BIND_QUERY_FUNC) VMF_VSRC_GetInfo;
    tBindOpt.pfnIspFunc = (VMF_BIND_CONFIG_ISP_FUNC) VMF_VSRC_ConfigISP;
    return VMF_BIND_Init(&tBindOpt);
}

static void release_bind(VMF_BIND_CONTEXT_T* pBind)
{
    VMF_BIND_Release(pBind);
}

void *example_sensor_image_thread(void *arg)
{
    EXAMPLE_SENSOR_INIT_OPT_T *pExampleSensorInit = (EXAMPLE_SENSOR_INIT_OPT_T*)arg;

    //Video source initialization
    unsigned int dwInferenceWidth = pExampleSensorInit->dwImageWidth;
    unsigned int dwInferenceHeight = pExampleSensorInit->dwImageHeight;
    ssm_handle_t *ptSsmHandle = NULL;
    ssm_buffer_t ssm_buf;
    const unsigned int getyuv_retry_cnt = 500;
    unsigned int wait_cnt = 0;
    char azReaderSsmName[64];
    char tmpName[30] = {0};

    VMF_SSM_READER_SCHEME eImageBufMode = (VMF_SSM_READER_SCHEME)pExampleSensorInit->dwGetImageBufMode;
    DMA_INFO_T *pDmaInfo = dma2d_init();

    if (NULL == pDmaInfo) {
        printf("init dma failed\n");
        goto EXIT_SENSOR_IMAGE_THREAD;
    }

    if (init_video_source(pExampleSensorInit)) {
        printf("init_video_source failed!!\n");
        goto EXIT_SENSOR_IMAGE_THREAD;
    }

    /* initializing the binder associated with the video source */
    if ((g_ptBind = init_bind(g_ptVsrcHandle)) == NULL) {
        printf("init_bind failed!!\n");
        release_video_source(g_ptVsrcHandle);
        goto EXIT_SENSOR_IMAGE_THREAD;
    }

    VMF_BIND_Request(g_ptBind, dwInferenceWidth, dwInferenceHeight, dwInferenceWidth, 0, &connect_info);

    gptSsmHandle = ptSsmHandle = SSM_Reader_Init(connect_info.szSrcPin);

    _input_data.input_buf_address = (uintptr_t)MemBroker_GetMemory(connect_info.dwSrcWidth * connect_info.dwSrcWidth * 1.5, VMF_ALIGN_TYPE_128_BYTE);

    if (!ptSsmHandle) {
        printf("%s() failed, SSM_Reader_Init failed!\n", __func__);
    }

    //! wait for video source being ready.
    memset(&ssm_buf, 0, sizeof(ssm_buffer_t));
    while (SSM_Reader_ReturnReceiveNewestBuff(ptSsmHandle, &ssm_buf, eImageBufMode) < 0) {
        if ((getyuv_retry_cnt < wait_cnt++) || (false == _blImageRunning)) {
            printf("%s err: get video yuv failed, exit\n", __func__);
            goto EXIT_SENSOR_IMAGE_THREAD;
        }
    }

    // run infinitely
    while (true == _blImageRunning) {
        SSM_Reader_ReturnReceiveNewestBuff(ptSsmHandle, &ssm_buf, eImageBufMode);//VMF_SSM_READER_BLOCK / VMF_SSM_READER_NONBLOCK
        VMF_VSRC_SSM_OUTPUT_INFO_T vsrc_ssm_info;
        VMF_VSRC_SSM_GetInfo(ssm_buf.buffer, &vsrc_ssm_info);
        if (!ssm_buf.buffer) {
            printf("[%s] ssm_buf.buffer = %p, EXIT_SENSOR_IMAGE_THREAD \n", __func__, ssm_buf.buffer);
            goto EXIT_SENSOR_IMAGE_THREAD;
        }

        pthread_mutex_lock(&_mutex_image);

        dma2d_copy(pDmaInfo->ptDmaHandle, pDmaInfo->ptDmaDesc, (void *)_input_data.input_buf_address, ssm_buf.buffer, &vsrc_ssm_info);

        _input_data.input_image_width = vsrc_ssm_info.dwWidth;
        _input_data.input_image_height = vsrc_ssm_info.dwHeight;
        _input_data.input_image_format = KP_IMAGE_FORMAT_YUV420;

        _input_data.input_buf_size = _input_data.input_image_width * _input_data.input_image_height * 1.5;
        _input_data.input_ready_inf = true;

        pthread_mutex_unlock(&_mutex_image);
    }

EXIT_SENSOR_IMAGE_THREAD:

    if (ptSsmHandle) {
        SSM_Reader_ReturnBuff(ptSsmHandle, &ssm_buf);
        SSM_Release(ptSsmHandle);
    }

    if (g_ptBind)
        release_bind(g_ptBind);

    if (g_ptVsrcHandle)
        release_video_source(g_ptVsrcHandle);

    if (0 != _input_data.input_buf_address)
        MemBroker_FreeMemory((void *)_input_data.input_buf_address);

    if (NULL != pDmaInfo)
        dma2d_release(pDmaInfo);

    _blSendInfRunning = false;
    _blResultRunning = false;
    _blDisplayRunning = false;

    _blDispatchRunning = false;
    _blFifoqManagerRunning = false;

    return NULL;
}
