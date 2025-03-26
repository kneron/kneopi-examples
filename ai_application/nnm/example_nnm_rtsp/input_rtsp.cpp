#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <unistd.h>

#include <sys/stat.h>
#include <signal.h>
#include <pthread.h>
#include <sys/time.h>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/bsf.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>

#include <mem_broker.h>
#include <iniparser/iniparser.h>
#include <mem_util.h>
#include <video_decoder.h>
#include <vector_dma.h>
}

#include "example_shared_struct.h"
#include "kp_struct.h"

extern bool _blDispatchRunning;
extern bool _blFifoqManagerRunning;

extern bool _blSendInfRunning;
extern bool _blResultRunning;
extern bool _blDisplayRunning;

bool _blImageRunning = true;

NNM_SHARED_INPUT_T _input_data = {0};
pthread_mutex_t _mutex_image = PTHREAD_MUTEX_INITIALIZER;

int open_rtsp_stream(const char *url, AVFormatContext **pAvFormatContext, AVCodecContext **pAvCodecContext, AVBSFContext **pAvBsfContext, int *pVideoStreamIndex)
{
    int Ret                                     = -1;
    bool blCameraOpened                         = false;

    AVCodec *pAvCodec                           = NULL;
    AVBitStreamFilter *pAvStreamFilter          = NULL;
    AVDictionary *pOptionsForAvFormatContext    = NULL;

    /* open the input file */
    *pAvFormatContext = avformat_alloc_context();

    av_dict_set(&pOptionsForAvFormatContext, "rtsp_transport", "tcp", 0);
    av_dict_set(&pOptionsForAvFormatContext, "max_delay", "5000000", 0);

    if (0 != avformat_open_input(pAvFormatContext, url, NULL, &pOptionsForAvFormatContext)) {
        fprintf(stderr, "Cannot open input file '%s'\n", url);
        goto FUNC_OUT;
    }

    if (0 > avformat_find_stream_info(*pAvFormatContext, NULL)) {
        fprintf(stderr, "Cannot find input stream information.\n");
        goto FUNC_OUT;
    }

    /* find the video stream information */
    Ret = av_find_best_stream(*pAvFormatContext, AVMEDIA_TYPE_VIDEO, -1, -1, (const AVCodec **)&pAvCodec, 0);
    if (0 > Ret) {
        fprintf(stderr, "Cannot find a video stream in the input file\n");
        return -1;
    }

    *pVideoStreamIndex = Ret;

    if (NULL == (*pAvCodecContext = avcodec_alloc_context3(pAvCodec)))
        return AVERROR(ENOMEM);

    if (0 > avcodec_parameters_to_context(*pAvCodecContext,
                                          (*pAvFormatContext)->streams[*pVideoStreamIndex]->codecpar)) {
        goto FUNC_OUT;
    }

    if (0 > avcodec_open2(*pAvCodecContext, pAvCodec, NULL)) {
        fprintf(stderr, "Failed to open codec for stream #%u\n", *pVideoStreamIndex);
        goto FUNC_OUT;
    }

    switch (pAvCodec->id) {
    case AV_CODEC_ID_H264:
        pAvStreamFilter = (AVBitStreamFilter *)av_bsf_get_by_name("h264_mp4toannexb");
        break;
    case AV_CODEC_ID_HEVC:
        pAvStreamFilter = (AVBitStreamFilter *)av_bsf_get_by_name("hevc_mp4toannexb");
        break;
    default:
        goto FUNC_OUT;
    }

    if (0 > av_bsf_alloc(pAvStreamFilter, pAvBsfContext)) {
        fprintf(stderr, "Failed to open Bsf Context\n");
        goto FUNC_OUT;
    }

    if (0 > avcodec_parameters_from_context((*pAvBsfContext)->par_in, *pAvCodecContext)) {
        fprintf(stderr, "Failed to set param to Bsf Context from codec context\n");
        goto FUNC_OUT;
    }

    if (0 > av_bsf_init(*pAvBsfContext)) {
        fprintf(stderr, "Failed to init Bsf Context\n");
        goto FUNC_OUT;
    }

    blCameraOpened = true;

FUNC_OUT:

    return blCameraOpened;
}

void *example_rtsp_input_thread(void *arg)
{
    EXAMPLE_RTSP_INIT_OPT_T *pInitOpt = (EXAMPLE_RTSP_INIT_OPT_T *)arg;

    int Ret = 0;
    unsigned int dwInferenceWidth = 0;
    unsigned int dwInferenceHeight = 0;

    AVFormatContext *fmt_ctx = NULL;
    AVCodecContext *codec_ctx = NULL;
    AVBSFContext *bsf_ctx = NULL;
    int video_stream_index = -1;
    AVPacket packet;
    AVPacket packet_filter;

    VMF_VDEC_INITOPT_T vdec_opt;
    VMF_VDEC_HANDLE_T *ptH26xDecoder = NULL;
	VMF_H26XDEC_STATE_T *ptH26xState = NULL;

	VMF_DMA_DESCRIPTOR_T* pDesc = NULL;
	VMF_DMA_HANDLE_T* hDma = NULL;
    VMF_DMA_ADDR_T dma_addr;

RE_INIT_LOOP:
    // Init RTSP Reader
    {
        if (true != open_rtsp_stream(pInitOpt->pszRtspURL, &fmt_ctx, &codec_ctx, &bsf_ctx, &video_stream_index)) {
            goto EXIT_FFMPEG_IMAGE_THREAD;
        }

        if ((AV_CODEC_ID_H264 != codec_ctx->codec_id) && (AV_CODEC_ID_H265 != codec_ctx->codec_id)) {
            printf("[Warning] Unsupported Video Codec!!!\n");
            goto EXIT_FFMPEG_IMAGE_THREAD;
        }

        dwInferenceWidth = codec_ctx->width;
        dwInferenceHeight = codec_ctx->height;
    }

    // Init Hardware Decoder
    {
        memset(&vdec_opt, 0, sizeof(VMF_VDEC_INITOPT_T));

        switch (codec_ctx->codec_id) {
        case AV_CODEC_ID_H264:
            vdec_opt.eCodecType = VMF_VDEC_CODEC_TYPE_H264;
            break;
        case AV_CODEC_ID_H265:
            vdec_opt.eCodecType = VMF_VDEC_CODEC_TYPE_H265;
            break;
        default:
            goto EXIT_FFMPEG_IMAGE_THREAD;
        }

        ptH26xDecoder = (NULL != ptH26xDecoder) ? ptH26xDecoder : VMF_VDEC_Init(&vdec_opt);
        if(NULL == ptH26xDecoder) {
            printf("h26x decoder init failed \n");
            goto EXIT_FFMPEG_IMAGE_THREAD;
        }

        ptH26xState = (NULL != ptH26xState) ? ptH26xState : (VMF_H26XDEC_STATE_T *)VMF_VDEC_GetState(ptH26xDecoder);

        ptH26xState->tStreamBuf.ulVirtAddr = (0 != ptH26xState->tStreamBuf.ulVirtAddr) ? ptH26xState->tStreamBuf.ulVirtAddr : addr2uint(MemBroker_GetMemory(512*1024, VMF_ALIGN_TYPE_DEFAULT));
        ptH26xState->tStreamBuf.ulPhysAddr = (unsigned long)MemBroker_GetPhysAddr((void*)ptH26xState->tStreamBuf.ulVirtAddr);

        VMF_DMA_2DCF_INIT_T init;
        memset(&init, 0, sizeof(VMF_DMA_2DCF_INIT_T));
        init.dwProcessCbCr = 1;
        pDesc = (NULL != pDesc) ? pDesc : VMF_DMA_Descriptor_Create(DMA_2D, &init);
        hDma = (NULL != hDma) ? hDma : VMF_DMA_Init(1,128);

        _input_data.input_buf_address = (0 != _input_data.input_buf_address) ? _input_data.input_buf_address : (uintptr_t)MemBroker_GetMemory((dwInferenceWidth * dwInferenceHeight * 1.5), VMF_ALIGN_TYPE_128_BYTE);
    }

    while (true == _blImageRunning)
    {
        Ret = av_read_frame(fmt_ctx, &packet);
        if (AVERROR_EOF == Ret) {
            avcodec_free_context(&codec_ctx);
            avformat_free_context(fmt_ctx);
            av_bsf_free(&bsf_ctx);

            goto RE_INIT_LOOP;
        } else if (0 > Ret) {
            printf("[FFmpge] AV read frame failed\n");
            goto EXIT_FFMPEG_IMAGE_THREAD;
        }

        if (packet.stream_index == video_stream_index) {
            if (0 > av_bsf_send_packet(bsf_ctx, &packet)) {
                printf("[FFmpge] BSF send packet failed\n");
                goto EXIT_FFMPEG_IMAGE_THREAD;
            }

            if (0 > av_bsf_receive_packet(bsf_ctx, &packet_filter)) {
                printf("[FFmpge] BSF receive packet failed\n");
                goto EXIT_FFMPEG_IMAGE_THREAD;
            }

            memcpy((void *)ptH26xState->tStreamBuf.ulVirtAddr, packet_filter.data, packet_filter.size);
            MemBroker_CacheFlush((void*)ptH26xState->tStreamBuf.ulVirtAddr, packet_filter.size);

            ptH26xState->tStreamBuf.dwSize = packet_filter.size;

            av_packet_unref(&packet);
            av_packet_unref(&packet_filter);
        } else {
            av_packet_unref(&packet);
            continue;
        }

        Ret = VMF_VDEC_ProcessOneFrame(ptH26xDecoder);
        if (ptH26xState->eResult != VMF_DEC_OK) {
            printf("H26xDec OneFrame Failed %d. \n", ptH26xState->eResult);
            goto EXIT_FFMPEG_IMAGE_THREAD;
        }

        pthread_mutex_lock(&_mutex_image);
        if (0 != _input_data.input_buf_address) {
            //update source and destination address
            dma_addr.dwCopyWidth = dwInferenceWidth;
            dma_addr.dwCopyHeight = dwInferenceHeight;
            dma_addr.pbySrcYPhysAddr = (unsigned char*)ptH26xState->tFrameBuf.ulPhysYAddr;
            dma_addr.pbySrcCbPhysAddr = (unsigned char*)ptH26xState->tFrameBuf.ulPhysCbAddr;
            dma_addr.pbySrcCrPhysAddr = (unsigned char*)ptH26xState->tFrameBuf.ulPhysCrAddr;
            dma_addr.dwSrcStride = ptH26xState->tFrameBuf.dwStride;

            dma_addr.pbyDstYPhysAddr = (unsigned char*)MemBroker_GetPhysAddr((void *)_input_data.input_buf_address);
            dma_addr.pbyDstCbPhysAddr = dma_addr.pbyDstYPhysAddr + dwInferenceWidth * dwInferenceHeight;
            dma_addr.pbyDstCrPhysAddr = dma_addr.pbyDstCbPhysAddr + dwInferenceWidth * dwInferenceHeight / 4;
            dma_addr.dwDstStride = dwInferenceWidth;

            VMF_DMA_Descriptor_Update_Addr(pDesc, &dma_addr);
            VMF_DMA_Setup(hDma, &pDesc, 1);
            VMF_DMA_Process(hDma);

            MemBroker_CacheCopyBack((void *)_input_data.input_buf_address, dwInferenceWidth * dwInferenceHeight * 1.5);

            _input_data.input_image_width = dwInferenceWidth;
            _input_data.input_image_height = dwInferenceHeight;
            _input_data.input_image_format = KP_IMAGE_FORMAT_YUV420;

            _input_data.input_buf_size = _input_data.input_image_width * _input_data.input_image_height * 1.5;
            _input_data.input_ready_inf = true;
        }
        pthread_mutex_unlock(&_mutex_image);
    }

EXIT_FFMPEG_IMAGE_THREAD:

    if (0 != _input_data.input_buf_address)
        MemBroker_FreeMemory((void *)_input_data.input_buf_address);

    if (0 != ptH26xState->tStreamBuf.ulVirtAddr)
        MemBroker_FreeMemory((void *)ptH26xState->tStreamBuf.ulVirtAddr);

    if (NULL != ptH26xDecoder)
        VMF_VDEC_Release(ptH26xDecoder);

    if (NULL != pDesc)
        VMF_DMA_Descriptor_Destroy(pDesc);

    if (NULL != hDma)
        VMF_DMA_Release(hDma);

    if (NULL != codec_ctx)
        avcodec_free_context(&codec_ctx);

    if (NULL != fmt_ctx)
       avformat_free_context(fmt_ctx);

    if (NULL != bsf_ctx)
        av_bsf_free(&bsf_ctx);

    _blSendInfRunning = false;
    _blResultRunning = false;
    _blDisplayRunning = false;

    _blDispatchRunning = false;
    _blFifoqManagerRunning = false;

    return NULL;
}