/*
 *******************************************************************************
 *  Copyright (c) 2010-2025 VATICS(KNERON) Inc. All rights reserved.
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
#include "user_pre_process_yolov5.h"
#include "vmf_nnm_inference_app.h"

int user_pre_process_yolov5(struct kdp_image_s *image_p, unsigned int index)
{
    unsigned int dwSrcWidth  = RAW_INPUT_COL(image_p, index);
    unsigned int dwSrcHeight = RAW_INPUT_ROW(image_p, index);
    unsigned char *pSrcBuffer = (unsigned char*)RAW_IMAGE_MEM_ADDR(image_p, index);
    unsigned int dwSrcBufferSize = RAW_IMAGE_MEM_LEN(image_p, index);

    unsigned int dwDstWidth  = DIM_INPUT_COL(image_p, index);
    unsigned int dwDstHeight = DIM_INPUT_ROW(image_p, index);
    unsigned char *pDstBuffer = (unsigned char*)PREPROC_INPUT_MEM_ADDR(image_p, index);
    unsigned int dwDstBufferSize = PREPROC_INPUT_MEM_LEN(image_p, index);
    VMF_NNM_IE_CONFIG_T ie_config = {0};

    /* setting pre-process related image configuration */
    float fScaleWidth = (float)(dwDstWidth - 1)/(float)(dwSrcWidth - 1);
    float fScaleHeight = (float)(dwDstHeight - 1)/(float)(dwSrcHeight - 1);
    float fScale = 0;
    unsigned int dwResizeWidth = 0;
    unsigned int dwResizeHeight = 0;
    unsigned int dwPadLeft = 0;
    unsigned int dwPadRight = 0;
    unsigned int dwPadTop = 0;
    unsigned int dwPadBottom = 0;

    ie_config.src_buffer_addr = pSrcBuffer;
    ie_config.src_buffer_size = dwSrcBufferSize;
    ie_config.src_format = RAW_FORMAT(image_p, index) & IMAGE_FORMAT_NPU;
    ie_config.src_width = dwSrcWidth;
    ie_config.src_height = dwSrcHeight;

    ie_config.dst_buffer_addr = pDstBuffer;
    ie_config.dst_buffer_size = dwDstBufferSize;
    ie_config.dst_format = KP_IMAGE_FORMAT_RGBA8888;
    ie_config.dst_width = dwDstWidth;
    ie_config.dst_height = dwDstHeight;
    ie_config.dst_angle = 0;

    ie_config.enable_crop = false;

    ie_config.image_norm = KP_NORMALIZE_KNERON;
    ie_config.image_resize = KP_RESIZE_ENABLE;
    ie_config.image_padding = KP_PADDING_CORNER;

    if (fScaleWidth < fScaleHeight) {
        fScale = fScaleWidth;
    } else {
        fScale = fScaleHeight;
    }
    dwResizeWidth = (dwSrcWidth - 1) * fScale + 1.5f;
    dwResizeHeight = (dwSrcHeight - 1) * fScale + 1.5f ;
    dwPadRight = dwDstWidth - dwResizeWidth;
    dwPadBottom = dwDstHeight - dwResizeHeight;

    RAW_PAD_TOP(image_p, index)         = dwPadTop;
    RAW_PAD_BOTTOM(image_p, index)      = dwPadBottom;
    RAW_PAD_LEFT(image_p, index)        = dwPadLeft;
    RAW_PAD_RIGHT(image_p, index)       = dwPadRight;
    RAW_CROP_TOP(image_p, index)        = 0;
    RAW_CROP_LEFT(image_p, index)       = 0;
    RAW_CROP_RIGHT(image_p, index)      = 0;
    RAW_CROP_BOTTOM(image_p, index)     = 0;

    RAW_SCALE_WIDTH(image_p, index)    = (float)dwSrcWidth/(float)dwResizeWidth;
    RAW_SCALE_HEIGHT(image_p, index)   = (float)dwSrcHeight/(float)dwResizeHeight;

    if (VMF_NNM_IE_Execute(&ie_config)) {
        printf("VMF_NNM_IE_Execute failed\n");
    }

    return 0;
}