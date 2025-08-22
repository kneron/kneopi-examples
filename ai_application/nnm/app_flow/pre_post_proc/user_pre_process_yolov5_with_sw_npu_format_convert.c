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

#include "user_pre_process_yolov5_with_sw_npu_format_convert.h"
#include "vmf_nnm_inference_app.h"
#include "model_type.h"
#include "user_utils.h"

static VMF_NNM_MODEL_TENSOR_DESCRIPTOR_T tensor_descriptor  = {0};
static uint32_t onnx_data_channel                           = 0;
static uint32_t onnx_data_height                            = 0;
static uint32_t onnx_data_width                             = 0;
static uint32_t onnx_data_buf_size                          = 0;
static float* onnx_data_buf                                 = NULL;

int user_pre_process_yolov5_with_sw_npu_format_convert_init()
{
    int ret = KP_SUCCESS;
    
    VMF_NNM_MODEL_TENSOR_INFO_V2_T *tensor_info = NULL;

    if (NULL == onnx_data_buf) {
        /* get YOLOv5 model input shape */
        ret = VMF_NNM_MODEL_Get_Input_Tensor_Descriptor(KNERON_YOLOV5S_COCO80_640_640_3, 0, &tensor_descriptor);
        if (0 == ret) {
            ret = KP_FW_GET_MODEL_INFO_FAILED_109;
            goto FUNC_OUT;
        } else {
            ret = KP_SUCCESS;
        }

        if (NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_2 != tensor_descriptor.tensor_shape_info.version) {
            ret = KP_FW_ERROR_UNSUPPORT_TOOLCHAIN_VERSION_136;
            goto FUNC_OUT;
        }

        /* get model input size (shape order: BxCxHxW) (KL730 only support kp_tensor_shape_info_v2_t) */
        tensor_info         = &(tensor_descriptor.tensor_shape_info.tensor_shape_info_data.v2);
        onnx_data_channel   = tensor_info->shape[1];
        onnx_data_height    = tensor_info->shape[2];
        onnx_data_width     = tensor_info->shape[3];

        /* malloc origin model input data */
        onnx_data_buf_size  = onnx_data_width * onnx_data_height * onnx_data_channel;
        onnx_data_buf       = calloc(onnx_data_buf_size, sizeof(float));

        if (NULL == onnx_data_buf) {
            ret = KP_FW_DDR_MALLOC_FAILED_102;
            goto FUNC_OUT;
        }
    }

FUNC_OUT:
    return ret;
}

int user_pre_process_yolov5_with_sw_npu_format_convert_deinit()
{
    int ret = KP_SUCCESS;

    if (NULL != onnx_data_buf) {
        free(onnx_data_buf);
        onnx_data_buf = NULL;
    } else {
        ret = KP_FW_DDR_MALLOC_FAILED_102;
        goto FUNC_OUT;
    }

FUNC_OUT:
    return ret;
}

int user_pre_process_yolov5_with_sw_npu_format_convert(struct kdp_image_s *image_p, unsigned int index)
{
    int ret                                             = KP_SUCCESS;

    /* get image source information and model input information */
    unsigned int dwSrcWidth                             = RAW_INPUT_COL(image_p, index);
    unsigned int dwSrcHeight                            = RAW_INPUT_ROW(image_p, index);
    unsigned char *pSrcBuffer                           = (unsigned char*)RAW_IMAGE_MEM_ADDR(image_p, index);
    unsigned int dwSrcBufferSize                        = RAW_IMAGE_MEM_LEN(image_p, index);
    unsigned int dwSrcNormalization                     = RAW_FORMAT(image_p, index);

    unsigned int dwDstWidth                             = DIM_INPUT_COL(image_p, index);
    unsigned int dwDstHeight                            = DIM_INPUT_ROW(image_p, index);
    unsigned char *pDstBuffer                           = (unsigned char*)PREPROC_INPUT_MEM_ADDR(image_p, index);
    unsigned int dwDstBufferSize                        = PREPROC_INPUT_MEM_LEN(image_p, index);

    /* setting pre-process related image configuration */
    float fScaleWidth                                   = (float)(dwDstWidth - 1) / (float)(dwSrcWidth - 1);
    float fScaleHeight                                  = (float)(dwDstHeight - 1) / (float)(dwSrcHeight - 1);
    float fScale                                        = 0;
    unsigned int dwResizeWidth                          = 0;
    unsigned int dwResizeHeight                         = 0;
    unsigned int dwPadLeft                              = 0;
    unsigned int dwPadRight                             = 0;
    unsigned int dwPadTop                               = 0;
    unsigned int dwPadBottom                            = 0;
    unsigned int img_buf_rgba8888_offset                = 0;
    unsigned int onnx_data_buf_offset                   = 0;

    /* to set the image engine (IE) for hardware image processing, the process involves resizing, padding, and converting the color space */
    /* Note: In this example function, we only use IE for general image processing without any exceptional Kneron pre-processing support, such as hardware normalization. */
    VMF_NNM_IE_CONFIG_T ie_config                       = {0};

    ie_config.src_buffer_addr                           = pSrcBuffer;
    ie_config.src_buffer_size                           = dwSrcBufferSize;
    ie_config.src_format                                = RAW_FORMAT(image_p, index) & IMAGE_FORMAT_NPU;
    ie_config.src_width                                 = dwSrcWidth;
    ie_config.src_height                                = dwSrcHeight;

    ie_config.dst_buffer_addr                           = pDstBuffer;
    ie_config.dst_buffer_size                           = dwDstBufferSize;
    ie_config.dst_format                                = KP_IMAGE_FORMAT_RGBA8888;
    ie_config.dst_width                                 = dwDstWidth;
    ie_config.dst_height                                = dwDstHeight;
    ie_config.dst_angle                                 = 0;

    ie_config.enable_crop                               = false;

    ie_config.image_norm                                = KP_NORMALIZE_DISABLE; /* Note: Disable hardware normalization when using software quantization and NPU layout conversion */
    ie_config.image_resize                              = KP_RESIZE_ENABLE;
    ie_config.image_padding                             = KP_PADDING_CORNER;

    ret = VMF_NNM_IE_Execute(&ie_config);
    if (KP_SUCCESS != ret) {
        printf("VMF_NNM_IE_Execute failed\n");
        goto FUNC_OUT;
    }

    /* prepare origin model input data (re-layout 640x640x3 rgba8888 image to 1x3x640x640 model input data) */
    for (unsigned int channel = 0; channel < onnx_data_channel; channel++) {
        img_buf_rgba8888_offset = channel;

        for (unsigned int pixel = 0; pixel < onnx_data_height * onnx_data_width; pixel++) {
            onnx_data_buf[onnx_data_buf_offset + pixel] = (float)pDstBuffer[img_buf_rgba8888_offset];
            img_buf_rgba8888_offset += 4;
        }

        onnx_data_buf_offset += (onnx_data_height * onnx_data_width);
    }

    /* do software normalization process (assume KP_NORMALIZE_KNERON normalization) */
    /* ---------------------------------------------------------------------------- */
    /* kp_normalize_mode_t and HW normalization flag mapping                        */
    /* ---------------------------------------------------------------------------- */
    /* KP_NORMALIZE_KNERON:             IMAGE_FORMAT_SUB128                         */
    /* KP_NORMALIZE_TENSOR_FLOW:        IMAGE_FORMAT_SUB128                         */
    /* KP_NORMALIZE_CUSTOMIZED_SUB128:  IMAGE_FORMAT_SUB128                         */
    /* KP_NORMALIZE_YOLO:               IMAGE_FORMAT_RIGHT_SHIFT_ONE_BIT            */
    /* KP_NORMALIZE_CUSTOMIZED_DIV2:    IMAGE_FORMAT_RIGHT_SHIFT_ONE_BIT            */
    if (!(dwSrcNormalization & IMAGE_FORMAT_SUB128)) {
        printf("Unsupport normalization configuration\n");
        ret = KP_FW_NOT_SUPPORT_PREPROCESSING_108;
        goto FUNC_OUT;
    }

    for (unsigned int idx = 0; idx < onnx_data_buf_size; idx++) {
        onnx_data_buf[idx] = onnx_data_buf[idx] / 256.0 - 0.5f;
    }

    /* convert ONNX data to NPU data */
    ret = ex_convert_onnx_data_to_npu_data(&tensor_descriptor,          /* tensor information */
                                           onnx_data_buf,               /* tensor data in ONNX format */
                                           onnx_data_buf_size,          /* element number of onnx_data_buf */
                                           (int8_t **)&pDstBuffer,      /* tensor data in NPU format */
                                           (int32_t *)&dwDstBufferSize  /* data size of npu_data_buf */);
    if (KP_SUCCESS != ret) {
        printf("ex_convert_onnx_data_to_npu_data failed\n");
        goto FUNC_OUT;
    }

    /* setting pre-process related image configuration */
    if (fScaleWidth < fScaleHeight) {
        fScale = fScaleWidth;
    } else {
        fScale = fScaleHeight;
    }

    dwResizeWidth                                       = (dwSrcWidth - 1) * fScale + 1.5f;
    dwResizeHeight                                      = (dwSrcHeight - 1) * fScale + 1.5f;
    dwPadRight                                          = dwDstWidth - dwResizeWidth;
    dwPadBottom                                         = dwDstHeight - dwResizeHeight;

    RAW_PAD_TOP(image_p, index)                         = dwPadTop;
    RAW_PAD_BOTTOM(image_p, index)                      = dwPadBottom;
    RAW_PAD_LEFT(image_p, index)                        = dwPadLeft;
    RAW_PAD_RIGHT(image_p, index)                       = dwPadRight;
    RAW_CROP_TOP(image_p, index)                        = 0;
    RAW_CROP_LEFT(image_p, index)                       = 0;
    RAW_CROP_RIGHT(image_p, index)                      = 0;
    RAW_CROP_BOTTOM(image_p, index)                     = 0;
    RAW_SCALE_WIDTH(image_p, index)                     = (float)dwSrcWidth / (float)dwResizeWidth;
    RAW_SCALE_HEIGHT(image_p, index)                    = (float)dwSrcHeight / (float)dwResizeHeight;

FUNC_OUT:
    return ret;
}