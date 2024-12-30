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

#ifndef EXAMPLE_SHARED_STRUCT_H
#define EXAMPLE_SHARED_STRUCT_H

#include "kp_struct.h"

/**
 * @brief describe example webcam configuration
 */
typedef struct
{
    char* pszModelPath;             //! Path of NN model file
    unsigned int dwJobId;           //e.g. KDP2_INF_ID_APP_YOLO;
    unsigned int dwModelId;
    float fThreshold;               // for post process

    unsigned int dwImageWidth;      //! Input image width
    unsigned int dwImageHeight;     //! Input image height

    char* pszImageName;
    char* pszImageFormat;           //! Input image format
    float fFps;                     // feed image fps
    unsigned int dwLoopTime;        //! Inference loop time
} EXAMPLE_IMAGE_INIT_OPT_T;

/**
 * @brief describe shared image structure between threads
 */
typedef struct {
    bool input_ready_inf;

    uintptr_t input_buf_address;
    unsigned int input_buf_size;

    /* Used when input is image */
    int input_image_width;
    int input_image_height;
    int input_image_format;
} NNM_SHARED_INPUT_T;

/**
 * @brief describe shared result structure between threads
 */
typedef struct {
    int result_buffer_size;
    uintptr_t result_buffer;
    bool result_ready_display;
} NNM_SHARED_RESULT_T;

#endif  // EXAMPLE_SHARED_STRUCT_H