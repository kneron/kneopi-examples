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
#ifndef USER_PRE_PROCESS_YOLOV5_WITH_SW_NPU_FORMAT_CONVERT_H
#define USER_PRE_PROCESS_YOLOV5_WITH_SW_NPU_FORMAT_CONVERT_H
#include "ncpu_gen_struct.h"

int user_pre_process_yolov5_with_sw_npu_format_convert_init();
int user_pre_process_yolov5_with_sw_npu_format_convert_deinit();
int user_pre_process_yolov5_with_sw_npu_format_convert(struct kdp_image_s *image_p, unsigned int index);

#endif
