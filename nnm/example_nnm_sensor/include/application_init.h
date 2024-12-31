
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

/**@addtogroup  APPLICATION_INIT
 * @{
 * @brief       Kneron application init
 * @copyright   Copyright (C) 2022 Kneron, Inc. All rights reserved.
 */
#ifndef KDP2_APP_INFERENCE_INIT_H
#define KDP2_APP_INFERENCE_INIT_H

#include "kp_struct.h"

/**
 * @brief       Add application layer initialization code
 *
 * @return      void
 */
void app_initialize(void);

/**
 * @brief Add application layer destory code
 */
void app_destroy(void);

#endif  //KDP2_APP_INFERENCE_INIT_H
