/*
 * memory manager headers for the pre/post-process.
 *
 * Copyright (C) 2023 Kneron, Inc. All rights reserved.
 *
 */
#ifndef USER_POST_MEM_MANAGER_H_
#define USER_POST_MEM_MANAGER_H_

#include <stdlib.h>

#include "base.h"

/******************************************************************
 * public functions
*******************************************************************/

void* ex_get_gp(void **gp, size_t len);
void ex_free_gp(void **gp);

#endif  // USER_POST_MEM_MANAGER_H_
