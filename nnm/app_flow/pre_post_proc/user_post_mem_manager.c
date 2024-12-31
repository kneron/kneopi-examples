/*
 * Utility functions for the postprocess functions.
 *
 * Copyright (C) 2023 Kneron, Inc. All rights reserved.
 *
 */
#include <stdio.h>

#include "user_post_mem_manager.h"

/******************************************************************
 * main function
*******************************************************************/
void* ex_get_gp(void **gp, size_t len)
{
    if (*gp == NULL) {
        *gp = realloc(*gp, len);
        if (*gp == NULL)
            printf("ncpu: ERROR: kneron_realloc %lu in get_gp()\n", (long unsigned int)len);
    }

    return *gp;
}

void ex_free_gp(void **gp)
{
    #ifdef FREE
        if (NULL != *gp) {
            free(*gp);
            *gp = NULL;
        }
    #else
        (void)gp;
    #endif

    return;
}
