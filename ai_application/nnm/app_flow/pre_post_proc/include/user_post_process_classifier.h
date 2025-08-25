/*
 * User Post Processing function for yolov5
 *
 * Copyright (C) 2021 Kneron, Inc. All rights reserved.
 *
 */
#ifndef USER_POST_PROCESS_CLASSIFIER_H
#define USER_POST_PROCESS_CLASSIFIER_H

#include "ncpu_gen_struct.h"
#include "user_utils.h"

#define EX_CLASSIFIER_MAX_SIZE_TOP_N (1000)                                     /**< max number of top N classification result */

/**
 * @brief describe a yolo pre-post process configurations for yolo series
 */
struct ex_classifier_result_s {
    int32_t class_num;                                                          /**< class number (index) */
    float score;                                                                /**< score for class number */
};

/**
 * @brief describe a yolo pre-post process configurations for yolo series
 */
struct ex_classifier_top_n_result_s {
    uint32_t top_n_num;                                                         /**< number of top N */
    struct ex_classifier_result_s top_n_results[EX_CLASSIFIER_MAX_SIZE_TOP_N];  /**< array of top N classifier results */
};

int user_post_classifier_top_n(int model_id, struct kdp_image_s *image_p);
#endif
