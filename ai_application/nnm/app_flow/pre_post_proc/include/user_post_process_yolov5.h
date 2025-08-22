/*
 * User Post Processing function for yolov5
 *
 * Copyright (C) 2021 Kneron, Inc. All rights reserved.
 *
 */
#ifndef USER_POST_PROCESS_YOLOV5_H
#define USER_POST_PROCESS_YOLOV5_H

#include "ncpu_gen_struct.h"
#include "user_utils.h"

#define EX_MAX_YOLO_ANCHOR_LAYER_NUM            (5)                                                                 /**< max number of layer for anchor */
#define EX_MAX_YOLO_ANCHOR_CELL_NUM_PER_LAYER   (3)                                                                 /**< max number of anchor per gride cell */
#define EX_BOXES_MAX_NUM                        (500)                                                               /**< max number of bounding box */

/**
 * @brief describe a yolo pre-post process configurations for yolo series
 */
typedef struct {
    float                           prob_thresh;                                                                    /**< probability thresh */
    float                           nms_thresh;                                                                     /**< NMS thresh */
    uint32_t                        max_detection;                                                                  /**< MAX number of detection */
    uint32_t                        max_detection_per_class;                                                        /**< MAX number of detection per class */
    ex_nms_mode_t                   nms_mode;                                                                       /**< NMS mode, please ref ex_nms_mode_t */
    uint16_t                        anchor_layer_num;                                                               /**< number of layer for anchor; 0 < anchor_layer_num <= 5 */
    uint16_t                        anchor_cell_num_per_layer;                                                      /**< number of anchor per gride cell; 0 < anchor_cell_num_per_layer <= 3 */
    uint32_t                        data[EX_MAX_YOLO_ANCHOR_LAYER_NUM][EX_MAX_YOLO_ANCHOR_CELL_NUM_PER_LAYER][2];   /**< anchor array[5][3][2] */
} ex_yolo_post_proc_config_t;

/**
 * @brief describe a object detection result information
 */
struct ex_object_detection_result_s {
    uint32_t class_count;                                                                                           /**< total class count */
    uint32_t box_count;                                                                                             /**< total box count */
    struct ex_bounding_box_s boxes[EX_BOXES_MAX_NUM];                                                               /**< found bounding boxes */
};

int user_post_yolov5_no_sigmoid(int model_id, struct kdp_image_s *image_p);
#endif
