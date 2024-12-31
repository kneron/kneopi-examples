/**
 * @file        helper_functions.h
 * @brief       header file of helper_functions 
 * @version     0.1
 * @date        2021-03-22
 *
 * @copyright   Copyright (c) 2021 Kneron Inc. All rights reserved.
 */

#ifndef __HELPER_FUNCTIONS_H__
#define __HELPER_FUNCTIONS_H__

#include "kp_struct.h"

void helper_measure_time_begin();
void helper_measure_time_end(double *measued_time);
char *helper_bmp_file_to_raw_buffer(const char *file_path, int *width, int *height, kp_image_format_t format);
char *helper_bin_file_to_raw_buffer(const char *file_path, int width, int height, kp_image_format_t format);
void helper_draw_box_on_bmp(const char *in_bmp_path, const char *out_bmp_path, kp_bounding_box_t boxes[], int box_count);
void helper_draw_box_on_bmp_from_bin(const char *in_bin_path, int in_bin_width, int in_bin_height, kp_image_format_t in_bin_format,
                                     const char *out_bmp_path, kp_bounding_box_t boxes[], int box_count);
void helper_draw_box_of_crop_area_on_bmp(const char *in_bmp_path, const char *out_bmp_path, kp_bounding_box_t boxes[], int box_count,
                                         kp_inf_crop_box_t crop_box);
void helper_draw_box_of_crop_area_on_bmp_from_bin(const char *in_bin_path, int in_bin_width, int in_bin_height, kp_image_format_t in_bin_format,
                                                  const char *out_bmp_path, kp_bounding_box_t boxes[], int box_count, kp_inf_crop_box_t crop_box);
void helper_print_yolo_box(kp_yolo_result_t *yolo_result);
void helper_print_yolo_box_on_bmp(kp_yolo_result_t *yolo_result, char *in_bmp_path);
void helper_print_yolo_box_on_bmp_from_bin(kp_yolo_result_t *yolo_result, char *in_bin_path, int in_bin_width, int in_bin_height,
                                           kp_image_format_t in_bin_format);
void helper_print_yolo_box_of_crop_area(kp_yolo_result_t *yolo_result, kp_inf_crop_box_t crop_box);
void helper_print_yolo_box_of_crop_area_on_bmp(kp_yolo_result_t *yolo_result, char *in_bmp_path, kp_inf_crop_box_t crop_box);
void helper_print_yolo_box_of_crop_area_on_bmp_from_bin(kp_yolo_result_t *yolo_result, char *in_bin_path, int in_bin_width, int in_bin_height,
                                                        kp_image_format_t in_bin_format, kp_inf_crop_box_t crop_box);

void helper_string_to_number_array(char *number_string, int *number_group, int *num_devs);
char *helper_file_name_from_path(char *path);
int helper_string_to_crop_box_array(char *crop_box_array, uint32_t *crop_count, kp_inf_crop_box_t *crop_boxes); // return 0 on success, -1 on fail
char *helper_kp_fixed_point_dtype_to_string(uint32_t fixed_point_dtype);
kp_inf_float_node_output_t* helper_fixed_to_floating_node_data(kp_inf_fixed_node_output_t *fixed_node_output);
void helper_dump_floating_node_data_to_files(kp_inf_float_node_output_t *node_output[], int num_nodes, char *in_bmp_path);
void helper_dump_floating_node_data_of_crop_area_to_files(kp_inf_float_node_output_t *output_nodes[], int num_nodes, int crop_num, char *in_bmp_path);
void helper_dump_fixed_node_data_to_files(kp_inf_fixed_node_output_t *node_output[], int num_nodes, char *in_bmp_path);
void helper_dump_fixed_node_data_of_crop_area_to_files(kp_inf_fixed_node_output_t *output_nodes[], int num_nodes, int crop_num, char *in_bmp_path);
char *helper_kp_model_tensor_data_layout_to_string(uint32_t tensor_data_layout);
char *helper_kp_model_target_chip_to_string(uint32_t target_chip);
void helper_print_kp_quantized_fixed_point_descriptor(kp_quantized_fixed_point_descriptor_t *quantized_fixed_point_descriptor);
void helper_print_model_info(kp_model_nef_descriptor_t *pModel_desc);

void helper_get_device_usb_speed_by_port_id(kp_devices_list_t *devices_list, int port_id, int *link_speed);

// box_count_last and boxes_last: box count and all boxes info in the last image frame
// box_count_lastest and boxes_lastest: box count and all boxes info in the current image frame
// box_count_stabilized and boxes_stabilized: stabilized boxes result and the total count of them
void helper_bounding_box_stabilization(uint32_t box_count_last, kp_bounding_box_t *boxes_last,
                                       uint32_t box_count_latest, kp_bounding_box_t *boxes_latest,
                                       uint32_t *box_count_stabilized, kp_bounding_box_t *boxes_stabilized,
                                       int pixel_offset, float box_score_threshold);

void helper_convert_rgb888_to_bmp(const char *out_bmp_path, uint32_t width, uint32_t height, uint8_t *rgb888_buf);

int helper_get_quantization_parameters_v1_information(kp_quantization_parameters_v1_t* quantization_parameters_v1, int quantized_fixed_point_descriptor_idx, int *radix, float *scale);
int helper_convert_onnx_data_to_npu_data(kp_tensor_descriptor_t *tensor_descriptor, float *onnx_data_buf, int32_t onnx_data_buf_size, int8_t **npu_data_buf, int32_t *npu_data_buf_size);
#endif
