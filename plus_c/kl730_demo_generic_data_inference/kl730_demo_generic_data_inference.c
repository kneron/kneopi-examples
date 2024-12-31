/**
 * @file        kl720_demo_generic_image_inference.c
 * @brief       main code of generic image inference (output raw data) for multiple input
 * @version     0.1
 * @date        2022-12-14
 *
 * @copyright   Copyright (c) 2022 Kneron Inc. All rights reserved.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>

#include "kp_core.h"
#include "kp_inference.h"
#include "helper_functions.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static char _scpu_fw_path[128] = "../res/firmware/KL730/kp_firmware.tar";
static char _model_file_path[128] = "../res/models/KL730/YoloV5s_640_640_3/models_730.nef";
static char _image_file_path[128] = "../res/images/people_talk_in_street_640x640.bmp";
static int _loop = 10;

static kp_devices_list_t *_device_list;
static kp_device_group_t _device;
static kp_model_nef_descriptor_t _model_desc;
static kp_generic_data_inference_desc_t _input_data;
static kp_generic_data_inference_result_header_t _output_desc;

int main(int argc, char *argv[])
{
    // each device has a unique port ID, 0 for auto-search
    int port_id = (argc > 1) ? atoi(argv[1]) : 0;
    int ret;

    /******* check the device USB speed *******/
    int link_speed;
    _device_list = kp_scan_devices();

    helper_get_device_usb_speed_by_port_id(_device_list, port_id, &link_speed);
    if ((KP_USB_SPEED_SUPER != link_speed) && (KP_USB_SPEED_HIGH != link_speed))
        printf("[warning] device is not run at super/high speed.\n");

    /******* connect the device *******/
    _device = kp_connect_devices(1, &port_id, NULL);
    printf("connect device ... %s\n", (_device) ? "OK" : "failed");

    if (NULL == _device) {
        return -1;
    }

    kp_set_timeout(_device, 5000); // 5 secs timeout

    /******* upload firmware to device *******/
    ret = kp_load_firmware_from_file(_device, _scpu_fw_path, NULL);
    printf("upload firmware ... %s\n", (ret == KP_SUCCESS) ? "OK" : "failed");

    /******* upload model to device *******/
    ret = kp_load_model_from_file(_device, _model_file_path, &_model_desc);
    printf("upload model ... %s\n", (ret == KP_SUCCESS) ? "OK" : "failed");

    /******* allocate memory for raw output *******/
    // by default here select first model
    uint32_t raw_buf_size = _model_desc.models[0].max_raw_out_size;
    uint8_t *raw_output_buf = (uint8_t *)malloc(raw_buf_size);

    _input_data.model_id = _model_desc.models[0].id;    // first model ID
    _input_data.inference_number = 0;                   // inference number, used to verify with output result
    _input_data.num_input_node_data = 1;                // number of data

    /******* prepare the pre-processed data for NPU inference *******/
    uint8_t *img_buf_rgba8888;
    int32_t img_buf_rgba8888_offset;
    int32_t img_width;
    int32_t img_height;
    int32_t img_channel             = 4;

    float *onnx_data_buf            = NULL;
    int32_t onnx_data_buf_size      = 0;
    int32_t onnx_data_buf_offset    = 0;

    int8_t *npu_data_buf            = NULL;
    int32_t npu_data_buf_size       = 0;

    // read data file
    img_buf_rgba8888 = (uint8_t *)helper_bmp_file_to_raw_buffer(_image_file_path, &img_width, &img_height, KP_IMAGE_FORMAT_RGBA8888);
    printf("read image ... %s\n", (img_buf_rgba8888) ? "OK" : "failed");

    if (NULL == img_buf_rgba8888) {
        kp_release_model_nef_descriptor(&_model_desc);
        kp_disconnect_devices(_device);
        free(raw_output_buf);
        return -1;
    }

    // get model input size (shape order: BxCxHxW) (KL730 only support kp_tensor_shape_info_v2_t)
    kp_tensor_descriptor_t *input_node_0            = &(_model_desc.models[0].input_nodes[0]);
    kp_tensor_shape_info_v2_t* tensor_shape_info    = &(input_node_0->tensor_shape_info.tensor_shape_info_data.v2);
    uint32_t onnx_data_channel                      = tensor_shape_info->shape[1];
    uint32_t onnx_data_height                       = tensor_shape_info->shape[2];
    uint32_t onnx_data_width                        = tensor_shape_info->shape[3];

    // prepare origin model input data (relayout 640x640x3 rgba8888 image to 1x3x640x640 model input data)
    onnx_data_buf_size    = onnx_data_width * onnx_data_height * onnx_data_channel;
    onnx_data_buf         = calloc(onnx_data_buf_size, sizeof(float));

    printf("prepare model input data ... %s\n", (onnx_data_buf) ? "OK" : "failed");

    if (NULL == onnx_data_buf) {
        kp_release_model_nef_descriptor(&_model_desc);
        kp_disconnect_devices(_device);
        free(raw_output_buf);
        free(img_buf_rgba8888);
        return -1;
    }

    for (int channel = 0; channel < onnx_data_channel; channel++) {
        img_buf_rgba8888_offset = channel;

        for (int pixel = 0; pixel < img_height * img_width; pixel++) {
            onnx_data_buf[onnx_data_buf_offset + pixel] = (float)img_buf_rgba8888[img_buf_rgba8888_offset];
            img_buf_rgba8888_offset += img_channel;
        }

        onnx_data_buf_offset += (onnx_data_height * onnx_data_width);
    }

    free(img_buf_rgba8888);

    // do normalization - this model is trained with normalize method: (data - 128) / 256
    for (int i = 0; i < onnx_data_buf_size; i++) {
        onnx_data_buf[i] = ((float)onnx_data_buf[i] - 128.0f) / 256.0f;
    }

    // convert the onnx data to npu data
    ret = helper_convert_onnx_data_to_npu_data(input_node_0, onnx_data_buf, onnx_data_buf_size, &npu_data_buf, &npu_data_buf_size);
    if (KP_SUCCESS != ret) {
        kp_release_model_nef_descriptor(&_model_desc);
        kp_disconnect_devices(_device);
        free(raw_output_buf);
        free(onnx_data_buf);
        return -1;
    }

    free(onnx_data_buf);

    /******* set up the input descriptor *******/
    _input_data.input_node_data_list[0].buffer_size = npu_data_buf_size;
    _input_data.input_node_data_list[0].buffer      = (uint8_t *)npu_data_buf;

    printf("\nstarting inference loop %d times:\n", _loop);

    /******* starting inference work *******/
    for (int i = 0; i < _loop; i++)
    {
        ret = kp_generic_data_inference_send(_device, &_input_data);
        if (ret != KP_SUCCESS)
            break;

        ret = kp_generic_data_inference_receive(_device, &_output_desc, raw_output_buf, raw_buf_size);
        if (ret != KP_SUCCESS)
            break;

        printf(".");
        fflush(stdout);
    }

    printf("\n");

    free(npu_data_buf);
    kp_release_model_nef_descriptor(&_model_desc);
    kp_disconnect_devices(_device);

    if (ret != KP_SUCCESS)
    {
        printf("\ninference failed, error = %d (%s)\n", ret, kp_error_string(ret));
        return -1;
    }

    printf("\ninference loop is done.\n");

    kp_inf_float_node_output_t *output_nodes[3] = {NULL}; // yolo v5s outputs only three nodes, described by _output_desc.num_output_node

    // retrieve output nodes in floating point format
    output_nodes[0] = kp_generic_inference_retrieve_float_node(0, raw_output_buf, KP_CHANNEL_ORDERING_DEFAULT);
    output_nodes[1] = kp_generic_inference_retrieve_float_node(1, raw_output_buf, KP_CHANNEL_ORDERING_DEFAULT);
    output_nodes[2] = kp_generic_inference_retrieve_float_node(2, raw_output_buf, KP_CHANNEL_ORDERING_DEFAULT);

    // print and save the content of node data to files
    helper_dump_floating_node_data_to_files(output_nodes, _output_desc.num_output_node, _image_file_path);

    kp_release_float_node_output(output_nodes[0]);
    kp_release_float_node_output(output_nodes[1]);
    kp_release_float_node_output(output_nodes[2]);

    free(raw_output_buf);

    return 0;
}
