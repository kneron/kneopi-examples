# ******************************************************************************
#  Copyright (c) 2021-2022. Kneron Inc. All rights reserved.                   *
# ******************************************************************************

import os
import sys
import argparse

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

from utils.ExampleHelper import get_device_usb_speed_by_port_id, convert_onnx_data_to_npu_data
from utils.ExamplePostProcess import post_process_yolo_v5
import kp
import cv2
import numpy as np
import math

SCPU_FW_PATH = os.path.join(PWD, 'res/firmware/KL730/kp_firmware.tar')
MODEL_FILE_PATH = os.path.join(PWD, 'res/models/KL730/YoloV5s_640_640_3/models_730.nef')
IMAGE_FILE_PATH = os.path.join(PWD, 'res/images/people_talk_in_street_640x640.bmp')
LOOP_TIME = 100

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KL730 Demo Generic Data Inference Example.')
    parser.add_argument('-p',
                        '--port_id',
                        help='Using specified port ID for connecting device (Default: port ID of first scanned Kneron '
                             'device)',
                        default=0,
                        type=int)
    args = parser.parse_args()

    usb_port_id = args.port_id

    """
    check device USB speed (Recommend run KL730 at super/high speed)
    """
    try:
        usb_speed = get_device_usb_speed_by_port_id(usb_port_id=usb_port_id)
        if kp.UsbSpeed.KP_USB_SPEED_SUPER != usb_speed and \
           kp.UsbSpeed.KP_USB_SPEED_HIGH != usb_speed:
            print('\033[91m' + '[Warning] Device is not run at super/high speed.' + '\033[0m')
    except Exception as exception:
        print('Error: check device USB speed fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                             str(exception)))
        exit(0)

    """
    connect the device
    """
    try:
        print('[Connect Device]')
        device_group = kp.core.connect_devices(usb_port_ids=[usb_port_id])
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: connect device fail, port ID = \'{}\', error msg: [{}]'.format(usb_port_id,
                                                                                     str(exception)))
        exit(0)

    """
    upload firmware to device
    """
    try:
        print('[Upload Firmware]')
        kp.core.load_firmware_from_file(device_group=device_group,
                                        scpu_fw_path=SCPU_FW_PATH,
                                        ncpu_fw_path="")
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: upload firmware failed, error = \'{}\''.format(str(exception)))
        exit(0)

    """
    setting timeout of the usb communication with the device
    """
    print('[Set Device Timeout]')
    kp.core.set_timeout(device_group=device_group, milliseconds=5000)
    print(' - Success')

    """
    upload model to device
    """
    try:
        print('[Upload Model]')
        model_nef_descriptor = kp.core.load_model_from_file(device_group=device_group,
                                                            file_path=MODEL_FILE_PATH)
        print(' - Success')
    except kp.ApiKPException as exception:
        print('Error: upload model failed, error = \'{}\''.format(str(exception)))
        exit(0)

    """
    prepare the pre-processed data for NPU inference
    (for more information, please refer: https://doc.kneron.com/docs/#plus_python/tutorial/chapter/model_inference_with_data_inference/)
    """
    print('[Prepare NPU Inference Data]')
    # read data file
    img = cv2.imread(filename=IMAGE_FILE_PATH)
    img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    data = img.astype(np.int32)

    # prepare origin model input data (relayout 640x640x3 rgba8888 image to 1x3x640x640 model input data)
    onnx_data_shape = model_nef_descriptor.models[0].input_nodes[0].tensor_shape_info.v2.shape
    onnx_data = data.transpose((2, 0, 1)).reshape(onnx_data_shape)

    # do normalization - this model is trained with normalize method: (data - 128) / 256
    onnx_data = (onnx_data - 128) / 256.0

    # convert the onnx data to npu data
    npu_input_buffer = convert_onnx_data_to_npu_data(tensor_descriptor=model_nef_descriptor.models[0].input_nodes[0],
                                                     onnx_data=onnx_data)
    print(' - Success')

    """
    prepare generic data inference input descriptor
    """
    generic_inference_input_descriptor = kp.GenericDataInferenceDescriptor(
        model_id=model_nef_descriptor.models[0].id,
        inference_number=0,
        input_node_data_list=[kp.GenericInputNodeData(buffer=npu_input_buffer)]
    )

    """
    starting inference work
    """
    print('[Starting Inference Work]')
    print(' - Starting inference loop {} times'.format(LOOP_TIME))
    print(' - ', end='')
    for i in range(LOOP_TIME):
        try:
            kp.inference.generic_data_inference_send(device_group=device_group,
                                                     generic_inference_input_descriptor=generic_inference_input_descriptor)

            generic_raw_result = kp.inference.generic_data_inference_receive(device_group=device_group)
        except kp.ApiKPException as exception:
            print(' - Error: inference failed, error = {}'.format(exception))
            exit(0)

        print('.', end='', flush=True)
    print()

    """
    retrieve inference node output 
    """
    print('[Retrieve Inference Node Output ]')
    inf_node_output_list = []
    for node_idx in range(generic_raw_result.header.num_output_node):
        inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(
            node_idx=node_idx,
            generic_raw_result=generic_raw_result,
            channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_DEFAULT)
        inf_node_output_list.append(inference_float_node_output)

    print(' - Success')

    """
    post-process the last raw output
    """
    print('[Yolo V5 Post-Processing]')
    hardware_preproc_info = kp.HwPreProcInfo(img_width=img_width,
                                             img_height=img_height,
                                             resized_img_width=img_width,
                                             resized_img_height=img_height,
                                             model_input_width=onnx_data_shape[3],
                                             model_input_height=onnx_data_shape[2])
    yolo_result = post_process_yolo_v5(inference_float_node_output_list=inf_node_output_list,
                                       hardware_preproc_info=hardware_preproc_info,
                                       thresh_value=0.5,
                                       with_sigmoid=False)

    print(' - Success')

    print('[Result]')
    print(yolo_result)
