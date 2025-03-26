import kp
import cv2
import os
import numpy as np
import glob
import torch
import torch.nn.functional as F
from utils.ExampleHelper import get_device_usb_speed_by_port_id, convert_onnx_data_to_npu_data

import sys
PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))
import argparse

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [(128, 64,128),
             (244, 35,232),
             ( 70, 70, 70),
             (102,102,156),
             (190,153,153),
             (153,153,153),
             (250,170, 30),
             (220,220,  0),
             (107,142, 35),
             (152,251,152),
             ( 70,130,180),
             (220, 20, 60),
             (255,  0,  0),
             (  0,  0,142),
             (  0,  0, 70),
             (  0, 60,100),
             (  0, 80,100),
             (  0,  0,230),
             (119, 11, 32)]

def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def preprocess(img, witdth, heigh):

    img = cv2.resize(img,(witdth,heigh))
    sv_img = np.zeros_like(img).astype(np.uint8)
    img = input_transform(img)
    np_data = img.transpose((2, 0, 1)).copy()
    np_data = np.expand_dims(np_data, axis=[0]).astype(np.float32)

    return np_data

def postprocess(inf_results, ori_image_shape):

    pred = torch.from_numpy(inf_results)
    raw_shape_info = ori_image_shape
    pred = F.interpolate(pred, size=(raw_shape_info[0],raw_shape_info[1]), 
                            mode='bilinear', align_corners=True)
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    return pred

IMAGE_FILE_PATH = None
MODEL_FILE_PATH = None
SCPU_FW_PATH = None

usb_port_id = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='KL730 Demo Generic Image Inference Example.')
    parser.add_argument('-p',
                        '--port_id',
                        help='Using specified port ID for connecting device (Default: port ID of first scanned Kneron '
                             'device)',
                        default=0,
                        type=int)
    parser.add_argument('-m',
                        '--model',
                        help='pidnet nef model path',
                        default=os.path.join(PWD, 'res/models/KL730/Pidnet_640_480/models_730.nef'),
                        type=str)
    parser.add_argument('-img',
                        '--image_path',
                        help='test image path',
                        default=os.path.join(PWD, 'res/images/frankfurt_000000_003025_leftImg8bit.png'),
                        type=str)
    parser.add_argument('-fw',
                        '--firmware_path',
                        help='firmware path',
                        default=os.path.join(PWD, 'res/firmware/KL730/kp_firmware.tar'),
                        type=str)
    args = parser.parse_args()

    usb_port_id = args.port_id
    IMAGE_FILE_PATH = args.image_path 
    MODEL_FILE_PATH = args.model
    SCPU_FW_PATH = args.firmware_path
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


    input_image = cv2.imread(IMAGE_FILE_PATH, cv2.IMREAD_COLOR)

    model_input_w, model_input_h = 640, 480
    in_data = preprocess(input_image, model_input_w, model_input_h)

    # convert the onnx data to npu data
    npu_input_buffer = convert_onnx_data_to_npu_data(tensor_descriptor=model_nef_descriptor.models[0].input_nodes[0],
                                                        onnx_data=in_data)

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

    try:
        kp.inference.generic_data_inference_send(device_group=device_group,
                                                    generic_inference_input_descriptor=generic_inference_input_descriptor)

        generic_raw_result = kp.inference.generic_data_inference_receive(device_group=device_group)
    except kp.ApiKPException as exception:
        print(' - Error: inference failed, error = {}'.format(exception))
        exit(0)


    print(' - Success')

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

    # onnx output data processing
    pred = postprocess(inf_node_output_list[0].ndarray, input_image.shape)

    # visualize segmentation result to img
    sv_img = np.zeros_like(input_image).astype(np.uint8)
    for i, color in enumerate(color_map):
        for j in range(3):
            sv_img[:,:,j][pred==i] = color_map[i][j]

    save_path = os.path.abspath("./kneopi_on_board_inf.png")
    cv2.imwrite(save_path, sv_img)
    print("save result to " + save_path)

