# ******************************************************************************
#  Copyright (c) 2022. Kneron Inc. All rights reserved.                        *
# ******************************************************************************
from typing import List, Union
from utils.ExampleEnum import *

import numpy as np
import re
import os
import sys
import cv2

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '../..'))

import kp

TARGET_FW_VERSION = 'KDP2'


def get_device_usb_speed_by_port_id(usb_port_id: int) -> kp.UsbSpeed:
    device_list = kp.core.scan_devices()

    for device_descriptor in device_list.device_descriptor_list:
        if 0 == usb_port_id:
            return device_descriptor.link_speed
        elif usb_port_id == device_descriptor.usb_port_id:
            return device_descriptor.link_speed

    raise IOError('Specified USB port ID {} not exist.'.format(usb_port_id))


def get_connect_device_descriptor(target_device: str,
                                  scan_index_list: Union[List[int], None],
                                  usb_port_id_list: Union[List[int], None]):
    print('[Check Device]')

    # scan devices
    _device_list = kp.core.scan_devices()

    # check Kneron device exist
    if _device_list.device_descriptor_number == 0:
        print('Error: no Kneron device !')
        exit(0)

    _index_device_descriptor_list = []

    # get device_descriptor of specified scan index
    if scan_index_list is not None:
        for _scan_index in scan_index_list:
            if _device_list.device_descriptor_number > _scan_index >= 0:
                _index_device_descriptor_list.append([_scan_index, _device_list.device_descriptor_list[_scan_index]])
            else:
                print('Error: no matched Kneron device of specified scan index !')
                exit(0)
    # get device_descriptor of specified port ID
    elif usb_port_id_list is not None:
        for _scan_index, __device_descriptor in enumerate(_device_list.device_descriptor_list):
            for _usb_port_id in usb_port_id_list:
                if __device_descriptor.usb_port_id == _usb_port_id:
                    _index_device_descriptor_list.append([_scan_index, __device_descriptor])

        if 0 == len(_index_device_descriptor_list):
            print('Error: no matched Kneron device of specified port ID !')
            exit(0)
    # get device_descriptor of by default
    else:
        _index_device_descriptor_list = [[_scan_index, __device_descriptor] for _scan_index, __device_descriptor in
                                         enumerate(_device_list.device_descriptor_list)]

    # check device_descriptor is specified target device
    if target_device.lower() == 'kl520':
        _target_device_product_id = kp.ProductId.KP_DEVICE_KL520
    elif target_device.lower() == 'kl720':
        _target_device_product_id = kp.ProductId.KP_DEVICE_KL720
    elif target_device.lower() == 'kl630':
        _target_device_product_id = kp.ProductId.KP_DEVICE_KL630
    elif target_device.lower() == 'kl730':
        _target_device_product_id = kp.ProductId.KP_DEVICE_KL730
    elif target_device.lower() == 'kl830':
        _target_device_product_id = kp.ProductId.KP_DEVICE_KL830

    for _scan_index, __device_descriptor in _index_device_descriptor_list:
        if kp.ProductId(__device_descriptor.product_id) != _target_device_product_id:
            print('Error: Not matched Kneron device of specified target device !')
            exit(0)

    for _scan_index, __device_descriptor in _index_device_descriptor_list:
        if TARGET_FW_VERSION not in __device_descriptor.firmware:
            print('Error: device is not running KDP2/KDP2 Loader firmware ...')
            print('please upload firmware first via \'kp.core.load_firmware_from_file()\'')
            exit(0)

    print(' - Success')

    return _index_device_descriptor_list


def read_image(img_path: str, img_type: str, img_format: str):
    print('[Read Image]')
    if img_type == ImageType.GENERAL.value:
        _img = cv2.imread(filename=img_path)

        if len(_img.shape) < 3:
            channel_num = 2
        else:
            channel_num = _img.shape[2]

        if channel_num == 1:
            if img_format == ImageFormat.RGB565.value:
                color_cvt_code = cv2.COLOR_GRAY2BGR565
            elif img_format == ImageFormat.RGBA8888.value:
                color_cvt_code = cv2.COLOR_GRAY2BGRA
            elif img_format == ImageFormat.RAW8.value:
                color_cvt_code = None
            else:
                print('Error: No matched image format !')
                exit(0)
        elif channel_num == 3:
            if img_format == ImageFormat.RGB565.value:
                color_cvt_code = cv2.COLOR_BGR2BGR565
            elif img_format == ImageFormat.RGBA8888.value:
                color_cvt_code = cv2.COLOR_BGR2BGRA
            elif img_format == ImageFormat.RAW8.value:
                color_cvt_code = cv2.COLOR_BGR2GRAY
            else:
                print('Error: No matched image format !')
                exit(0)
        else:
            print('Error: Not support image format !')
            exit(0)

        if color_cvt_code is not None:
            _img = cv2.cvtColor(src=_img, code=color_cvt_code)

    elif img_type == ImageType.BINARY.value:
        with open(file=img_path, mode='rb') as file:
            _img = file.read()
    else:
        print('Error: Not support image type !')
        exit(0)

    print(' - Success')
    return _img


def get_kp_image_format(image_format: str) -> kp.ImageFormat:
    if image_format == ImageFormat.RGB565.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_RGB565
    elif image_format == ImageFormat.RGBA8888.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_RGBA8888
    elif image_format == ImageFormat.YUYV.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_YUYV
    elif image_format == ImageFormat.CRY1CBY0.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_YCBCR422_CRY1CBY0
    elif image_format == ImageFormat.CBY1CRY0.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_YCBCR422_CBY1CRY0
    elif image_format == ImageFormat.Y1CRY0CB.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_YCBCR422_Y1CRY0CB
    elif image_format == ImageFormat.Y1CBY0CR.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_YCBCR422_Y1CBY0CR
    elif image_format == ImageFormat.CRY0CBY1.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_YCBCR422_CRY0CBY1
    elif image_format == ImageFormat.CBY0CRY1.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_YCBCR422_CBY0CRY1
    elif image_format == ImageFormat.Y0CRY1CB.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_YCBCR422_Y0CRY1CB
    elif image_format == ImageFormat.Y0CBY1CR.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_YCBCR422_Y0CBY1CR
    elif image_format == ImageFormat.RAW8.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_RAW8
    elif image_format == ImageFormat.YUV420p.value:
        _kp_image_format = kp.ImageFormat.KP_IMAGE_FORMAT_YUV420
    else:
        print('Error: Not support image format !')
        exit(0)

    return _kp_image_format


def get_kp_normalize_mode(norm_mode: str) -> kp.NormalizeMode:
    if norm_mode == NormalizeMode.NONE.value:
        _kp_norm = kp.NormalizeMode.KP_NORMALIZE_DISABLE
    elif norm_mode == NormalizeMode.KNERON.value:
        _kp_norm = kp.NormalizeMode.KP_NORMALIZE_KNERON
    elif norm_mode == NormalizeMode.YOLO.value:
        _kp_norm = kp.NormalizeMode.KP_NORMALIZE_YOLO
    elif norm_mode == NormalizeMode.TENSORFLOW.value:
        _kp_norm = kp.NormalizeMode.KP_NORMALIZE_TENSOR_FLOW
    elif norm_mode == NormalizeMode.CUSTOMIZED_DEFAULT.value:
        _kp_norm = kp.NormalizeMode.KP_NORMALIZE_CUSTOMIZED_DEFAULT
    elif norm_mode == NormalizeMode.CUSTOMIZED_SUB128.value:
        _kp_norm = kp.NormalizeMode.KP_NORMALIZE_CUSTOMIZED_SUB128
    elif norm_mode == NormalizeMode.CUSTOMIZED_DIV2.value:
        _kp_norm = kp.NormalizeMode.KP_NORMALIZE_CUSTOMIZED_DIV2
    elif norm_mode == NormalizeMode.CUSTOMIZED_SUB128_DIV2.value:
        _kp_norm = kp.NormalizeMode.KP_NORMALIZE_CUSTOMIZED_SUB128_DIV2
    else:
        print('Error: Not support normalize mode !')
        exit(0)

    return _kp_norm


def get_kp_pre_process_resize_mode(resize_mode: str) -> kp.ResizeMode:
    if resize_mode == ResizeMode.NONE.value:
        _kp_resize_mode = kp.ResizeMode.KP_RESIZE_DISABLE
    elif resize_mode == ResizeMode.ENABLE.value:
        _kp_resize_mode = kp.ResizeMode.KP_RESIZE_ENABLE
    else:
        print('Error: Not support pre process resize mode !')
        exit(0)

    return _kp_resize_mode


def get_kp_pre_process_padding_mode(padding_mode: str) -> kp.PaddingMode:
    if padding_mode == PaddingMode.NONE.value:
        _kp_padding_mode = kp.PaddingMode.KP_PADDING_DISABLE
    elif padding_mode == PaddingMode.PADDING_CORNER.value:
        _kp_padding_mode = kp.PaddingMode.KP_PADDING_CORNER
    elif padding_mode == PaddingMode.PADDING_SYMMETRIC.value:
        _kp_padding_mode = kp.PaddingMode.KP_PADDING_SYMMETRIC
    else:
        print('Error: Not support pre process padding mode !')
        exit(0)

    return _kp_padding_mode


def get_ex_post_process_mode(post_proc: str) -> PostprocessMode:
    if post_proc in PostprocessMode._value2member_map_:
        _ex_post_proc = PostprocessMode(post_proc)
    else:
        print('Error: Not support post process mode !')
        exit(0)

    return _ex_post_proc


def parse_crop_box_from_str(crop_box_str: str) -> List[kp.InferenceCropBox]:
    _group_list = re.compile(r'\([\s]*(\d+)[\s]*,[\s]*(\d+)[\s]*,[\s]*(\d+)[\s]*,[\s]*(\d+)[\s]*\)').findall(
        crop_box_str)
    _crop_box_list = []

    for _idx, _crop_box in enumerate(_group_list):
        _crop_box_list.append(
            kp.InferenceCropBox(
                crop_box_index=_idx,
                x=int(_crop_box[0]),
                y=int(_crop_box[1]),
                width=int(_crop_box[2]),
                height=int(_crop_box[3])
            )
        )

    return _crop_box_list


def convert_onnx_data_to_npu_data(tensor_descriptor: kp.TensorDescriptor, onnx_data: np.ndarray) -> bytes:
    def __get_npu_ndarray(tensor_shape_info: kp.TensorShapeInfoV2, npu_ndarray_dtype: np.dtype):
        max_element_num = 0
        dimension_num = len(tensor_shape_info.shape)

        for dimension in range(dimension_num):
            element_num = tensor_shape_info.shape[dimension] * tensor_shape_info.stride_npu[dimension]
            if element_num > max_element_num:
                max_element_num = element_num

        return np.zeros(shape=max_element_num, dtype=npu_ndarray_dtype).flatten()

    quantization_parameters = tensor_descriptor.quantization_parameters
    tensor_shape_info = tensor_descriptor.tensor_shape_info
    npu_data_layout = tensor_descriptor.data_layout

    quantization_max_value = 0
    quantization_min_value = 0
    radix = 0
    scale = 0
    quantization_factor = 0

    onnx_data_shape_index = None
    onnx_data_buf_offset = 0
    npu_data_buf_offset = 0

    npu_data_element_u16b = 0
    npu_data_high_bit_offset = 16

    npu_data_dtype = np.int8

    if tensor_shape_info.version != kp.ModelTensorShapeInformationVersion.KP_MODEL_TENSOR_SHAPE_INFO_VERSION_2:
        raise AttributeError('Unsupport ModelTensorShapeInformationVersion {}'.format(tensor_descriptor.tensor_shape_info.version))

    """
    input data quantization
    """
    if npu_data_layout in [kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B,
                           kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B,
                           kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B,
                           kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_RAW_8B]:
        quantization_max_value = np.iinfo(np.int8).max
        quantization_min_value = np.iinfo(np.int8).min
        npu_data_dtype = np.int8
    elif npu_data_layout in [kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B,
                             kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_RAW_16B,
                             kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8BHL,
                             kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8BHL,
                             kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8BHL]:
        quantization_max_value = np.iinfo(np.int16).max
        quantization_min_value = np.iinfo(np.int16).min
        npu_data_dtype = np.int16
    elif npu_data_layout in [kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_RAW_FLOAT]:
        quantization_max_value = np.finfo(np.float32).max
        quantization_min_value = np.finfo(np.float32).min
        npu_data_dtype = np.float32
    else:
        raise AttributeError('Unsupport ModelTensorDataLayout {}'.format(npu_data_layout))


    shape = np.array(tensor_shape_info.v2.shape, dtype=np.int32)
    dimension_num = len(shape)
    quantized_axis = quantization_parameters.v1.quantized_axis
    radix = np.array([quantized_fixed_point_descriptor.radix for quantized_fixed_point_descriptor in quantization_parameters.v1.quantized_fixed_point_descriptor_list], dtype=np.int32)
    scale = np.array([quantized_fixed_point_descriptor.scale.value for quantized_fixed_point_descriptor in quantization_parameters.v1.quantized_fixed_point_descriptor_list], dtype=np.float64)

    quantization_factor = np.power(2, radix) * scale
    if 1 < len(quantization_parameters.v1.quantized_fixed_point_descriptor_list):
        quantization_factor = np.expand_dims(quantization_factor, axis=tuple([dimension for dimension in range(dimension_num) if dimension is not quantized_axis]))
        quantization_factor = np.broadcast_to(array=quantization_factor, shape=shape)

    onnx_quantized_data = onnx_data * quantization_factor
    onnx_quantized_data = np.round(onnx_quantized_data)
    onnx_quantized_data = np.clip(onnx_quantized_data, quantization_min_value, quantization_max_value).astype(npu_data_dtype)

    """
    flatten onnx/npu data
    """
    onnx_quantized_data_flatten = onnx_quantized_data.flatten()
    npu_data_flatten = __get_npu_ndarray(tensor_shape_info=tensor_shape_info.data, npu_ndarray_dtype=npu_data_dtype)

    '''
    re-arrange data from onnx to npu
    '''
    onnx_data_shape_index = np.zeros(shape=(len(shape)), dtype=int)
    stride_onnx = np.array(tensor_shape_info.v2.stride_onnx, dtype=int)
    stride_npu = np.array(tensor_shape_info.v2.stride_npu, dtype=int)

    if npu_data_layout in [kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B,
                           kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B,
                           kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B,
                           kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_RAW_8B]:
        while True:
            onnx_data_buf_offset = onnx_data_shape_index.dot(stride_onnx)
            npu_data_buf_offset = onnx_data_shape_index.dot(stride_npu)

            npu_data_flatten[npu_data_buf_offset] = onnx_quantized_data_flatten[onnx_data_buf_offset]

            '''
            update onnx_data_shape_index
            '''
            for dimension in range(dimension_num - 1, -1, -1):
                onnx_data_shape_index[dimension] += 1
                if onnx_data_shape_index[dimension] == shape[dimension]:
                    if dimension == 0:
                        break
                    onnx_data_shape_index[dimension] = 0
                    continue
                else:
                    break

            if onnx_data_shape_index[0] == shape[0]:
                break
    elif npu_data_layout in [kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B,
                             kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_RAW_16B]:
        while True:
            onnx_data_buf_offset = onnx_data_shape_index.dot(stride_onnx)
            npu_data_buf_offset = onnx_data_shape_index.dot(stride_npu)

            npu_data_element_u16b = np.frombuffer(buffer=onnx_quantized_data_flatten[onnx_data_buf_offset].tobytes(), dtype=np.uint16)
            npu_data_flatten[npu_data_buf_offset] = np.frombuffer(buffer=(npu_data_element_u16b & 0xfffe).tobytes(), dtype=np.int16)

            '''
            update onnx_data_shape_index
            '''
            for dimension in range(dimension_num - 1, -1, -1):
                onnx_data_shape_index[dimension] += 1
                if onnx_data_shape_index[dimension] == shape[dimension]:
                    if dimension == 0:
                        break
                    onnx_data_shape_index[dimension] = 0
                    continue
                else:
                    break

            if onnx_data_shape_index[0] == shape[0]:
                break
    elif npu_data_layout in [kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8BHL,
                             kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8BHL,
                             kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8BHL]:
        
        npu_data_flatten = np.frombuffer(buffer=npu_data_flatten.tobytes(), dtype=np.uint8).copy()

        while True:
            onnx_data_buf_offset = onnx_data_shape_index.dot(stride_onnx)
            npu_data_buf_offset = onnx_data_shape_index.dot(stride_npu)

            """
            npu_data_buf_offset = (npu_data_buf_offset / 16) * 32 + (npu_data_buf_offset % 16)
            """
            npu_data_buf_offset = ((npu_data_buf_offset >> 4) << 5) + (npu_data_buf_offset & 15)

            npu_data_element_u16b = np.frombuffer(buffer=onnx_quantized_data_flatten[onnx_data_buf_offset].tobytes(), dtype=np.uint16)
            npu_data_element_u16b = (npu_data_element_u16b >> 1)
            npu_data_flatten[npu_data_buf_offset] = (npu_data_element_u16b & 0x007f).astype(dtype=np.uint8)
            npu_data_flatten[npu_data_buf_offset + npu_data_high_bit_offset] = ((npu_data_element_u16b >> 7) & 0x00ff).astype(dtype=np.uint8)

            '''
            update onnx_data_shape_index
            '''
            for dimension in range(dimension_num - 1, -1, -1):
                onnx_data_shape_index[dimension] += 1
                if onnx_data_shape_index[dimension] == shape[dimension]:
                    if dimension == 0:
                        break
                    onnx_data_shape_index[dimension] = 0
                    continue
                else:
                    break

            if onnx_data_shape_index[0] == shape[0]:
                break
    elif npu_data_layout in [kp.ModelTensorDataLayout.KP_MODEL_TENSOR_DATA_LAYOUT_RAW_FLOAT]:
        while True:
            onnx_data_buf_offset = onnx_data_shape_index.dot(stride_onnx)
            npu_data_buf_offset = onnx_data_shape_index.dot(stride_npu)

            npu_data_flatten[npu_data_buf_offset] = onnx_quantized_data_flatten[onnx_data_buf_offset]

            '''
            update onnx_data_shape_index
            '''
            for dimension in range(dimension_num - 1, -1, -1):
                onnx_data_shape_index[dimension] += 1
                if onnx_data_shape_index[dimension] == shape[dimension]:
                    if dimension == 0:
                        break
                    onnx_data_shape_index[dimension] = 0
                    continue
                else:
                    break

            if onnx_data_shape_index[0] == shape[0]:
                break
    else:
        raise AttributeError('Unsupport ModelTensorDataLayout {}'.format(npu_data_layout))
    
    return npu_data_flatten.tobytes()