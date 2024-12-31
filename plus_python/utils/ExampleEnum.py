# ******************************************************************************
#  Copyright (c) 2021-2022. Kneron Inc. All rights reserved.                   *
# ******************************************************************************

from enum import Enum


class ImageType(Enum):
    GENERAL = 'general'
    BINARY = 'binary'


class ImageFormat(Enum):
    RGB565 = 'RGB565'
    RGBA8888 = 'RGBA8888'
    YUYV = 'YUYV'
    CRY1CBY0 = 'CrY1CbY0'
    CBY1CRY0 = 'CbY1CrY0'
    Y1CRY0CB = 'Y1CrY0Cb'
    Y1CBY0CR = 'Y1CbY0Cr'
    CRY0CBY1 = 'CrY0CbY1'
    CBY0CRY1 = 'CbY0CrY1'
    Y0CRY1CB = 'Y0CrY1Cb'
    Y0CBY1CR = 'Y0CbY1Cr'
    RAW8 = 'RAW8'
    YUV420p = 'YUV420p'


class ResizeMode(Enum):
    NONE = 'none'
    ENABLE = 'auto'


class PaddingMode(Enum):
    NONE = 'none'
    PADDING_CORNER = 'corner'
    PADDING_SYMMETRIC = 'symmetric'


class PostprocessMode(Enum):
    NONE = 'none'
    YOLO_V3 = 'yolo_v3'
    YOLO_V5 = 'yolo_v5'


class NormalizeMode(Enum):
    NONE = 'none'
    KNERON = 'kneron'
    TENSORFLOW = 'tensorflow'
    YOLO = 'yolo'
    CUSTOMIZED_DEFAULT = 'customized_default'
    CUSTOMIZED_SUB128 = 'customized_sub128'
    CUSTOMIZED_DIV2 = 'customized_div2'
    CUSTOMIZED_SUB128_DIV2 = 'customized_sub128_div2'


class InferenceRetrieveNodeMode(Enum):
    FIXED = 'fixed'
    FLOAT = 'float'
