# ******************************************************************************
#  Copyright (c) 2022. Kneron Inc. All rights reserved.                        *
# ******************************************************************************
from typing import List

import os
import sys

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '../..'))

from kp.KPBaseClass.ValueBase import ValueRepresentBase


class ExampleBoundingBox(ValueRepresentBase):
    """
    Example Bounding box descriptor.

    Attributes
    ----------
    x1 : int, default=0
        X coordinate of bounding box top-left corner.
    y1 : int, default=0
        Y coordinate of bounding box top-left corner.
    x2 : int, default=0
        X coordinate of bounding box bottom-right corner.
    y2 : int, default=0
        Y coordinate of bounding box bottom-right corner.
    score : float, default=0
        Probability score.
    class_num : int, default=0
        Class # (of many) with highest probability.
    """

    def __init__(self,
                 x1: int = 0,
                 y1: int = 0,
                 x2: int = 0,
                 y2: int = 0,
                 score: float = 0,
                 class_num: int = 0):
        """
        Example Bounding box descriptor.

        Parameters
        ----------
        x1 : int, default=0
            X coordinate of bounding box top-left corner.
        y1 : int, default=0
            Y coordinate of bounding box top-left corner.
        x2 : int, default=0
            X coordinate of bounding box bottom-right corner.
        y2 : int, default=0
            Y coordinate of bounding box bottom-right corner.
        score : float, default=0
            Probability score.
        class_num : int, default=0
            Class # (of many) with highest probability.
        """

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.score = score
        self.class_num = class_num

    def get_member_variable_dict(self) -> dict:
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'score': self.score,
            'class_num': self.class_num
        }


class ExampleYoloResult(ValueRepresentBase):
    """
    Example YOLO output result descriptor.

    Attributes
    ----------
    class_count : int, default=0
        Total detectable class count.
    box_count : int, default=0
        Total bounding box number.
    box_list : List[ExampleBoundingBox], default=[]
        bounding boxes.
    """

    def __init__(self,
                 class_count: int = 0,
                 box_count: int = 0,
                 box_list: List[ExampleBoundingBox] = []):
        """
        Example YOLO output result descriptor.

        Parameters
        ----------
        class_count : int, default=0
            Total detectable class count.
        box_count : int, default=0
            Total bounding box number.
        box_list : List[ExampleBoundingBox], default=[]
            bounding boxes.
        """
        self.class_count = class_count
        self.box_count = box_count
        self.box_list = box_list

    def _cast_element_buffer(self) -> None:
        pass

    def get_member_variable_dict(self) -> dict:
        member_variable_dict = {
            'class_count': self.class_count,
            'box_count': self.box_count,
            'box_list': {}
        }

        for idx, box_element in enumerate(self.box_list):
            member_variable_dict['box_list'][idx] = box_element.get_member_variable_dict()

        return member_variable_dict
