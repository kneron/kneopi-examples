# ******************************************************************************
#  Copyright (c) 2021-2022. Kneron Inc. All rights reserved.                   *
# ******************************************************************************

import os
import sys

PWD = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(PWD, '..'))

import kp


def get_product_name(product_id: int) -> str:
    if product_id == kp.ProductId.KP_DEVICE_KL520.value:
        return '(KL520)'
    elif product_id == kp.ProductId.KP_DEVICE_KL720.value:
        return '(KL720)'
    elif product_id == 0x200:
        return '(KL720) (Please update firmware by Kneron DFUT)'
    elif product_id == kp.ProductId.KP_DEVICE_KL630.value:
        return '(KL630)'
    elif product_id == kp.ProductId.KP_DEVICE_KL730.value:
        return '(KL730)'
    elif product_id == kp.ProductId.KP_DEVICE_KL830.value:
        return '(KL830)'
    else:
        return '(Unknown Device)'


if __name__ == "__main__":
    print('scanning kneron devices ...')

    device_list = kp.core.scan_devices()

    print('number of Kneron devices found: {}'.format(device_list.device_descriptor_number))

    if device_list.device_descriptor_number == 0:
        exit(0)

    print('listing devices infomation as follows:')

    for idx, device_descriptor in enumerate(device_list.device_descriptor_list):
        print()
        print('[{}] USB scan index: \'{}\''.format(idx, idx))
        print('[{}] USB port ID: \'{}\''.format(idx, device_descriptor.usb_port_id))
        print('[{}] Product ID: \'0x{:X} {}\''.format(idx,
                                                      device_descriptor.product_id,
                                                      get_product_name(product_id=device_descriptor.product_id)))

        print('[{}] USB link speed: \'{}\''.format(idx, device_descriptor.link_speed))
        print('[{}] USB port path: \'{}\''.format(idx, device_descriptor.usb_port_path))
        print('[{}] KN number: \'0x{:X}\''.format(idx, device_descriptor.kn_number))
        print('[{}] Connectable: \'{}\''.format(idx, device_descriptor.is_connectable))
        print('[{}] Firmware: \'{}\''.format(idx, device_descriptor.firmware))
