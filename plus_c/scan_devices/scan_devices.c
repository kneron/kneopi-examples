/**
 * @file        scan_devices.c
 * @brief       example of scanning Kneron KL series device
 * @version     0.1
 * @date        2021-03-22
 *
 * @copyright   Copyright (c) 2021 Kneron Inc. All rights reserved.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kp_core.h"

int main(int argc, char *argv[])
{
    printf("\n");
    printf("scanning kneron devices ...\n");

    kp_devices_list_t *list;
    list = kp_scan_devices();

    printf("number of Kneron devices found: %d\n", list->num_dev);

    if (list->num_dev == 0)
        return 0;

    printf("\n");
    printf("listing devices infomation as follows:\n");

    for (int i = 0; i < list->num_dev; i++)
    {
        kp_device_descriptor_t *dev_descp = &(list->device[i]);

        printf("\n");
        printf("[%d] scan_index: '%d'\n", i, i);
        printf("[%d] port ID: '%d'\n", i, dev_descp->port_id);
        printf("[%d] product_id: '0x%x'", i, dev_descp->product_id);

        if (dev_descp->product_id == KP_DEVICE_KL520)
            printf(" (KL520)");
        else if (dev_descp->product_id == KP_DEVICE_KL720)
            printf(" (KL720)");
        else if (dev_descp->product_id == KP_DEVICE_KL630)
            printf(" (KL630)");
        else if (dev_descp->product_id == KP_DEVICE_KL730)
            printf(" (KL730)");
        else if (dev_descp->product_id == KP_DEVICE_KL830)
            printf(" (KL830)");
        else
            printf(" (Unknown)");
        printf("\n");

        printf("[%d] USB link speed: ", i);
        switch (dev_descp->link_speed)
        {
        case KP_USB_SPEED_LOW:
            printf("'Low-Speed'\n");
            break;
        case KP_USB_SPEED_FULL:
            printf("'Full-Speed'\n");
            break;
        case KP_USB_SPEED_HIGH:
            printf("'High-Speed'\n");
            break;
        case KP_USB_SPEED_SUPER:
            printf("'Super-Speed'\n");
            break;
        default:
            printf("'Unknown'\n");
            break;
        }
        printf("[%d] USB port path: '%s'\n", i, dev_descp->port_path);
        printf("[%d] kn_number: '0x%8X' %s\n", i, dev_descp->kn_number, dev_descp->kn_number == 0x0 ? "(invalid)" : "");
        printf("[%d] Connectable: '%s'\n", i, dev_descp->isConnectable == true ? "True" : "False");
        printf("[%d] Firmware: '%s'\n", i, dev_descp->firmware);
    }

    return 0;
}
