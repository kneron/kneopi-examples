#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
# SPDX-FileCopyrightText: 2023 Kent Gibson <warthog618@gmail.com>

"""Minimal example of toggling a single line."""

import gpiod
import time

from gpiod.line import Direction, Value, Drive

LED_GPIO_CHIP ="/dev/gpiochip2"
LED_GPIO_PIN = 18

#KNEO PI on board LED
#LED_G: GPIO_2_IO_D[18]
#LED_B: GPIO_2_IO_D[19]

def toggle_value(value):
    if value == Value.INACTIVE:
        return Value.ACTIVE
    return Value.INACTIVE


def toggle_line_value(chip_path, line_offset):
    value_str = {Value.ACTIVE: "Active", Value.INACTIVE: "Inactive"}
    value = Value.ACTIVE

    with gpiod.request_lines(
        chip_path,
        consumer="toggle-line-value",
        config={
            line_offset: gpiod.LineSettings(
                direction=Direction.OUTPUT
            )
        },
    ) as request:
        try:
            while True:
                print("gpio {}={}".format(line_offset, value_str[value]))
                time.sleep(1)
                value = toggle_value(value)
                request.set_value(line_offset, value)
        except KeyboardInterrupt:
            print ("\nEnd of program")


if __name__ == "__main__":
    try:
        toggle_line_value(LED_GPIO_CHIP, LED_GPIO_PIN)
    except OSError as ex:
        print(ex, "\nCustomise the example configuration to suit your situation")