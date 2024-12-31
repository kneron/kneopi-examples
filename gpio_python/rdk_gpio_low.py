import os
import time

# Pin Header corresponding to GPIO
pin_to_gpio = {
    3: "70", 5: "69", 7: "72", 8: "23", 10: "22", 11: "53", 12: "74",
    13: "52", 15: "56", 16: "55", 18: "54", 19: "66", 21: "68", 22: "19",
    23: "67", 24: "65", 26: "32", 27: "51", 28: "41", 29: "57", 31: "75",
    32: "20", 33: "21", 35: "73", 36: "60", 37: "8", 38: "76", 40: "71"
}

def export_gpio(pin):
    
    try:
        with open(f"/sys/class/gpio/export", "w") as f:
            f.write(pin)
    except IOError:
        # GPIO may have already been exported, ignore the error
        pass

def set_gpio_direction(pin, direction="out"):

    with open(f"/sys/class/gpio/gpio{pin}/direction", "w") as f:
        f.write(direction)

def write_gpio_value(pin, value):
    
    with open(f"/sys/class/gpio/gpio{pin}/value", "w") as f:
        f.write(str(value))

def gpio_exists(pin):
    
    return os.path.isdir(f"/sys/class/gpio/gpio{pin}")

def control_gpio(gpio_pins):
    
    for pin in gpio_pins:
        if not gpio_exists(pin):
            export_gpio(pin)
            set_gpio_direction(pin)
        else:
            set_gpio_direction(pin)

    while True:
        for pin in gpio_pins:
            # GPIO LOW (LED ON)
            write_gpio_value(pin, 0)
        time.sleep(1)  # Wait for 1 second

        for pin in gpio_pins:
            # GPIO HIGH (LED OFF)
            write_gpio_value(pin, 1)
        time.sleep(1)  # Wait for 1 second

def main():
    # Ask the user to input the Pin Header to select the GPIO
    user_input = input("Please enter the Pin Header numbers to control (e.g., 27 33 35): ")
    # Split the input Pin Header into a list
    selected_pins = [int(pin) for pin in user_input.split()]

    # Corresponding GPIOs
    gpio_pins = [pin_to_gpio[pin] for pin in selected_pins if pin in pin_to_gpio]

    if not gpio_pins:
        print("Invalid Pin Header input. Please ensure the numbers are correct.")
    else:
        print(f"The following GPIOs will be controlled: {', '.join(gpio_pins)}")
        control_gpio(gpio_pins)

if __name__ == "__main__":
    main()
