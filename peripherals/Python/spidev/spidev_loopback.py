#spidev loopback test:

#KNEO PI Pin Header (SPI bus 1 device 0. /dev/spidev1.0)
#PIN19 SPI_1_O_TXD/GPIO_2_IO_D[2] <-> PIN21 SPI_1_I_RXD/GPIO_2_IO_D[4]

import spidev
import time

# Create an SPI instance
spi = spidev.SpiDev()

# Open a connection to a specific bus and device (bus 1, device 0)
spi.open(1, 0)

# Set SPI speed and mode
spi.max_speed_hz = 50000
spi.mode = 0

# Function to send and receive specific data value from SPI device
def send_recv_spi(data_value):
    # Perform SPI transaction with the specific data value
    response = spi.xfer2([data_value])
    # Process the response to get the actual data
    received_data = response[0]
    return received_data

# Example data value to send
data_value = 0x55  # Example data value (0x55 in hexadecimal)

try:
    while True:
        received_data = send_recv_spi(data_value)
        print(f"Sent data: {data_value}, Received data: {received_data}")
        time.sleep(1)
except KeyboardInterrupt:
    print("Keyboard interrupt received. Exiting...")
finally:
    # Close the SPI connection
    spi.close()