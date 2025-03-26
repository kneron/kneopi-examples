/*****************************************************************************
* | File      	:   DEV_Config.c
* | Author      :   Waveshare team
* | Function    :   Hardware underlying interface
* | Info        :
*----------------
* |	This version:   V2.0
* | Date        :   2019-07-08
* | Info        :   Basic version
*
******************************************************************************/
#include "DEV_Config.h"
#include <stdlib.h>
#include <fcntl.h>

int GPIO_Handle;
int GPIO_Handle1;
int SPI_Handle;

/*****************************************
                PWM
*****************************************/
#define SYSFS_PWM_PATH "/sys/class/pwm/pwmchip0"
#define PWM_CHANNEL LCD_BL
#define PERIOD_NS 1000000  // Hardcoded period of 1 ms (1,000,000 ns)

// Helper function to write a value to a sysfs file
static int write_sysfs_file(const char *path, const char *value)
{
    int fd = open(path, O_WRONLY);
    if (fd < 0) {
        perror("Failed to open sysfs file");
        return -1;
    }

    ssize_t written = write(fd, value, strlen(value));
    if (written < 0) {
        perror("Failed to write to sysfs file");
        close(fd);
        return -1;
    }

    close(fd);
    return 0;
}

// Helper function to read a value from a sysfs file
static int read_sysfs_file(const char *path, char *buffer, size_t buffer_size)
{
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror("Failed to open sysfs file");
        return -1;
    }

    ssize_t read_bytes = read(fd, buffer, buffer_size - 1);
    if (read_bytes < 0) {
        perror("Failed to read from sysfs file");
        close(fd);
        return -1;
    }

    buffer[read_bytes] = '\0';  // Null-terminate the string
    close(fd);
    return 0;
}

static int unexport_pwm()
{
    char path[256];
    char buffer[64];

    // Construct the path to the unexport file
    snprintf(path, sizeof(path), "%s/unexport", SYSFS_PWM_PATH);
    snprintf(buffer, sizeof(buffer), "%d", PWM_CHANNEL);

    // Write the PWM channel number to the unexport file
    int ret = write_sysfs_file(path, buffer);
    if (ret < 0) {
        fprintf(stderr, "Failed to unexport PWM channel %d\n", PWM_CHANNEL);
        return -1;
    }

    printf("PWM channel %d unexported successfully\n", PWM_CHANNEL);
    return 0;
}

// Simplified function to set LED backlight brightness
UBYTE DEV_SetBacklight(UWORD duty_cycle_percent)
{
    char path[256];
    char buffer[64];
    int ret;

    // Validate duty cycle percentage
    if (duty_cycle_percent < 0 || duty_cycle_percent > 100) {
        fprintf(stderr, "Duty cycle must be between 0 and 100\n");
        return -1;
    }

    // Calculate duty cycle in nanoseconds
    int duty_cycle_ns = (duty_cycle_percent * PERIOD_NS) / 100;

    // Construct the path to the PWM channel
    snprintf(path, sizeof(path), "%s/pwm%d", SYSFS_PWM_PATH, PWM_CHANNEL);

    // Check if the PWM channel is exported
    if (access(path, F_OK) != 0) {
        // Export the PWM channel
        snprintf(path, sizeof(path), "%s/export", SYSFS_PWM_PATH);
        snprintf(buffer, sizeof(buffer), "%d", PWM_CHANNEL);
        ret = write_sysfs_file(path, buffer);
        if (ret < 0) {
            fprintf(stderr, "Failed to export PWM channel %d, %s %s\n", PWM_CHANNEL, path, buffer);
            return -1;
        }
    }

    // Construct the path to the PWM channel's enable file
    snprintf(path, sizeof(path), "%s/pwm%d/enable", SYSFS_PWM_PATH, PWM_CHANNEL);

    // Check if the PWM is enabled
    ret = read_sysfs_file(path, buffer, sizeof(buffer));
    if (ret < 0) {
        fprintf(stderr, "Failed to read PWM enable state\n");
        return -1;
    }

    int is_enabled = atoi(buffer);

    // Disable PWM if it is enabled
    if (is_enabled) {
        ret = write_sysfs_file(path, "0");
        if (ret < 0) {
            fprintf(stderr, "Failed to disable PWM\n");
            return -1;
        }
    }

    // Set the period (hardcoded)
    snprintf(path, sizeof(path), "%s/pwm%d/period", SYSFS_PWM_PATH, PWM_CHANNEL);
    snprintf(buffer, sizeof(buffer), "%d", PERIOD_NS);
    ret = write_sysfs_file(path, buffer);
    if (ret < 0) {
        fprintf(stderr, "Failed to set PWM period, %s %s\n", path, buffer);
        return -1;
    }

    // Set the duty cycle
    snprintf(path, sizeof(path), "%s/pwm%d/duty_cycle", SYSFS_PWM_PATH, PWM_CHANNEL);
    snprintf(buffer, sizeof(buffer), "%d", duty_cycle_ns);
    ret = write_sysfs_file(path, buffer);
    if (ret < 0) {
        fprintf(stderr, "Failed to set PWM duty cycle, %s %s\n", path, buffer);
        return -1;
    }

    // enable PWM if it was enabled
    snprintf(path, sizeof(path), "%s/pwm%d/enable", SYSFS_PWM_PATH, PWM_CHANNEL);
    ret = write_sysfs_file(path, "1");
    if (ret < 0) {
        fprintf(stderr, "Failed to re-enable PWM\n");
        return -1;
    }

    return 0;
}

/*****************************************
                GPIO
*****************************************/
void DEV_Digital_Write(UWORD Pin, UBYTE Value)
{
    if (Pin < 32) {
        lgGpioWrite(GPIO_Handle, Pin, Value);
    } else {
        Pin %= 32;
        lgGpioWrite(GPIO_Handle1, Pin, Value);
    }
}

UBYTE DEV_Digital_Read(UWORD Pin)
{
    UBYTE Read_value = 0;
    Read_value = lgGpioRead(GPIO_Handle,Pin);
    return Read_value;
}

void DEV_GPIO_Mode(UWORD Pin, UWORD Mode)
{
    int ret;

    if(Mode == 0 || Mode == LG_SET_INPUT) {
        lgGpioClaimInput(GPIO_Handle,LFLAGS,Pin);
        printf("IN Pin = %d\r\n",Pin);
    } else {
        if (Pin < 32) {
            ret = lgGpioClaimOutput(GPIO_Handle, LFLAGS, Pin, LG_LOW);
        } else {
            Pin %= 32;
            ret = lgGpioClaimOutput(GPIO_Handle1, LFLAGS, Pin, LG_LOW);
        }
        printf("OUT Pin = %d\r\n",Pin);
        printf("ret=%d\r\n", ret);
    }
}

/**
 * delay x ms
**/
void DEV_Delay_ms(UDOUBLE xms)
{
    lguSleep(xms/1000.0);
}

static void DEV_GPIO_Init(void)
{
    DEV_GPIO_Mode(LCD_RST, 1);
    DEV_GPIO_Mode(LCD_DC, 1);
}
/******************************************************************************
function:	Module Initialize, the library and initialize the pins, SPI protocol
parameter:
Info:
******************************************************************************/
UBYTE DEV_ModuleInit(void)
{
    char buffer[NUM_MAXBUF];
    FILE *fp;

    fp = popen("cat /proc/cpuinfo | grep 'Raspberry Pi 5'", "r");
    if (fp == NULL) {
        DEBUG("It is not possible to determine the model of the Raspberry PI\n");
        return -1;
    }

    if(fgets(buffer, sizeof(buffer), fp) != NULL) {
        GPIO_Handle = lgGpiochipOpen(4);
        if (GPIO_Handle < 0) {
            DEBUG( "gpiochip4 Export Failed\n");
            return -1;
        }
    } else {
        GPIO_Handle = lgGpiochipOpen(0);
        if (GPIO_Handle < 0) {
            DEBUG( "gpiochip0 Export Failed\n");
            return -1;
        }
        GPIO_Handle1 = lgGpiochipOpen(1);
        if (GPIO_Handle1 < 0) {
            DEBUG( "gpiochip1 Export Failed\n");
            return -1;
        }
    }
    SPI_Handle = lgSpiOpen(1, 0, 25000000, 0);
    DEV_GPIO_Init();
    DEV_SetBacklight(100);

    return 0;
}

void DEV_SPI_WriteByte(uint8_t Value)
{
    lgSpiWrite(SPI_Handle,(char*)&Value, 1);
}

void DEV_SPI_Write_nByte(uint8_t *pData, uint32_t Len)
{
    lgSpiWrite(SPI_Handle,(char*) pData, Len);
}

/******************************************************************************
function:	Module exits, closes SPI, GPIO and PWM
parameter:
Info:
******************************************************************************/
void DEV_ModuleExit(void)
{
    lgSpiClose(SPI_Handle);
    lgGpiochipClose(GPIO_Handle);
    lgGpiochipClose(GPIO_Handle1);
    unexport_pwm();
}
