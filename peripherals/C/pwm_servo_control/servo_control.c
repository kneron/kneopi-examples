#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

#define PWM_CHIP "/sys/class/pwm/pwmchip0"
#define PWM_CHANNEL "0"  // KNEO Pi, pin-32

// For TowerPro Micro Servo 9g SG90
#define PERIOD 20000000  // 20ms (50Hz)
#define MIN_DUTY 500000  // 0.5ms (0 degrees)
#define MAX_DUTY 2400000 // 2.4ms (180 degrees)

// Function to write a value to a sysfs file
void pwm_write(const char *path, const char *value) {
    int fd = open(path, O_WRONLY);
    if (fd < 0) {
        perror("Error opening PWM sysfs file");
        exit(EXIT_FAILURE);
    }
    write(fd, value, strlen(value));
    close(fd);
}

// Function to set the servo angle (0 to 180 degrees)
void set_servo_angle(int angle) {
    if (angle < 0) angle = 0;
    if (angle > 180) angle = 180;

    // Map angle (0-180) to duty cycle (500µs-2400µs)
    int duty_cycle = MIN_DUTY + (angle * (MAX_DUTY - MIN_DUTY) / 180);
    
    char duty_path[128];
    snprintf(duty_path, sizeof(duty_path), PWM_CHIP "/pwm%s/duty_cycle", PWM_CHANNEL);
    
    char duty_value[32];
    snprintf(duty_value, sizeof(duty_value), "%d", duty_cycle);
    pwm_write(duty_path, duty_value);
}

int main() {
    char path[128];

    // Export PWM channel
    snprintf(path, sizeof(path), PWM_CHIP "/export");
    pwm_write(path, PWM_CHANNEL);

    usleep(100000); // Allow some time for the PWM channel to be available

    // Set period (20ms)
    snprintf(path, sizeof(path), PWM_CHIP "/pwm%s/period", PWM_CHANNEL);
    char period_value[32];
    snprintf(period_value, sizeof(period_value), "%d", PERIOD);
    pwm_write(path, period_value);

    // Enable PWM
    snprintf(path, sizeof(path), PWM_CHIP "/pwm%s/enable", PWM_CHANNEL);
    pwm_write(path, "1");

    // Sweep servo from 0 to 180 degrees and back
    for (int angle = 0; angle <= 180; angle += 10) {
        set_servo_angle(angle);
        usleep(500000);  // Wait 500ms
    }
    for (int angle = 180; angle >= 0; angle -= 10) {
        set_servo_angle(angle);
        usleep(500000);
    }

    // Disable PWM before exiting
    snprintf(path, sizeof(path), PWM_CHIP "/pwm%s/enable", PWM_CHANNEL);
    pwm_write(path, "0");

    return 0;
}

