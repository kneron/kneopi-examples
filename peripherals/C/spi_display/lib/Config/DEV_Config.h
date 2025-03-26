/*****************************************************************************
* | File      	:   DEV_Config.h
* | Author      :   Waveshare team
* | Function    :   Hardware underlying interface
* | Info        :
*----------------
* |	This version:   V2.0
* | Date        :   2019-07-08
* | Info        :   Basic version
*
******************************************************************************/
#ifndef _DEV_CONFIG_H_
#define _DEV_CONFIG_H_

#include "Debug.h"

#include <lgpio.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define LFLAGS 0
#define NUM_MAXBUF  4

/**
 * data
**/
#define UBYTE   uint8_t
#define UWORD   uint16_t
#define UDOUBLE uint32_t

// config for KNEO Pi
#define LCD_RST  52     // gpio_1[20], pin 13
#define LCD_DC   19     // gpio_0[19], pin 25
#define LCD_BL   1      // agpo_d[1],  pin 33

//LCD
#define LCD_RST_0		DEV_Digital_Write(LCD_RST,0)
#define LCD_RST_1		DEV_Digital_Write(LCD_RST,1)

#define LCD_DC_0		DEV_Digital_Write(LCD_DC,0)
#define LCD_DC_1		DEV_Digital_Write(LCD_DC,1)

#define LCD_SetBacklight(Value) DEV_SetBacklight(Value)

/*------------------------------------------------------------------------------------------------------*/
UBYTE DEV_ModuleInit(void);
void DEV_ModuleExit(void);

void DEV_GPIO_Mode(UWORD Pin, UWORD Mode);
void DEV_Digital_Write(UWORD Pin, UBYTE Value);
UBYTE DEV_Digital_Read(UWORD Pin);
void DEV_Delay_ms(UDOUBLE xms);

void DEV_SPI_WriteByte(UBYTE Value);
void DEV_SPI_Write_nByte(uint8_t *pData, uint32_t Len);
UBYTE DEV_SetBacklight(UWORD Value);
#endif
