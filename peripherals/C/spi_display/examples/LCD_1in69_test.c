#include "DEV_Config.h"
#include "LCD_1in69.h"
#include "GUI_Paint.h"
#include "GUI_BMP.h"
#include "test.h"
#include <stdio.h>      //printf()
#include <stdlib.h>     //exit()
#include <signal.h>     //signal()

void LCD_1IN69_test(void)
{
    // Exception handling:ctrl + c
    signal(SIGINT, Handler_1IN69_LCD);
    
    /* Module Init */
    if(DEV_ModuleInit() != 0){
        DEV_ModuleExit();
        exit(0);
    }
    
    /* LCD Init */
    printf("1.69inch LCD demo...\r\n");
    LCD_1IN69_Init(VERTICAL);
    LCD_1IN69_Clear(WHITE);
    LCD_1IN69_SetBacklight(100);
    
    UWORD *BlackImage;
    UDOUBLE Imagesize = LCD_1IN69_HEIGHT * LCD_1IN69_WIDTH * 2;
    printf("Imagesize = %d\r\n", Imagesize);
    if((BlackImage = (UWORD *)malloc(Imagesize)) == NULL) {
        printf("Failed to apply for black memory...\r\n");
        exit(0);
    }
    /*1.Create a new image cache named IMAGE_RGB and fill it with white*/
    Paint_NewImage(BlackImage, LCD_1IN69_WIDTH, LCD_1IN69_HEIGHT, 90, WHITE, 16);
    Paint_SelectImage(BlackImage);
    Paint_Clear(WHITE);
    /* GUI */
    
    printf("drawing...\r\n");
    /*2.Drawing on the image*/
    Paint_DrawPoint(12,28, BLACK, DOT_PIXEL_1X1,  DOT_FILL_RIGHTUP);
    Paint_DrawPoint(12,30, BLACK, DOT_PIXEL_2X2,  DOT_FILL_RIGHTUP);
    Paint_DrawPoint(12,33, BLACK, DOT_PIXEL_3X3, DOT_FILL_RIGHTUP);
    Paint_DrawPoint(12,38, BLACK, DOT_PIXEL_4X4, DOT_FILL_RIGHTUP);
    Paint_DrawPoint(12,43, BLACK, DOT_PIXEL_5X5, DOT_FILL_RIGHTUP);

    Paint_DrawLine( 20,  5, 80, 65, MAGENTA, DOT_PIXEL_2X2, LINE_STYLE_SOLID);
    Paint_DrawLine( 20, 65, 80,  5, MAGENTA, DOT_PIXEL_2X2, LINE_STYLE_SOLID);

    Paint_DrawLine( 148,  35, 208, 35, CYAN, DOT_PIXEL_1X1, LINE_STYLE_DOTTED);
    Paint_DrawLine( 178,   5,  178, 65, CYAN, DOT_PIXEL_1X1, LINE_STYLE_DOTTED);

    Paint_DrawRectangle(20, 5, 80, 65, RED, DOT_PIXEL_2X2,DRAW_FILL_EMPTY);
    Paint_DrawRectangle(85, 5, 145, 65, BLUE, DOT_PIXEL_2X2,DRAW_FILL_FULL);

    Paint_DrawCircle(178, 35, 30, GREEN, DOT_PIXEL_1X1, DRAW_FILL_EMPTY);
    Paint_DrawCircle(240, 35, 30, GREEN, DOT_PIXEL_1X1, DRAW_FILL_FULL);

    Paint_DrawString_EN(11, 70, "AaBbCc123", &Font16, RED, WHITE);
    Paint_DrawString_EN(11, 85, "AaBbCc123", &Font20, 0x000f, 0xfff0);
    Paint_DrawString_EN(11, 105, "AaBbCc123", &Font24, RED, WHITE);   
    Paint_DrawString_CN(11,125, "Î¢Ñ©µç×ÓAbc",  &Font24CN, WHITE, BLUE);

    // /*3.Refresh the picture in RAM to LCD*/
    LCD_1IN69_Display(BlackImage);
    DEV_Delay_ms(2000);

    Paint_NewImage(BlackImage, 40, 110, 90, WHITE, 16);
    PAINT_TIME sPaint_time; //time struct
    sPaint_time.Hour = 12;
    sPaint_time.Min = 34;
    sPaint_time.Sec = 56;
    UWORD num = 500;
    for (;;) {
        sPaint_time.Sec = sPaint_time.Sec + 1;
        if (sPaint_time.Sec == 60) {
            sPaint_time.Min = sPaint_time.Min + 1;
            sPaint_time.Sec = 0;
            if (sPaint_time.Min == 60) {
                sPaint_time.Hour =  sPaint_time.Hour + 1;
                sPaint_time.Min = 0;
                if (sPaint_time.Hour == 24) {
                    sPaint_time.Hour = 0;
                    sPaint_time.Min = 0;
                    sPaint_time.Sec = 0;
                }
            }
        }
        
        Paint_Clear(WHITE);
        Paint_DrawTime(5, 10, &sPaint_time, &Font20, WHITE, BLACK);

        if(num-- == 0) {
            break;
        }

        LCD_1IN69_DisplayWindows(130, 170, 170, 280, BlackImage);
        // DEV_Delay_ms(500);
    }
    DEV_Delay_ms(1000);
    
    // /* show bmp */
    printf("show bmp\r\n");
    Paint_NewImage(BlackImage, LCD_1IN69_WIDTH, LCD_1IN69_HEIGHT, 0, BLACK, 16);
    char *BmpPath[3] = {"./pic/LCD_1inch69_1.bmp", "./pic/LCD_1inch69_2.bmp", "./pic/LCD_1inch69_3.bmp"};
    for(UBYTE i=0; i<3; i++) {
        GUI_ReadBmp(BmpPath[i]);
        LCD_1IN69_Display(BlackImage);
        DEV_Delay_ms(2000);
    }
    
    // /* Module Exit */
    free(BlackImage);
    BlackImage = NULL;
    DEV_ModuleExit();
}

