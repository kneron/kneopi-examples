/**
 * @file        helper_functions.c
 * @brief       implementation of helper functions
 * @version     0.1
 * @date        2021-03-22
 *
 * @copyright   Copyright (c) 2021 Kneron Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#include <limits.h>

#include "kp_inference.h"
#include "helper_functions.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static struct timeval time_begin;
static struct timeval time_end;

static float pow2(int exp)
{
    if (0 <= exp) {
        return (float)(0x1ULL << exp);
    } else {
        return (float)1 / (float)(0x1ULL << abs(exp));
    }
}

static int kneron_round(float _input)
{
    if (_input > (float)INT_MAX) {
        return INT_MAX;
    } else if (_input < (float)INT_MIN) {
        return INT_MIN;
    }

    int f               = (int)_input;
    float fractionPart  = _input - f;

    if (fractionPart < 0) {
        fractionPart *= -1;
    }

    bool oddFlag = true;

    if (f % 2 == 0) {
        oddFlag = false;
    }

    if (fractionPart > 0.5) {
        // always round to nearest
        if (_input >= 0) {
            ++f;
        } else {
            --f;
        }
    } else if (fractionPart == 0.5) {
        // ties to even depending odd or even
        if (oddFlag) {
            if (_input >= 0) {
                ++f;
            } else {
                --f;
            }
        }
    }

    return f;
}

void helper_measure_time_begin()
{
    gettimeofday(&time_begin, NULL);
}

void helper_measure_time_end(double *measued_time)
{
    gettimeofday(&time_end, NULL);

    *measued_time = (double)(time_end.tv_sec - time_begin.tv_sec) + (double)(time_end.tv_usec - time_begin.tv_usec) * .000001;
}

#pragma pack(push, 1)
typedef struct
{
    unsigned short int type;
    unsigned int size;
    unsigned short int reserved1, reserved2;
    unsigned int offset;
} FILEHEADER;
#pragma pack(pop)

#pragma pack(push, 1)
typedef struct
{
    unsigned int size;             /* Info Header size in bytes */
    int width, height;             /* Width and height of image */
    unsigned short int planes;     /* Number of colour planes */
    unsigned short int bits;       /* Bits per pixel */
    unsigned int compression;      /* Compression type */
    unsigned int imagesize;        /* Image size in bytes */
    int xresolution, yresolution;  /* Pixels per meter */
    unsigned int ncolours;         /* Number of colours */
    unsigned int importantcolours; /* Important colours */
} INFOHEADER;
#pragma pack(pop)

// coverity[ -taint_source : arg-0 ]
static size_t custom_fread(void *ptr, size_t size, size_t count, FILE *stream)
{
    size_t read_size = fread(ptr, size, count, stream);

    return read_size;
}

static void generate_ycbcr_to_ycbcr_order_mapping(kp_image_format_t in_format, kp_image_format_t out_format, int order_mapping[])
{
    // initialize
    int y0_index = 0;
    int cb_index = 0;
    int y1_index = 0;
    int cr_index = 0;

    switch (in_format)
    {
        case KP_IMAGE_FORMAT_YCBCR422_CRY1CBY0:
            cr_index = 0;
            y1_index = 1;
            cb_index = 2;
            y0_index = 3;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_CBY1CRY0:
            cb_index = 0;
            y1_index = 1;
            cr_index = 2;
            y0_index = 3;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_Y1CRY0CB:
            y1_index = 0;
            cr_index = 1;
            y0_index = 2;
            cb_index = 3;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_Y1CBY0CR:
            y1_index = 0;
            cb_index = 1;
            y0_index = 2;
            cr_index = 3;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_CRY0CBY1:
            cr_index = 0;
            y0_index = 1;
            cb_index = 2;
            y1_index = 3;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_CBY0CRY1:
            cb_index = 0;
            y0_index = 1;
            cr_index = 2;
            y1_index = 3;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_Y0CRY1CB:
            y0_index = 0;
            cr_index = 1;
            y1_index = 2;
            cb_index = 3;
            break;
        case KP_IMAGE_FORMAT_YUYV:
        case KP_IMAGE_FORMAT_YCBCR422_Y0CBY1CR:
            y0_index = 0;
            cb_index = 1;
            y1_index = 2;
            cr_index = 3;
            break;
        default:
            printf("input format is not supported\n");
            break;
    }

    switch (out_format)
    {
        case KP_IMAGE_FORMAT_YCBCR422_CRY1CBY0:
            order_mapping[0] = cr_index;
            order_mapping[1] = y1_index;
            order_mapping[2] = cb_index;
            order_mapping[3] = y0_index;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_CBY1CRY0:
            order_mapping[0] = cb_index;
            order_mapping[1] = y1_index;
            order_mapping[2] = cr_index;
            order_mapping[3] = y0_index;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_Y1CRY0CB:
            order_mapping[0] = y1_index;
            order_mapping[1] = cr_index;
            order_mapping[2] = y0_index;
            order_mapping[3] = cb_index;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_Y1CBY0CR:
            order_mapping[0] = y1_index;
            order_mapping[1] = cb_index;
            order_mapping[2] = y0_index;
            order_mapping[3] = cr_index;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_CRY0CBY1:
            order_mapping[0] = cr_index;
            order_mapping[1] = y0_index;
            order_mapping[2] = cb_index;
            order_mapping[3] = y1_index;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_CBY0CRY1:
            order_mapping[0] = cb_index;
            order_mapping[1] = y0_index;
            order_mapping[2] = cr_index;
            order_mapping[3] = y1_index;
            break;
        case KP_IMAGE_FORMAT_YCBCR422_Y0CRY1CB:
            order_mapping[0] = y0_index;
            order_mapping[1] = cr_index;
            order_mapping[2] = y1_index;
            order_mapping[3] = cb_index;
            break;
        case KP_IMAGE_FORMAT_YUYV:
        case KP_IMAGE_FORMAT_YCBCR422_Y0CBY1CR:
            order_mapping[0] = y0_index;
            order_mapping[1] = cb_index;
            order_mapping[2] = y1_index;
            order_mapping[3] = cr_index;
            break;
        default:
            printf("output format is not supported\n");
            break;
    }
}

static void dump_bmp_file_from_bmp_pixel_data(const char *out_bmp_path, int bmp_width, int bmp_height, unsigned char *pixel_data)
{
    // Open .bmp file to write
    FILE *out_bmp_file = fopen(out_bmp_path, "wb");
    if (!out_bmp_file)
    {
        printf("file read failed\n");
        return;
    }

    // Prepare BMP file header
    FILEHEADER header1 = {0};
    INFOHEADER header2 = {0};

    unsigned int bytes_per_pixel_bmp = 3; // This is constant (also for RAW8) for all supported raw formats
    int bmp_buf_size = 0;
    int padding_byte_num = 0;

    header1.type = 0x4D42; // 'B' and 'M' in ASCII code (little endian)
    header1.size = sizeof(FILEHEADER) + sizeof(INFOHEADER) + ((uint32_t)bmp_width * (uint32_t)bmp_height * bytes_per_pixel_bmp);
    header1.offset = sizeof(FILEHEADER) + sizeof(INFOHEADER);

    header2.size = sizeof(INFOHEADER);
    header2.width = bmp_width;
    header2.height = bmp_height;
    header2.planes = 1;
    header2.bits = bytes_per_pixel_bmp * 8;
    header2.imagesize = (uint32_t)bmp_width * (uint32_t)bmp_height * bytes_per_pixel_bmp;

    // length of bmp image width in byte should be aligned with 4
    if (bmp_width * bytes_per_pixel_bmp % 4 != 0)
        padding_byte_num = 4 - bmp_width * bytes_per_pixel_bmp % 4;

    bmp_buf_size = (bmp_width * bytes_per_pixel_bmp + padding_byte_num) * bmp_height;

    fwrite(&header1, 1, sizeof(header1), out_bmp_file);
    fwrite(&header2, 1, sizeof(header2), out_bmp_file);

    fwrite(pixel_data, 1, bmp_buf_size, out_bmp_file);

    fclose(out_bmp_file);
	return;
}

char *helper_bmp_file_to_raw_buffer(const char *file_path, int *width, int *height, kp_image_format_t format)
{
    FILEHEADER header1;
    INFOHEADER header2;

    unsigned short *b16_buf = NULL;
    unsigned char  *b8_buf  = NULL;
    unsigned char *bmp_buf = NULL;

    char *raw_buf = NULL;

    if(NULL == file_path)
        return NULL;

    FILE *bmp_file = fopen(file_path, "rb");

    if (!bmp_file)
    {
        printf("file read failed\n");
        return NULL;
    }

    size_t read_size = custom_fread(&header1, 1, sizeof(header1), bmp_file);

    if (0 == read_size) {
        printf("Error! Read file header failed\n");
        fclose(bmp_file);
        return NULL;
    }

    read_size = custom_fread(&header2, 1, sizeof(header2), bmp_file);

    if (0 == read_size) {
        printf("Error! Read info header failed\n");
        fclose(bmp_file);
        return NULL;
    }

    *width = header2.width;
    *height = header2.height;

    unsigned short bytes_per_pixel = header2.bits / 8;

    // length of bmp image width in byte should be aligned with 4
    int padding_byte_num = 0;
    if (header2.width * bytes_per_pixel % 4 != 0)
        padding_byte_num = 4 - header2.width * bytes_per_pixel % 4;

    // bmp_buf is used to store pixel data of bmp image
    int bmp_buf_size = (header2.width * bytes_per_pixel + padding_byte_num) * header2.height; // image width of bmp file should be a multiple of 4
    bmp_buf = (unsigned char *)malloc(bmp_buf_size);
    if (NULL == bmp_buf) {
        fclose(bmp_file);
        printf("Error! malloc memory for image buffer failed\n");
        goto err;
    }

    if (header1.offset != sizeof(FILEHEADER) + sizeof(INFOHEADER))
    {
        // move the position indicator of the bmp_file stream to the data offset specified by the FILEHEADER
        int ret = fseek(bmp_file, (long)header1.offset, SEEK_SET);

        if (0 != ret) {
            fclose(bmp_file);
            goto err;
        }
    }

    // read pixel data from bmp_file
    read_size = custom_fread(bmp_buf, 1, bmp_buf_size, bmp_file);

    if (0 == read_size) {
        printf("Error! %s(): read bmp failed\n", __FUNCTION__);
        fclose(bmp_file);
        goto err;
    }

    fclose(bmp_file);

    if (header2.bits == 24)
    {
        switch (format)
        {
        case KP_IMAGE_FORMAT_RGB565:
        {
            b16_buf = (unsigned short *)malloc(header2.width * header2.height * 2);
            if ( NULL == b16_buf ) {
                printf("Error! malloc memory for converted data failed\n");
                goto err;
            }
            int rgb565_pos = 0;

            for (int row = header2.height - 1; row >= 0; row--)
            {
                int bmp_pos = (header2.width * 3 + padding_byte_num) * row; // we only support bmp file with 3 bytes per pixel
                for (int col = 0; col < header2.width; col++)
                {
                    unsigned char blue = bmp_buf[bmp_pos];
                    unsigned char green = bmp_buf[bmp_pos + 1];
                    unsigned char red = bmp_buf[bmp_pos + 2];

                    b16_buf[rgb565_pos] = ((red & 0b11111000) << 8) | ((green & 0b11111100) << 3) | (blue >> 3);

                    bmp_pos += 3;
                    rgb565_pos++;
                }
            }

            raw_buf = (char *)b16_buf;
            break;
        }
        case KP_IMAGE_FORMAT_RGBA8888:
        {
            b8_buf = (unsigned char *)malloc(header2.width * header2.height * 4);
            if ( NULL == b8_buf ) {
                printf("Error! malloc memory for converted data failed\n");
                goto err;
            }
            int rgba8888_pos = 0;

            for (int row = header2.height - 1; row >= 0; row--)
            {
                int bmp_pos = (header2.width * 3 + padding_byte_num) * row; // we only support bmp file with 3 bytes per pixel
                for (int col = 0; col < header2.width; col++)
                {
                    unsigned char blue = bmp_buf[bmp_pos];
                    unsigned char green = bmp_buf[bmp_pos + 1];
                    unsigned char red = bmp_buf[bmp_pos + 2];

                    b8_buf[rgba8888_pos++] = red;
                    b8_buf[rgba8888_pos++] = green;
                    b8_buf[rgba8888_pos++] = blue;
                    b8_buf[rgba8888_pos++] = 0;

                    bmp_pos += 3;
                }
            }

            raw_buf = (char *)b8_buf;
            break;
        }
        case KP_IMAGE_FORMAT_YUYV:
        case KP_IMAGE_FORMAT_YCBCR422_CRY1CBY0:
        case KP_IMAGE_FORMAT_YCBCR422_CBY1CRY0:
        case KP_IMAGE_FORMAT_YCBCR422_Y1CRY0CB:
        case KP_IMAGE_FORMAT_YCBCR422_Y1CBY0CR:
        case KP_IMAGE_FORMAT_YCBCR422_CRY0CBY1:
        case KP_IMAGE_FORMAT_YCBCR422_CBY0CRY1:
        case KP_IMAGE_FORMAT_YCBCR422_Y0CRY1CB:
        case KP_IMAGE_FORMAT_YCBCR422_Y0CBY1CR:
        {
            b8_buf = (unsigned char *)malloc(header2.width * header2.height * 2);
            if ( NULL == b8_buf ) {
                printf("Error! malloc memory for converted data failed\n");
                goto err;
            }
            int write_pos = 0;

            // generate the order mappings of (y0, cb, y1, cr) to (cr, y1, cb, y0), (cb, y1, cr, y0), ... described in the yuyv or ycbcr image format
            int order_mapping[4] = {0};
            generate_ycbcr_to_ycbcr_order_mapping(KP_IMAGE_FORMAT_YCBCR422_Y0CBY1CR, format, order_mapping);

            for (int row = header2.height - 1; row >= 0; row--)
            {
                int bmp_pos = (header2.width * 3 + padding_byte_num) * row; // we only support bmp file with 3 bytes per pixel

                // FIXME? How if the width is an odd number?
                for (int col = 0; col < header2.width / 2; col++)
                {
                    unsigned char blue0 = bmp_buf[bmp_pos];
                    unsigned char green0 = bmp_buf[bmp_pos + 1];
                    unsigned char red0 = bmp_buf[bmp_pos + 2];

                    unsigned char blue1 = bmp_buf[bmp_pos + 3];
                    unsigned char green1 = bmp_buf[bmp_pos + 4];
                    unsigned char red1 = bmp_buf[bmp_pos + 5];

                    // FIXME: use integer method to speed up
                    float output[4];
                    output[0] = 0.299 * red0 + 0.587 * green0 + 0.114 * blue0; // y0
                    output[1] = -0.169 * (red0 + red1) / 2 - 0.331 * (green0 + green1) / 2 + 0.5 * (blue0 + blue1) / 2 + 128; // cb
                    output[2] = 0.299 * red1 + 0.587 * green1 + 0.114 * blue1; // y1
                    output[3] = 0.5 * (red0 + red1) / 2 - 0.419 * (green0 + green1) / 2 - 0.081 * (blue0 + blue1) / 2 + 128;  // cr

                    b8_buf[write_pos++] = (unsigned char)output[order_mapping[0]];
                    b8_buf[write_pos++] = (unsigned char)output[order_mapping[1]];
                    b8_buf[write_pos++] = (unsigned char)output[order_mapping[2]];
                    b8_buf[write_pos++] = (unsigned char)output[order_mapping[3]];

                    bmp_pos += 6;
                }
            }

            raw_buf = (char *)b8_buf;
            break;
        }
        case KP_IMAGE_FORMAT_YUV420:
        {
            if ((0 != header2.width % 2) || (0 != header2.height % 2)) {
                printf("Error! width or height is not even number\n");
                goto err;
            }

            b8_buf = (unsigned char *)malloc(header2.width * header2.height * 1.5);
            if ( NULL == b8_buf ) {
                printf("Error! malloc memory for converted data failed\n");
                goto err;
            }

            int y_bias = header2.width * header2.height;
            int u_bias = y_bias / 4;
            int u_pos = y_bias;
            int v_pos = y_bias + u_bias;

            for (int row = header2.height - 1; row >= 0; row -= 2)
            {
                int bmp_pos_row_0 = (header2.width * 3 + padding_byte_num) * row;       // we only support bmp file with 3 bytes per pixel
                int bmp_pos_row_1 = (header2.width * 3 + padding_byte_num) * (row - 1); // we only support bmp file with 3 bytes per pixel
                int y_write_pos_row_0 = header2.width * (header2.height - row - 1);
                int y_write_pos_row_1 = header2.width * (header2.height - row);

                for (int col = 0; col < header2.width; col += 2)
                {
                    unsigned char blue0 = bmp_buf[bmp_pos_row_0];
                    unsigned char green0 = bmp_buf[bmp_pos_row_0 + 1];
                    unsigned char red0 = bmp_buf[bmp_pos_row_0 + 2];

                    unsigned char blue1 = bmp_buf[bmp_pos_row_0 + 3];
                    unsigned char green1 = bmp_buf[bmp_pos_row_0 + 4];
                    unsigned char red1 = bmp_buf[bmp_pos_row_0 + 5];

                    unsigned char blue2 = bmp_buf[bmp_pos_row_1];
                    unsigned char green2 = bmp_buf[bmp_pos_row_1 + 1];
                    unsigned char red2 = bmp_buf[bmp_pos_row_1 + 2];

                    unsigned char blue3 = bmp_buf[bmp_pos_row_1 + 3];
                    unsigned char green3 = bmp_buf[bmp_pos_row_1 + 4];
                    unsigned char red3 = bmp_buf[bmp_pos_row_1 + 5];

                    float y[2][2];
                    float u, v;

                    y[0][0] = 0.299 * red0 + 0.587 * green0 + 0.114 * blue0;
                    y[0][1] = 0.299 * red1 + 0.587 * green1 + 0.114 * blue1;
                    y[1][0] = 0.299 * red2 + 0.587 * green2 + 0.114 * blue2;
                    y[1][1] = 0.299 * red3 + 0.587 * green3 + 0.114 * blue3;

                    u = -0.169 * (red0 + red1 + red2 + red3) / 4 - 0.331 * (green0 + green1 + green2 + green3) / 4 + 0.5 * (blue0 + blue1 + blue2 + blue3) / 4 + 128;
                    v = 0.5 * (red0 + red1 + red2 + red3) / 4 - 0.419 * (green0 + green1 + green2 + green3) / 4 - 0.081 * (blue0 + blue1 + blue2 + blue3) / 4 + 128;

                    b8_buf[y_write_pos_row_0++] = (unsigned char)y[0][0];
                    b8_buf[y_write_pos_row_0++] = (unsigned char)y[0][1];
                    b8_buf[y_write_pos_row_1++] = (unsigned char)y[1][0];
                    b8_buf[y_write_pos_row_1++] = (unsigned char)y[1][0];
                    b8_buf[u_pos++] = (unsigned char)u;
                    b8_buf[v_pos++] = (unsigned char)v;

                    bmp_pos_row_0 += 6;
                    bmp_pos_row_1 += 6;
                }
            }

            raw_buf = (char *)b8_buf;
            break;
        }
        case KP_IMAGE_FORMAT_UNKNOWN:
        default:
            printf("image format is not supported\n");
            break;
        }
    }
    else
    {
        printf("support only 24 bit bmp\n");
    }

    free(bmp_buf);

    return raw_buf;

err:
    free(bmp_buf);
    free(b16_buf);
    free(b8_buf);

    return NULL;
}

char *helper_bin_file_to_raw_buffer(const char *file_path, int width, int height, kp_image_format_t format)
{
    if (width <= 0 || height <= 0)
        return NULL;

    float bytes_per_pixel = 0;

    switch (format)
    {
    case KP_IMAGE_FORMAT_RGB565:
    case KP_IMAGE_FORMAT_YUYV:
    case KP_IMAGE_FORMAT_YCBCR422_CRY1CBY0:
    case KP_IMAGE_FORMAT_YCBCR422_CBY1CRY0:
    case KP_IMAGE_FORMAT_YCBCR422_Y1CRY0CB:
    case KP_IMAGE_FORMAT_YCBCR422_Y1CBY0CR:
    case KP_IMAGE_FORMAT_YCBCR422_CRY0CBY1:
    case KP_IMAGE_FORMAT_YCBCR422_CBY0CRY1:
    case KP_IMAGE_FORMAT_YCBCR422_Y0CRY1CB:
    case KP_IMAGE_FORMAT_YCBCR422_Y0CBY1CR:
        bytes_per_pixel = 2;
        break;
    case KP_IMAGE_FORMAT_RGBA8888:
        bytes_per_pixel = 4;
        break;
    case KP_IMAGE_FORMAT_RAW8:
        bytes_per_pixel = 1;
        break;
    case KP_IMAGE_FORMAT_YUV420:
        if ((0 != width % 2) || (0 != height % 2)) {
            printf("image width and height must be even number\n");
            return NULL;
        }

        bytes_per_pixel = 1.5;
        break;
    default:
    case KP_IMAGE_FORMAT_UNKNOWN:
        printf("image format is not supported\n");
        return NULL;
    }

    FILE *bin_file = fopen(file_path, "rb");
    if (!bin_file)
    {
        printf("%s(): fopen failed, file:%s, %s\n", __FUNCTION__, file_path, strerror(errno));
        return NULL;
    }

    fseek(bin_file, 0, SEEK_END);
    long file_size = ftell(bin_file); //get the size

    if (file_size != width * height * bytes_per_pixel)
    {
        printf("%s(): file size does not match input image width, height or format\n", __FUNCTION__);
        fclose(bin_file);
        return NULL;
    }

    char *buffer = (char *)malloc(file_size);
    if(NULL == buffer) {
        printf("Error! %s(): malloc for buffer failed\n", __FUNCTION__);
        fclose(bin_file);
        return NULL;
    }

    fseek(bin_file, 0, SEEK_SET); //move to beginning

    size_t read_size = custom_fread(buffer, 1, file_size, bin_file);
    if (read_size != (size_t)file_size)
    {
        printf("%s(): fread failed, file size: %u, read size %u\n", __FUNCTION__,
               (unsigned int)file_size, (unsigned int)read_size);
        free(buffer);
        buffer = NULL;
    }

    fclose(bin_file);

    return buffer;
}

static void draw_box_on_bmp_raw_buffer(kp_bounding_box_t boxes[], int box_count, int width, int height, unsigned char *bmp_buf, int padding_byte_num)
{
    const int thickness = 2;

    int bmp_buf_size = (width * 3 + padding_byte_num) * height;

    for (int i = 0; i < box_count; i++)
    {
        if (boxes[i].x2 >= (float)width)
            boxes[i].x2 = (float)width - 1;

        if (boxes[i].y2 >= (float)height)
            boxes[i].y2 = (float)height - 1;

        int bmp_x1 = (int)boxes[i].x1;
        int bmp_y1 = (int)(height - boxes[i].y1 - 1);
        int bmp_x2 = (int)boxes[i].x2;
        int bmp_y2 = (int)(height - boxes[i].y2 - 1);

        int pos;
        int diff_x = bmp_x2 - bmp_x1;
        int diff_y = bmp_y1 - bmp_y2;

        int class = boxes[i].class_num;
        int bb = 100 + (0 + 25 * class) % 156;
        int gg = 100 + (0 + 80 + 40 * class) % 156;
        int rr = 100 + (0 + 120 + 60 * class) % 156;

        // (x1,y1) -> (x2,y1)
        for (int k = 0; k < thickness; k++){
            pos = (width * 3 + padding_byte_num) * (bmp_y1-k) + bmp_x1 * 3;
            for (int j = 0; j <= diff_x; j++) // also draw the right top corner
            {
                if ((bmp_buf_size <= (pos + thickness)) || (0 > pos))
                    break;

                bmp_buf[pos] = bb;
                bmp_buf[pos + 1] = gg;
                bmp_buf[pos + 2] = rr;
                pos += 3;
            }
        }

        // (x1,y2) -> (x2,y2)
        for (int k = 0; k < thickness; k++){
            pos = (width * 3 + padding_byte_num) * (bmp_y2+k) + bmp_x1 * 3;

            for (int j = 0; j < diff_x; j++)
            {
                if ((bmp_buf_size <= (pos + thickness)) || (0 > pos))
                    break;

                bmp_buf[pos] = bb;
                bmp_buf[pos + 1] = gg;
                bmp_buf[pos + 2] = rr;
                pos += 3;
            }
        }

        // (x1,y1) -> (x1,y2)
        for (int k = 0; k < thickness; k++){
            pos = (width * 3 + padding_byte_num) * bmp_y2 + (bmp_x1+k) * 3;

            for (int j = 0; j < diff_y; j++)
            {
                if ((bmp_buf_size <= (pos + thickness)) || (0 > pos))
                    break;

                bmp_buf[pos] = bb;
                bmp_buf[pos + 1] = gg;
                bmp_buf[pos + 2] = rr;
                pos += width * 3 + padding_byte_num;
            }
        }

        // (x2,y1) -> (x2,y2)
        for (int k = 0; k < thickness; k++){
            pos = (width * 3 + padding_byte_num) * bmp_y2 + (bmp_x2-k) * 3;

            for (int j = 0; j < diff_y; j++)
            {
                if ((bmp_buf_size <= (pos + thickness)) || (0 > pos))
                    break;

                bmp_buf[pos] = bb;
                bmp_buf[pos + 1] = gg;
                bmp_buf[pos + 2] = rr;
                pos += width * 3 + padding_byte_num;
            }
        }
    }
}

void helper_draw_box_on_bmp(const char *in_bmp_path, const char *out_bmp_path, kp_bounding_box_t boxes[], int box_count)
{
    FILEHEADER header1;
    INFOHEADER header2;

    unsigned char *bmp_buf = NULL;
    unsigned char *extended_info_header = NULL;

    size_t extended_info_header_length = 0;
    unsigned short bytes_per_pixel = 0;
    int padding_byte_num = 0;
    int bmp_buf_size = 0;

    FILE *out_bmp_file;

    FILE *in_bmp_file = fopen(in_bmp_path, "rb");
    if (!in_bmp_file)
    {
        printf("file read failed\n");
        return;
    }

    size_t read_size = custom_fread(&header1, 1, sizeof(header1), in_bmp_file);

    if (0 == read_size) {
        printf("Error! %s(): Read file header failed\n", __FUNCTION__);
        goto err;
    }

    read_size = custom_fread(&header2, 1, sizeof(header2), in_bmp_file);

    if (0 == read_size) {
        printf("Error! %s(): Read info header failed\n", __FUNCTION__);
        goto err;
    }

    bytes_per_pixel = header2.bits / 8;

    // length of bmp image width in byte should be aligned with 4
    if (header2.width * bytes_per_pixel % 4 != 0)
        padding_byte_num = 4 - header2.width * bytes_per_pixel % 4;

    // bmp_buf is used to store pixel data of bmp image
    bmp_buf_size = (header2.width * bytes_per_pixel + padding_byte_num) * header2.height; // image width of bmp file should be a multiple of 4
    bmp_buf = (unsigned char *)malloc(bmp_buf_size);
    if(NULL == bmp_buf) {
        printf("Error! %s(): malloc for drawing buffer failed\n", __FUNCTION__);
        goto err;
    }

    // read extended info header if the length of info header exceeds sizeof(INFOHEADER)
    if (header1.offset != sizeof(FILEHEADER) + sizeof(INFOHEADER))
    {
        extended_info_header_length = header1.offset - (sizeof(FILEHEADER) + sizeof(INFOHEADER));
        extended_info_header = (unsigned char *)malloc(extended_info_header_length);
        if (NULL == extended_info_header ) {
            printf("Error! %s(): malloc for extended info header failed\n", __FUNCTION__);
            goto err;
        }
        read_size = custom_fread(extended_info_header, 1, extended_info_header_length, in_bmp_file);
        if (0 == read_size) {
            printf("Error! %s(): read extended info header failed\n", __FUNCTION__);
            goto err;
        }
    }

    read_size = custom_fread(bmp_buf, 1, bmp_buf_size, in_bmp_file);

    if (0 == read_size) {
        printf("Error! %s(): read bmp failed\n", __FUNCTION__);
        goto err;
    }

    fclose(in_bmp_file);

    if (header2.bits == 24)
    {
        draw_box_on_bmp_raw_buffer(boxes, box_count, header2.width, header2.height, bmp_buf, padding_byte_num);
    }
    else
    {
        printf("support only 24 bit bmp\n");
    }

    out_bmp_file = fopen(out_bmp_path, "wb");

    if (out_bmp_file) {
        fwrite(&header1, 1, sizeof(header1), out_bmp_file);
        fwrite(&header2, 1, sizeof(header2), out_bmp_file);

        // write extended info header
        if (extended_info_header)
        {
            fwrite(extended_info_header, 1, extended_info_header_length, out_bmp_file);
            free(extended_info_header);
            extended_info_header = NULL;
        }

        fwrite(bmp_buf, 1, bmp_buf_size, out_bmp_file);
        fclose(out_bmp_file);
    } else {
        printf("Error! %s() open output file failed\n", __FUNCTION__);
    }

    free(bmp_buf);

    if (NULL != extended_info_header)
        free(extended_info_header);

    return;

err:
    fclose(in_bmp_file);

    if (NULL != bmp_buf)
        free(bmp_buf);

    if (NULL != extended_info_header)
        free(extended_info_header);

    return;
}

static unsigned char clamp_to_0_255(float num)
{
    if (num > 255)
        return 255;
    else if (num < 0)
        return 0;
    else
        return (unsigned char)num;
}

// Passing bmp_padding_byte_num to avoid recalculation
// bmp_buf is used to store pixel data of bmp image
static void convert_bin_to_bmp_pixel_data(unsigned char *bin_buf, unsigned char *bmp_buf, int width, int height, kp_image_format_t bin_format, int bmp_padding_byte_num)
{
    switch (bin_format)
    {
    case KP_IMAGE_FORMAT_RGB565:
    {
        unsigned short *rgb565_buf = (unsigned short *)bin_buf;

        int rgb565_pos = 0;

        for (int row = height - 1; row >= 0; row--)
        {
            int bmp_pos = (width * 3 + bmp_padding_byte_num) * row; // we only support bmp file with 3 bytes per pixel
            for (int col = 0; col < width; col++)
            {
                bmp_buf[bmp_pos] = (rgb565_buf[rgb565_pos] << 3) & 0b11111000;     // blue
                bmp_buf[bmp_pos + 1] = (rgb565_buf[rgb565_pos] >> 3) & 0b11111100; // green
                bmp_buf[bmp_pos + 2] = (rgb565_buf[rgb565_pos] >> 8) & 0b11111000; // red

                bmp_pos += 3;
                rgb565_pos++;
            }
        }
        break;
    }
    case KP_IMAGE_FORMAT_RGBA8888:
    {
        // Just an alias for buffer naming consistency
        unsigned char *rgba8888_buf = bin_buf;

        int rgba8888_pos = 0;

        for (int row = height - 1; row >= 0; row--)
        {
            int bmp_pos = (width * 3 + bmp_padding_byte_num) * row; // we only support bmp file with 3 bytes per pixel
            for (int col = 0; col < width; col++)
            {
                bmp_buf[bmp_pos] = rgba8888_buf[rgba8888_pos + 2];     // blue
                bmp_buf[bmp_pos + 1] = rgba8888_buf[rgba8888_pos + 1]; // green
                bmp_buf[bmp_pos + 2] = rgba8888_buf[rgba8888_pos];     // red

                bmp_pos += 3;
                rgba8888_pos += 4;
            }
        }
        break;
    }
    case KP_IMAGE_FORMAT_YUYV:
    case KP_IMAGE_FORMAT_YCBCR422_CRY1CBY0:
    case KP_IMAGE_FORMAT_YCBCR422_CBY1CRY0:
    case KP_IMAGE_FORMAT_YCBCR422_Y1CRY0CB:
    case KP_IMAGE_FORMAT_YCBCR422_Y1CBY0CR:
    case KP_IMAGE_FORMAT_YCBCR422_CRY0CBY1:
    case KP_IMAGE_FORMAT_YCBCR422_CBY0CRY1:
    case KP_IMAGE_FORMAT_YCBCR422_Y0CRY1CB:
    case KP_IMAGE_FORMAT_YCBCR422_Y0CBY1CR:
    {
        // Just an alias for buffer naming consistency
        unsigned char *ycbcr_buf = bin_buf;

        int ycbcr_pos = 0;

        // generate the order mappings of (cr, y1, cb, y0), (cb, y1, cr, y0) to (y0, cb, y1, cr)... described in the yuyv or ycbcr image format
        int order_mapping[4] = {0};
        generate_ycbcr_to_ycbcr_order_mapping(bin_format, KP_IMAGE_FORMAT_YCBCR422_Y0CBY1CR, order_mapping);

        for (int row = height - 1; row >= 0; row--)
        {
            int bmp_pos = (width * 3 + bmp_padding_byte_num) * row; // we only support bmp file with 3 bytes per pixel

            // FIXME: How if the width is an odd number?
            for (int col = 0; col < width / 2; col++)
            {
                unsigned char output[4];
                output[0] = ycbcr_buf[ycbcr_pos++]; // one of (y0, cb, y1, cr)
                output[1] = ycbcr_buf[ycbcr_pos++]; // one of (y0, cb, y1, cr)
                output[2] = ycbcr_buf[ycbcr_pos++]; // one of (y0, cb, y1, cr)
                output[3] = ycbcr_buf[ycbcr_pos++]; // one of (y0, cb, y1, cr)

                unsigned char y0 = output[order_mapping[0]];
                unsigned char cb = output[order_mapping[1]];
                unsigned char y1 = output[order_mapping[2]];
                unsigned char cr = output[order_mapping[3]];

                // FIXME: use integer method to speed up
                float b_diff = 1.77 * (cb - 128);
                float g_diff = -0.343 * (cb - 128) - 0.714 * (cr - 128);
                float r_diff = 1.403 * (cr - 128);

                bmp_buf[bmp_pos++] = clamp_to_0_255(y0 + b_diff);
                bmp_buf[bmp_pos++] = clamp_to_0_255(y0 + g_diff);
                bmp_buf[bmp_pos++] = clamp_to_0_255(y0 + r_diff);
                bmp_buf[bmp_pos++] = clamp_to_0_255(y1 + b_diff);
                bmp_buf[bmp_pos++] = clamp_to_0_255(y1 + g_diff);
                bmp_buf[bmp_pos++] = clamp_to_0_255(y1 + r_diff);
            }
        }
        break;
    }
    case KP_IMAGE_FORMAT_RAW8:
    {
        // Just an alias for buffer naming consistency
        unsigned char *raw8_buf = bin_buf;

        int raw8_pos = 0;

        for (int row = height - 1; row >= 0; row--)
        {
            int bmp_pos = (width * 3 + bmp_padding_byte_num) * row; // we only support bmp file with 3 bytes per pixel
            for (int col = 0; col < width; col++)
            {
                memset(bmp_buf + bmp_pos, raw8_buf[raw8_pos++], 3);
                bmp_pos += 3;
            }
        }
        break;
    }
    case KP_IMAGE_FORMAT_YUV420:
    {
        if ((0 != width % 2) || (0 != height % 2)) {
            printf("Error! width or height is not even number\n");
            return;
        }

        // Just an alias for buffer naming consistency
        unsigned char *yuv_buf = bin_buf;
        int y_bias = width * height;
        int u_bias = y_bias / 4;
        int u_pos = y_bias;
        int v_pos = y_bias + u_bias;

        for (int row = height - 1; row >= 0; row -= 2)
        {
            int bmp_pos_row_0 = (width * 3 + bmp_padding_byte_num) * row; // we only support bmp file with 3 bytes per pixel
            int bmp_pos_row_1 = (width * 3 + bmp_padding_byte_num) * (row - 1); // we only support bmp file with 3 bytes per pixel
            int y_pos_row_0 = width * (height - row - 1);
            int y_pos_row_1 = width * (height - row);

            for (int col = 0; col < width; col += 2)
            {
                unsigned char y[2][2];
                unsigned char u, v;

                y[0][0] = yuv_buf[y_pos_row_0++];
                y[0][1] = yuv_buf[y_pos_row_0++];
                y[1][0] = yuv_buf[y_pos_row_1++];
                y[1][1] = yuv_buf[y_pos_row_1++];

                u = yuv_buf[u_pos++];
                v = yuv_buf[v_pos++];

                // FIXME: use integer method to speed up
                float b_diff = 1.77 * (u - 128);
                float g_diff = -0.343 * (u - 128) - 0.714 * (v - 128);
                float r_diff = 1.403 * (v - 128);

                bmp_buf[bmp_pos_row_0++] = clamp_to_0_255(y[0][0] + b_diff);
                bmp_buf[bmp_pos_row_0++] = clamp_to_0_255(y[0][0] + g_diff);
                bmp_buf[bmp_pos_row_0++] = clamp_to_0_255(y[0][0] + r_diff);
                bmp_buf[bmp_pos_row_0++] = clamp_to_0_255(y[0][1] + b_diff);
                bmp_buf[bmp_pos_row_0++] = clamp_to_0_255(y[0][1] + g_diff);
                bmp_buf[bmp_pos_row_0++] = clamp_to_0_255(y[0][1] + r_diff);

                bmp_buf[bmp_pos_row_1++] = clamp_to_0_255(y[1][0] + b_diff);
                bmp_buf[bmp_pos_row_1++] = clamp_to_0_255(y[1][0] + g_diff);
                bmp_buf[bmp_pos_row_1++] = clamp_to_0_255(y[1][0] + r_diff);
                bmp_buf[bmp_pos_row_1++] = clamp_to_0_255(y[1][1] + b_diff);
                bmp_buf[bmp_pos_row_1++] = clamp_to_0_255(y[1][1] + g_diff);
                bmp_buf[bmp_pos_row_1++] = clamp_to_0_255(y[1][1] + r_diff);
            }
        }
        break;
    }
    default:
    case KP_IMAGE_FORMAT_UNKNOWN:
        printf("image format is not supported\n");
        return;
    }
}

void helper_draw_box_on_bmp_from_bin(const char *in_bin_path, int in_bin_width, int in_bin_height, kp_image_format_t in_bin_format,
                                     const char *out_bmp_path, kp_bounding_box_t boxes[], int box_count)
{

    // Open .bmp file to write
    FILE *out_bmp_file = fopen(out_bmp_path, "wb");
    if (!out_bmp_file)
    {
        printf("file read failed\n");
        return;
    }

    unsigned char *bin_buf = NULL;
    unsigned char *bmp_buf = NULL;

    // Prepare BMP file header
    FILEHEADER header1 = {0};
    INFOHEADER header2 = {0};

    unsigned int bytes_per_pixel_bmp = 3; // This is constant (also for RAW8) for all supported raw formats
    int bmp_buf_size = 0;
    int padding_byte_num = 0;

    bin_buf = (unsigned char *)helper_bin_file_to_raw_buffer(in_bin_path, in_bin_width, in_bin_height, in_bin_format);
    if( NULL == bin_buf )
    {
        printf("Error! read file to buffer failed\n");
        goto err;
    }

    header1.type = 0x4D42; // 'B' and 'M' in ASCII code (little endian)
    header1.size = sizeof(FILEHEADER) + sizeof(INFOHEADER) + ((unsigned int)in_bin_width * (unsigned int)in_bin_height * bytes_per_pixel_bmp);
    header1.offset = sizeof(FILEHEADER) + sizeof(INFOHEADER);

    header2.size = sizeof(INFOHEADER);
    header2.width = in_bin_width;
    header2.height = in_bin_height;
    header2.planes = 1;
    header2.bits = bytes_per_pixel_bmp * 8;
    header2.imagesize = in_bin_width * in_bin_height * bytes_per_pixel_bmp;

    // length of bmp image width in byte should be aligned with 4
    if (in_bin_width * bytes_per_pixel_bmp % 4 != 0)
        padding_byte_num = 4 - in_bin_width * bytes_per_pixel_bmp % 4;

    // bmp_buf is used to store pixel data of bmp image
    bmp_buf_size = (in_bin_width * bytes_per_pixel_bmp + padding_byte_num) * in_bin_height; // image width of bmp file should be a multiple of 4
    bmp_buf = (unsigned char *)malloc(bmp_buf_size);
    if(NULL == bmp_buf ) {
        printf("Error! malloc for image buffer failed\n");
        goto err;
    }

    convert_bin_to_bmp_pixel_data(bin_buf, bmp_buf, in_bin_width, in_bin_height, in_bin_format, padding_byte_num);

    fwrite(&header1, 1, sizeof(header1), out_bmp_file);
    fwrite(&header2, 1, sizeof(header2), out_bmp_file);

    draw_box_on_bmp_raw_buffer(boxes, box_count, in_bin_width, in_bin_height, bmp_buf, padding_byte_num);

    fwrite(bmp_buf, 1, bmp_buf_size, out_bmp_file);

    fclose(out_bmp_file);
    free(bin_buf);
    free(bmp_buf);
	return;

err:
    fclose(out_bmp_file);

    free(bin_buf);
    free(bmp_buf);

    return;
}

void helper_draw_box_of_crop_area_on_bmp(const char *in_bmp_path, const char *out_bmp_path, kp_bounding_box_t boxes[], int box_count, kp_inf_crop_box_t crop_box)
{
    FILEHEADER header1;
    INFOHEADER header2;
    unsigned char *extended_info_header = NULL;
    unsigned char *bmp_buf = NULL;
    unsigned char *crop_bmp_buf = NULL;

    FILE *in_bmp_file = fopen(in_bmp_path, "rb");
    if (!in_bmp_file)
    {
        printf("file read failed\n");
        return;
    }

    size_t extended_info_header_length = 0;
    unsigned short bytes_per_pixel = 0;
    int bmp_buf_size = 0;
    int padding_byte_num_origin = 0;
    int padding_byte_num_crop = 0;
    int crop_bmp_buf_size = 0;

    FILE *out_bmp_file;

    size_t read_size = custom_fread(&header1, 1, sizeof(header1), in_bmp_file);

    if (0 == read_size) {
        printf("Error! %s(): Read File Header failed\n", __FUNCTION__);
        goto err;
    }

    read_size = custom_fread(&header2, 1, sizeof(header2), in_bmp_file);

    if (0 == read_size) {
        printf("Error! %s(): Read Info Header failed\n", __FUNCTION__);
        goto err;
    }

    // read extended info header if the length of info header exceeds sizeof(INFOHEADER)
    if (header1.offset != sizeof(FILEHEADER) + sizeof(INFOHEADER))
    {
        extended_info_header_length = header1.offset - (sizeof(FILEHEADER) + sizeof(INFOHEADER));
        extended_info_header = (unsigned char *)malloc(extended_info_header_length);
        if (NULL == extended_info_header ) {
            printf("Error! %s(): malloc for extended info header failed\n", __FUNCTION__);
            goto err;
        }
        read_size = custom_fread(extended_info_header, 1, extended_info_header_length, in_bmp_file);
        if (0 == read_size) {
            printf("Error! %s(): read extended info header failed\n", __FUNCTION__);
            goto err;
        }
    }

    // Init buffer for origin image
    bytes_per_pixel = header2.bits / 8;

    // length of bmp image width in byte should be aligned with 4
    if (header2.width * bytes_per_pixel % 4 != 0)
        padding_byte_num_origin = 4 - header2.width * bytes_per_pixel % 4;

    // bmp_buf is used to store pixel data of bmp image
    bmp_buf_size = (header2.width * bytes_per_pixel + padding_byte_num_origin) * header2.height; // image width of bmp file should be a multiple of 4
    bmp_buf = (unsigned char *)malloc(bmp_buf_size);
    if(NULL == bmp_buf){
        printf("Error! malloc for image buffer failed\n");
        goto err;
    }

    // Init buffer for crop area
    // length of bmp image width in byte should be aligned with 4
    if (crop_box.width * bytes_per_pixel % 4 != 0)
        padding_byte_num_crop = 4 - crop_box.width * bytes_per_pixel % 4;

    crop_bmp_buf_size = (crop_box.width * bytes_per_pixel + padding_byte_num_crop) * crop_box.height; // image width of bmp file should be a multiple of 4
    crop_bmp_buf = (unsigned char *)malloc(crop_bmp_buf_size);
    if(NULL == crop_bmp_buf){
        printf("Error! malloc for cropped area buffer failed\n");
        goto err;
    }

    // Read whole image raw data
    read_size = custom_fread(bmp_buf, 1, bmp_buf_size, in_bmp_file);

    if (0 == read_size) {
        printf("Error! %s(): read bmp failed\n", __FUNCTION__);
        goto err;
    }

    // Find out the raw data in crop area, copy data row by row. Note that image width (in byte) of bmp file should be a multiple of 4
    // otherwise padding byte(s) is needed (not necessarily 0 (?) but use 0 here)
    for (int row = 0; row < crop_box.height; row++)
    {
        // we only support bmp file with 3 bytes per pixel
        memcpy(crop_bmp_buf + row * (crop_box.width * 3 + padding_byte_num_crop),
               bmp_buf + crop_box.x1 * 3 + (header2.height - crop_box.y1 - crop_box.height + row) * (header2.width * 3 + padding_byte_num_origin),
               crop_box.width * 3 * sizeof(unsigned char));

        // set padding byte(s) to 0
        memset(crop_bmp_buf + row * (crop_box.width * 3 + padding_byte_num_crop) + crop_box.width * 3, 0, padding_byte_num_crop);
    }

    free(bmp_buf);
    bmp_buf = NULL;

    if (header2.bits == 24)
    {
        draw_box_on_bmp_raw_buffer(boxes, box_count, crop_box.width, crop_box.height, crop_bmp_buf, padding_byte_num_crop);
    }
    else
    {
        printf("support only 24 bit bmp\n");
    }

    out_bmp_file = fopen(out_bmp_path, "wb");

    if (!out_bmp_file)
    {
        printf("open output file failed\n");
        goto err;
    }

    header1.size = crop_bmp_buf_size + sizeof(FILEHEADER) + sizeof(INFOHEADER);
    header2.width = crop_box.width;
    header2.height = crop_box.height;
    header2.imagesize = 0;

    fwrite(&header1, 1, sizeof(header1), out_bmp_file);
    fwrite(&header2, 1, sizeof(header2), out_bmp_file);

    // write extended info header
    if (extended_info_header)
    {
        fwrite(extended_info_header, 1, extended_info_header_length, out_bmp_file);
        free(extended_info_header);
    }

    fwrite(crop_bmp_buf, 1, crop_bmp_buf_size, out_bmp_file);
    fclose(in_bmp_file);
    fclose(out_bmp_file);
    free(crop_bmp_buf);

    return;

err:
    fclose(in_bmp_file);

    if (NULL != bmp_buf)
        free(bmp_buf);

    if (NULL != crop_bmp_buf)
        free(crop_bmp_buf);

    if (NULL != extended_info_header)
        free(extended_info_header);

    return;
}

void helper_draw_box_of_crop_area_on_bmp_from_bin(const char *in_bin_path, int in_bin_width, int in_bin_height, kp_image_format_t in_bin_format,
                                                  const char *out_bmp_path, kp_bounding_box_t boxes[], int box_count, kp_inf_crop_box_t crop_box)
{
    unsigned char *bmp_buf = NULL;
    unsigned char *crop_bmp_buf = NULL;
    unsigned char *bin_buf = NULL;

    // Open .bmp file to write
    FILE *out_bmp_file = NULL;
    out_bmp_file = fopen(out_bmp_path, "wb");
    if (!out_bmp_file)
    {
        printf("file read failed\n");
        return;
    }

    unsigned short bytes_per_pixel = 3;
    int padding_byte_num_origin = 0;
    int bmp_buf_size = 0;
    int padding_byte_num_crop = 0;
    int crop_bmp_buf_size = 0;

    FILEHEADER header1 = {0};
    INFOHEADER header2 = {0};
    unsigned short bytes_per_pixel_bmp = 3; // This is constant (also for RAW8) for all supported raw formats

    bin_buf = (unsigned char *)helper_bin_file_to_raw_buffer(in_bin_path, in_bin_width, in_bin_height, in_bin_format);
    if(NULL == bin_buf) {
        printf("error! read file to buffer failed\n");
        goto err;
    }

    // Create corresponding pixel data (in bmp format) of raw image

    // Init buffer for origin image
    // length of bmp image width in byte should be aligned with 4
    if (in_bin_width * bytes_per_pixel % 4 != 0)
        padding_byte_num_origin = 4 - in_bin_width * bytes_per_pixel % 4;

    // bmp_buf is used to store pixel data of bmp image
    bmp_buf_size = (in_bin_width * bytes_per_pixel + padding_byte_num_origin) * in_bin_height; // image width of bmp file should be a multiple of 4
    bmp_buf = (unsigned char *)malloc(bmp_buf_size);
    if(NULL == bmp_buf) {
        printf("error! malloc %s failed\n", "bmp_buf");
        goto err;
    }

    convert_bin_to_bmp_pixel_data(bin_buf, bmp_buf, in_bin_width, in_bin_height, in_bin_format, padding_byte_num_origin);

    // Init buffer for crop area
    // length of bmp image width in byte should be aligned with 4
    if (crop_box.width * bytes_per_pixel % 4 != 0)
        padding_byte_num_crop = 4 - crop_box.width * bytes_per_pixel % 4;

    crop_bmp_buf_size = (crop_box.width * bytes_per_pixel + padding_byte_num_crop) * crop_box.height; // image width of bmp file should be a multiple of 4
    crop_bmp_buf = (unsigned char *)malloc(crop_bmp_buf_size);
    if (NULL == crop_bmp_buf) {
        printf("error! malloc %s failed\n", "crop_bmp_buf");
        goto err;
    }

    // Find out the raw data in crop area, copy data row by row. Note that image width (in byte) of bmp file should be a multiple of 4
    // otherwise padding byte(s) is needed (not necessarily 0 (?) but use 0 here)
    for (int row = 0; row < crop_box.height; row++)
    {
        // we only support bmp file with 3 bytes per pixel
        memcpy(crop_bmp_buf + row * (crop_box.width * 3 + padding_byte_num_crop),
               bmp_buf + crop_box.x1 * 3 + (in_bin_height - crop_box.y1 - crop_box.height + row) * (in_bin_width * 3 + padding_byte_num_origin),
               crop_box.width * 3 * sizeof(unsigned char));

        // set padding byte(s) to 0
        memset(crop_bmp_buf + row * (crop_box.width * 3 + padding_byte_num_crop) + crop_box.width * 3, 0, padding_byte_num_crop);
    }

    free(bmp_buf);
    bmp_buf=NULL;

    draw_box_on_bmp_raw_buffer(boxes, box_count, crop_box.width, crop_box.height, crop_bmp_buf, padding_byte_num_crop);

    // Prepare BMP file header
    header1.type = 0x4D42; // 'B' and 'M' in ASCII code (little endian)
    header1.size = sizeof(FILEHEADER) + sizeof(INFOHEADER) + crop_box.width * crop_box.height * bytes_per_pixel_bmp;
    header1.offset = sizeof(FILEHEADER) + sizeof(INFOHEADER);

    header2.size = sizeof(INFOHEADER);
    header2.width = crop_box.width;
    header2.height = crop_box.height;
    header2.planes = 1;
    header2.bits = bytes_per_pixel_bmp * 8;
    header2.imagesize = crop_box.width * crop_box.height * bytes_per_pixel_bmp;

    fwrite(&header1, 1, sizeof(header1), out_bmp_file);
    fwrite(&header2, 1, sizeof(header2), out_bmp_file);

    fwrite(crop_bmp_buf, 1, crop_bmp_buf_size, out_bmp_file);

    fclose(out_bmp_file);
    free(bin_buf);
    free(crop_bmp_buf);

    return;

err:
    fclose(out_bmp_file);

    free(bin_buf);
    free(bmp_buf);
    free(crop_bmp_buf);

    return;
}

void helper_print_yolo_box(kp_yolo_result_t *yolo_result)
{
    printf("\n");
    printf("detectable class count : %d\n", yolo_result->class_count);
    printf("box count : %d\n", yolo_result->box_count);
    for (int i = 0; i < yolo_result->box_count; i++)
    {
        printf("Box %d (x1, y1, x2, y2, score, class) = %.1f, %.1f, %.1f, %.1f, %f, %d\n",
               i,
               yolo_result->boxes[i].x1, yolo_result->boxes[i].y1,
               yolo_result->boxes[i].x2, yolo_result->boxes[i].y2,
               yolo_result->boxes[i].score, yolo_result->boxes[i].class_num);
    }
}

void helper_print_yolo_box_on_bmp(kp_yolo_result_t *yolo_result, char *in_bmp_path)
{
    // output bounding boxes to a new created bmp file
    char out_bmp_path[128];
    sprintf(out_bmp_path, "output_%s", helper_file_name_from_path(in_bmp_path));
    helper_draw_box_on_bmp(in_bmp_path, out_bmp_path, yolo_result->boxes, yolo_result->box_count);

    helper_print_yolo_box(yolo_result);

    if (yolo_result->box_count > 0)
        printf("\noutput bounding boxes on '%s'\n", out_bmp_path);
    else
        printf("\noutput result on '%s'\n", out_bmp_path);
}

void helper_print_yolo_box_on_bmp_from_bin(kp_yolo_result_t *yolo_result, char *in_bin_path, int in_bin_width, int in_bin_height,
                                           kp_image_format_t in_bin_format)
{
    // output bounding boxes to a new created bmp file from bin file with raw format
    char out_bmp_path[128] = {0};

    // 128 - 11 = 117 -> see sprintf below, output_.bmp takes up 11 bytes,
    // so the length of this array should be length of out_bmp_path - 11 to prevent overflow
    char file_name_without_ext[117] = {0};
    char *file_name = helper_file_name_from_path(in_bin_path);

    strncpy(file_name_without_ext, file_name, 116);

    int len = strlen(file_name_without_ext);
    if (len <= 4)  {
        printf("ERROR! invalid file name: %s\n", file_name_without_ext);
        return;
    }

    // .bin takes 4 chars
    file_name_without_ext[len - 4] = '\0';

    sprintf(out_bmp_path, "output_%s.bmp", file_name_without_ext);

    helper_draw_box_on_bmp_from_bin(in_bin_path, in_bin_width, in_bin_height, in_bin_format, out_bmp_path, yolo_result->boxes, yolo_result->box_count);

    helper_print_yolo_box(yolo_result);

    if (yolo_result->box_count > 0)
        printf("\noutput bounding boxes on '%s'\n", out_bmp_path);
    else
        printf("\noutput result on '%s'\n", out_bmp_path);
}

void helper_print_yolo_box_of_crop_area(kp_yolo_result_t *yolo_result, kp_inf_crop_box_t crop_box)
{
    printf("\n");
    printf("crop box width : %d\n", crop_box.width);
    printf("crop box height : %d\n", crop_box.height);
    printf("crop box number : %d\n", crop_box.crop_number);
    printf("detectable class count : %d\n", yolo_result->class_count);
    printf("box count : %d\n", yolo_result->box_count);
    for (int i = 0; i < yolo_result->box_count; i++)
    {
        printf("Box %d (x1, y1, x2, y2, score, class) = %.1f, %.1f, %.1f, %.1f, %f, %d\n",
               i,
               yolo_result->boxes[i].x1, yolo_result->boxes[i].y1,
               yolo_result->boxes[i].x2, yolo_result->boxes[i].y2,
               yolo_result->boxes[i].score, yolo_result->boxes[i].class_num);
    }
}

void helper_print_yolo_box_of_crop_area_on_bmp(kp_yolo_result_t *yolo_result, char *in_bmp_path, kp_inf_crop_box_t crop_box)
{
    // output bounding boxes to a new created bmp file
    char out_bmp_path[128] = {0};

    // 128 - 17 = 111 -> see sprintf below, output__crop%d.bmp takes up 17 bytes (crop num max is 4),
    // so the length of this array should be length of out_bmp_path - 17 to prevent overflow
    char in_bmp_filename_without_ext[111] = {0};
    char *in_bmp_filename = helper_file_name_from_path(in_bmp_path);

    strncpy(in_bmp_filename_without_ext, in_bmp_filename, 110);

    int len = strlen(in_bmp_filename_without_ext);
    if (len <= 4)  {
        printf("ERROR! invalid file name: %s\n", in_bmp_filename_without_ext);
        return;
    }

    // .bmp or .bin takes 4 chars
    in_bmp_filename_without_ext[len - 4] = '\0';

    sprintf(out_bmp_path, "output_%s_crop%d.bmp", in_bmp_filename_without_ext, crop_box.crop_number);

    helper_draw_box_of_crop_area_on_bmp(in_bmp_path, out_bmp_path, yolo_result->boxes, yolo_result->box_count, crop_box);

    helper_print_yolo_box_of_crop_area(yolo_result, crop_box);

    if (yolo_result->box_count > 0)
        printf("\noutput bounding boxes on '%s'\n", out_bmp_path);
    else
        printf("\noutput result on '%s'\n", out_bmp_path);
}

void helper_print_yolo_box_of_crop_area_on_bmp_from_bin(kp_yolo_result_t *yolo_result, char *in_bin_path, int in_bin_width, int in_bin_height,
                                                        kp_image_format_t in_bin_format, kp_inf_crop_box_t crop_box)
{
    // output bounding boxes to a new created bmp file
    char out_bmp_path[128] = {0};

    // 128 - 17 = 111 -> see sprintf below, output__crop%d.bmp takes up 17 bytes (crop num max is 4),
    // so the length of this array should be length of out_bmp_path - 17 to prevent overflow
    char in_bin_filename_without_ext[111] = {0};
    char *in_bin_filename = helper_file_name_from_path(in_bin_path);

    strncpy(in_bin_filename_without_ext, in_bin_filename, 110);

    int len = strlen(in_bin_filename_without_ext);
    if (len <= 4)  {
        printf("ERROR! invalid file name: %s\n", in_bin_filename_without_ext);
        return;
    }

    // .bmp or .bin takes 4 chars
    in_bin_filename_without_ext[len - 4] = '\0';

    sprintf(out_bmp_path, "output_%s_crop%d.bmp", in_bin_filename_without_ext, crop_box.crop_number);

    helper_draw_box_of_crop_area_on_bmp_from_bin(in_bin_path, in_bin_width, in_bin_height, in_bin_format, out_bmp_path, yolo_result->boxes, yolo_result->box_count, crop_box);

    helper_print_yolo_box_of_crop_area(yolo_result, crop_box);

    if (yolo_result->box_count > 0)
        printf("\noutput bounding boxes on '%s'\n", out_bmp_path);
    else
        printf("\noutput result on '%s'\n", out_bmp_path);
}

void helper_string_to_number_array(char *number_string, int *number_group, int *num_devs)
{
    *num_devs = 0;
    char *token;

    /* get the first token */
    token = strtok(number_string, ",");

    /* walk through other tokens */
    while (token != NULL)
    {
        number_group[(*num_devs)++] = atoi(token);
        token = strtok(NULL, ",");
    }
}

char *helper_file_name_from_path(char *path)
{
    if (path == NULL)
        return NULL;

    char *pFileName = path;
    for (char *pCur = path; *pCur != '\0'; pCur++)
    {
        if (*pCur == '/' || *pCur == '\\')
            pFileName = pCur + 1;
    }

    return pFileName;
}

int helper_string_to_crop_box_array(char *crop_box_array, uint32_t *crop_count, kp_inf_crop_box_t *crop_boxes)
{
    char *token;
    uint32_t count = 0U;

    /* get the first token */
    token = strtok(crop_box_array, "(,) ");

    /* walk through other tokens */
    while (token != NULL)
    {
        // 4 means there are 4 numbers (stands for x1, y1, width, height)
        switch (count % 4)
        {
        case 0:
            crop_boxes[count / 4].crop_number = count / 4;
            crop_boxes[count / 4].x1 = (uint32_t)atoi(token);
            break;
        case 1:
            crop_boxes[count / 4].y1 = (uint32_t)atoi(token);
            break;
        case 2:
            crop_boxes[count / 4].width = (uint32_t)atoi(token);
            break;
        case 3:
            crop_boxes[count / 4].height = (uint32_t)atoi(token);
            break;
        }

        token = strtok(NULL, "(,) ");
        count++;
    }

    if (count % 4 != 0)
        return -1;

    *crop_count = count / 4;
    return 0;
}

char *helper_kp_fixed_point_dtype_to_string(uint32_t fixed_point_dtype)
{
    switch (fixed_point_dtype)
    {
    case KP_FIXED_POINT_DTYPE_INT8:
        return "INT8";
    case KP_FIXED_POINT_DTYPE_INT16:
        return "INT16";
    default:
        return "UNKNOWN";
    }
}

kp_inf_float_node_output_t* helper_fixed_to_floating_node_data(kp_inf_fixed_node_output_t *fixed_node_output)
{
    kp_inf_float_node_output_t *floating_output_nodes = (kp_inf_float_node_output_t *)calloc(1, sizeof(kp_inf_float_node_output_t) + fixed_node_output->num_data * sizeof(float));

    if (NULL == floating_output_nodes) {
        printf("memory is insufficient to allocate buffer for node output\n");
        return NULL;
    }

    floating_output_nodes->num_data     = fixed_node_output->num_data;
    floating_output_nodes->shape_len    = fixed_node_output->shape_len;
    floating_output_nodes->shape        = (int32_t *)calloc(floating_output_nodes->shape_len, sizeof(int32_t));

    if (NULL == floating_output_nodes->shape) {
        printf("memory is insufficient to allocate buffer for node output\n");
        kp_release_float_node_output(floating_output_nodes);
        return NULL;
    }

    memcpy(floating_output_nodes->shape, fixed_node_output->shape, fixed_node_output->shape_len * sizeof(int32_t));

    if (1 == fixed_node_output->quantization_parameters.quantization_parameters_data.v1.quantized_fixed_point_descriptor_num) {
        float ffactor = fixed_node_output->quantization_parameters.quantization_parameters_data.v1.quantized_fixed_point_descriptor[0].scale.scale_float32 *
                        pow2(fixed_node_output->quantization_parameters.quantization_parameters_data.v1.quantized_fixed_point_descriptor[0].radix);

        if (KP_FIXED_POINT_DTYPE_INT8 == fixed_node_output->fixed_point_dtype) {
            for (int i = 0; i < fixed_node_output->num_data; i++) {
                floating_output_nodes->data[i] = (float)fixed_node_output->data.int8[i] / ffactor;
            }
        } else if (KP_FIXED_POINT_DTYPE_INT16 == fixed_node_output->fixed_point_dtype) {
            for (int i = 0; i < fixed_node_output->num_data; i++) {
                floating_output_nodes->data[i] = (float)fixed_node_output->data.int16[i] / ffactor;
            }
        } else {
            printf("unknown fixed point data type %d ...\n", fixed_node_output->fixed_point_dtype);
            kp_release_float_node_output(floating_output_nodes);
            return NULL;
        }
    } else {
        if (fixed_node_output->num_data != fixed_node_output->quantization_parameters.quantization_parameters_data.v1.quantized_fixed_point_descriptor_num) {
            printf("quantization info len (%d) does not match data len (%d) ...\n", fixed_node_output->quantization_parameters.quantization_parameters_data.v1.quantized_fixed_point_descriptor_num,
                                                                                    fixed_node_output->num_data);
            kp_release_float_node_output(floating_output_nodes);
            return NULL;
        }

        float ffactor = 1;

        if (KP_FIXED_POINT_DTYPE_INT8 == fixed_node_output->fixed_point_dtype) {
            for (int i = 0; i < fixed_node_output->num_data; i++) {
                ffactor = fixed_node_output->quantization_parameters.quantization_parameters_data.v1.quantized_fixed_point_descriptor[i].scale.scale_float32 *
                          pow2(fixed_node_output->quantization_parameters.quantization_parameters_data.v1.quantized_fixed_point_descriptor[i].radix);
                floating_output_nodes->data[i] = (float)fixed_node_output->data.int8[i] / ffactor;
            }
        } else if (KP_FIXED_POINT_DTYPE_INT16 == fixed_node_output->fixed_point_dtype) {
            for (int i = 0; i < fixed_node_output->num_data; i++) {
                ffactor = fixed_node_output->quantization_parameters.quantization_parameters_data.v1.quantized_fixed_point_descriptor[i].scale.scale_float32 *
                          pow2(fixed_node_output->quantization_parameters.quantization_parameters_data.v1.quantized_fixed_point_descriptor[i].radix);
                floating_output_nodes->data[i] = (float)fixed_node_output->data.int16[i] / ffactor;
            }
        } else {
            printf("unknown fixed point data type %d ...\n", fixed_node_output->fixed_point_dtype);
            kp_release_float_node_output(floating_output_nodes);
            return NULL;
        }
    }

    return floating_output_nodes;
}

// crop_num < 0 means no cropping
static void print_floating_node_data(kp_inf_float_node_output_t *output_nodes[], int num_nodes, int crop_num)
{
    printf("\n");

    if (crop_num >= 0)
        printf("crop box: %d\n", crop_num);

    printf("number of output node : %d\n", num_nodes);

    for (int i = 0; i < num_nodes; i++)
    {
        printf("\n");
        printf("node %d:\n", i);
        for (uint32_t shape_idx = 0; shape_idx < output_nodes[i]->shape_len; shape_idx++) {
            printf("shape [%u]: %d:\n", shape_idx, output_nodes[i]->shape[shape_idx]);
        }
        printf("number of data (float): %d:\n", output_nodes[i]->num_data);

        printf("first 20 data:\n\t");
        for (int j = 0; j < fmin(20, output_nodes[i]->num_data); j++)
        {
            printf("%.3f, ", output_nodes[i]->data[j]);
            if (j > 0 && j % 5 == 0)
                printf("\n\t");
        }

        printf("\n");
    }

    printf("\n");
}

// crop_num < 0 means no cropping
static void dump_floating_node_data_to_files(kp_inf_float_node_output_t *output_nodes[], int num_nodes, int crop_num, char *in_bmp_file)
{
    char in_bmp_filename_without_ext[128] = {0};
    char *in_bmp_filename = helper_file_name_from_path(in_bmp_file);

    strncpy(in_bmp_filename_without_ext, in_bmp_filename, 127);

    int len = strlen(in_bmp_filename_without_ext);
    if (len <= 4)  {
        printf("ERROR! invalid file name: %s\n", in_bmp_filename_without_ext);
        return;
    }

    // .bmp or .bin takes 4 chars
    in_bmp_filename_without_ext[len - 4] = '\0';

    for (int i = 0; i < num_nodes; i++)
    {
        char file_name[512] = {0};
        char shape_str[256] = {0};
        char shape_str_tmp[256] = {0};

        for (int axis = 0; axis < output_nodes[i]->shape_len; axis++) {
            if (0 == axis)
                sprintf(shape_str, "%d", output_nodes[i]->shape[axis]);
            else {
                memset(shape_str_tmp, 0, sizeof(shape_str_tmp));
                sprintf(shape_str_tmp, "x%d", output_nodes[i]->shape[axis]);
                strcat(shape_str, shape_str_tmp);
            }
        }

        if (crop_num >= 0) {
            sprintf(file_name, "output_%s_crop%d_node%d_%s.txt", in_bmp_filename_without_ext, crop_num, i, shape_str);
        } else {
            sprintf(file_name, "output_%s_node%d_%s.txt", in_bmp_filename_without_ext, i, shape_str);
        }

        /* FIXME : Support KL730*/

        FILE *file = fopen(file_name, "w");
        if (file)
        {
            for (int j = 0; j < output_nodes[i]->num_data; j++)
            {
                fprintf(file, "%.3f", output_nodes[i]->data[j]);
                if (j != output_nodes[i]->num_data - 1)
                    fprintf(file, ", ");
            }

            fclose(file);
            printf("dumped node %d output to '%s'\n", i, file_name);
        }
    }
}

// crop_num < 0 means no cropping
static void print_fixed_node_data(kp_inf_fixed_node_output_t *output_nodes[], int num_nodes, int crop_num)
{
    printf("\n");

    if (crop_num >= 0)
        printf("crop box: %d\n", crop_num);

    printf("number of output node : %d\n", num_nodes);

    for (int i = 0; i < num_nodes; i++)
    {
        printf("\n");
        printf("node %d:\n", i);
        for (uint32_t shape_idx = 0; shape_idx < output_nodes[i]->shape_len; shape_idx++) {
            printf("shape [%u]: %d:\n", shape_idx, output_nodes[i]->shape[shape_idx]);
        }

        for (uint32_t quant_idx = 0; quant_idx < output_nodes[i]->quantization_parameters.quantization_parameters_data.v1.quantized_fixed_point_descriptor_num; quant_idx++) {
            printf("quantization idx [%u] ", quant_idx);
            helper_print_kp_quantized_fixed_point_descriptor(&output_nodes[i]->quantization_parameters.quantization_parameters_data.v1.quantized_fixed_point_descriptor[quant_idx]);
        }

        printf("fixed point dtype: %s:\n", helper_kp_fixed_point_dtype_to_string(output_nodes[i]->fixed_point_dtype));
        printf("number of data (fixed): %d:\n", output_nodes[i]->num_data);

        printf("first 20 data:\n\t");
        if (KP_FIXED_POINT_DTYPE_INT8 == output_nodes[i]->fixed_point_dtype) {
            for (int j = 0; j < fmin(20, output_nodes[i]->num_data); j++)
            {
                printf("%d, ", output_nodes[i]->data.int8[j]);
                if (j > 0 && j % 5 == 0)
                    printf("\n\t");
            }
        } else if (KP_FIXED_POINT_DTYPE_INT16 == output_nodes[i]->fixed_point_dtype) {
            for (int j = 0; j < fmin(20, output_nodes[i]->num_data); j++)
            {
                printf("%d, ", output_nodes[i]->data.int16[j]);
                if (j > 0 && j % 5 == 0)
                    printf("\n\t");
            }
        } else {
            printf("unknown fixed point data type %d ...\n", output_nodes[i]->fixed_point_dtype);
        }

        printf("\n");
    }

    printf("\n");
}

// crop_num < 0 means no cropping
static void dump_fixed_node_data_to_files(kp_inf_fixed_node_output_t *output_nodes[], int num_nodes, int crop_num, char *in_bmp_file)
{
    char in_bmp_filename_without_ext[128] = {0};
    char *in_bmp_filename = helper_file_name_from_path(in_bmp_file);

    strncpy(in_bmp_filename_without_ext, in_bmp_filename, 127);

    int len = strlen(in_bmp_filename_without_ext);
    if (len <= 4)  {
        printf("ERROR! invalid file name: %s\n", in_bmp_filename_without_ext);
        return;
    }

    // .bmp or .bin takes 4 chars
    in_bmp_filename_without_ext[len - 4] = '\0';

    for (int i = 0; i < num_nodes; i++)
    {
        char file_name[512] = {0};
        char shape_str[256] = {0};
        char shape_str_tmp[256] = {0};

        for (int axis = 0; axis < output_nodes[i]->shape_len; axis++) {
            if (0 == axis)
                sprintf(shape_str, "%d", output_nodes[i]->shape[axis]);
            else {
                memset(shape_str_tmp, 0, sizeof(shape_str_tmp));
                sprintf(shape_str_tmp, "x%d", output_nodes[i]->shape[axis]);
                strcat(shape_str, shape_str_tmp);
            }
        }

        if (crop_num >= 0) {
            sprintf(file_name, "output_%s_crop%d_node%d_%s.txt", in_bmp_filename_without_ext, crop_num, i, shape_str);
        } else {
            sprintf(file_name, "output_%s_node%d_%s.txt", in_bmp_filename_without_ext, i, shape_str);
        }

        /* FIXME : Support KL730 */

        FILE *file = fopen(file_name, "w");
        if (file)
        {
            if (KP_FIXED_POINT_DTYPE_INT8 == output_nodes[i]->fixed_point_dtype) {
                for (int j = 0; j < output_nodes[i]->num_data; j++)
                {
                    fprintf(file, "%d", output_nodes[i]->data.int8[j]);
                    if (j != output_nodes[i]->num_data - 1)
                        fprintf(file, ", ");
                }
            } else if (KP_FIXED_POINT_DTYPE_INT16 == output_nodes[i]->fixed_point_dtype) {
                for (int j = 0; j < output_nodes[i]->num_data; j++)
                {
                    fprintf(file, "%d", output_nodes[i]->data.int16[j]);
                    if (j != output_nodes[i]->num_data - 1)
                        fprintf(file, ", ");
                }
            } else {
                printf("unknown fixed point data type %d ...\n", output_nodes[i]->fixed_point_dtype);
            }

            fclose(file);
            printf("dumped node %d output to '%s'\n", i, file_name);
        }
    }
}

void helper_dump_floating_node_data_to_files(kp_inf_float_node_output_t *output_nodes[], int num_nodes, char *in_img_path)
{
    print_floating_node_data(output_nodes, num_nodes, -1);                      // -1 means no cropping
    dump_floating_node_data_to_files(output_nodes, num_nodes, -1, in_img_path); // -1 means no cropping
}

void helper_dump_floating_node_data_of_crop_area_to_files(kp_inf_float_node_output_t *output_nodes[], int num_nodes, int crop_num, char *in_img_path)
{
    print_floating_node_data(output_nodes, num_nodes, crop_num);
    dump_floating_node_data_to_files(output_nodes, num_nodes, crop_num, in_img_path);
}

void helper_dump_fixed_node_data_to_files(kp_inf_fixed_node_output_t *output_nodes[], int num_nodes, char *in_img_path)
{
    print_fixed_node_data(output_nodes, num_nodes, -1);                      // -1 means no cropping
    dump_fixed_node_data_to_files(output_nodes, num_nodes, -1, in_img_path); // -1 means no cropping
}

void helper_dump_fixed_node_data_of_crop_area_to_files(kp_inf_fixed_node_output_t *output_nodes[], int num_nodes, int crop_num, char *in_img_path)
{
    print_fixed_node_data(output_nodes, num_nodes, crop_num);
    dump_fixed_node_data_to_files(output_nodes, num_nodes, crop_num, in_img_path);
}

char *helper_kp_model_tensor_data_layout_to_string(uint32_t tensor_data_layout)
{
    switch (tensor_data_layout)
    {
    case KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B:
        return "16W1C8B";
    case KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B:
        return "4W4C8B";
    case KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B:
        return "1W16C8B";
    case KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B:
        return "8W1C16B";
    case KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8BHL:
        return "4W4C8BHL";
    case KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8BHL:
        return "1W16C8BHL";
    case KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8BHL:
        return "16W1C8BHL";
    case KP_MODEL_TENSOR_DATA_LAYOUT_RAW_8B:
        return "RAW_8B";
    case KP_MODEL_TENSOR_DATA_LAYOUT_RAW_16B:
        return "RAW_16B";
    case KP_MODEL_TENSOR_DATA_LAYOUT_RAW_FLOAT:
        return "RAW_FLOAT";
    default:
        return "UNKNOWN";
    }
}

char *helper_kp_model_target_chip_to_string(uint32_t target_chip)
{
    switch (target_chip)
    {
    case KP_MODEL_TARGET_CHIP_KL520:
        return "KL520";
    case KP_MODEL_TARGET_CHIP_KL720:
        return "KL720";
    case KP_MODEL_TARGET_CHIP_KL530:
        return "KL530";
    case KP_MODEL_TARGET_CHIP_KL730:
        return "KL730";
    case KP_MODEL_TARGET_CHIP_KL630:
        return "KL630";
    case KP_MODEL_TARGET_CHIP_KL540:
        return "KL540";
    default:
        return "UNKNOWN";
    }
}

void helper_print_kp_quantized_fixed_point_descriptor(kp_quantized_fixed_point_descriptor_t* quantized_fixed_point_descriptor) {
    if (NULL == quantized_fixed_point_descriptor) {
        printf("print quantized fixed point info in model_descriptor fail: NULL pointer input parameters ...\n");
        return;
    }

    switch (quantized_fixed_point_descriptor->scale_dtype)
    {
    case KP_DTYPE_INT8:
        printf("scale %d radix %d\n", (int)quantized_fixed_point_descriptor->scale.scale_int8, quantized_fixed_point_descriptor->radix);
        break;
    case KP_DTYPE_INT16:
        printf("scale %d radix %d\n", (int)quantized_fixed_point_descriptor->scale.scale_int16, quantized_fixed_point_descriptor->radix);
        break;
    case KP_DTYPE_INT32:
        printf("scale %d radix %d\n", (int)quantized_fixed_point_descriptor->scale.scale_int32, quantized_fixed_point_descriptor->radix);
        break;
    case KP_DTYPE_UINT8:
        printf("scale %u radix %d\n", (unsigned int)quantized_fixed_point_descriptor->scale.scale_uint8, quantized_fixed_point_descriptor->radix);
        break;
    case KP_DTYPE_UINT16:
        printf("scale %u radix %d\n", (unsigned int)quantized_fixed_point_descriptor->scale.scale_uint16, quantized_fixed_point_descriptor->radix);
        break;
    case KP_DTYPE_UINT32:
        printf("scale %u radix %d\n", (unsigned int)quantized_fixed_point_descriptor->scale.scale_uint32, quantized_fixed_point_descriptor->radix);
        break;
    case KP_DTYPE_FLOAT32:
        printf("scale %lf radix %d\n", quantized_fixed_point_descriptor->scale.scale_float32, quantized_fixed_point_descriptor->radix);
        break;
    case KP_DTYPE_BFLOAT16:
        printf("scale %d radix %d\n", (int)quantized_fixed_point_descriptor->scale.scale_bfloat16, quantized_fixed_point_descriptor->radix);
        break;
    case KP_DTYPE_DOUBLE64:
        printf("scale %lf radix %d\n", quantized_fixed_point_descriptor->scale.scale_double64, quantized_fixed_point_descriptor->radix);
        break;
    case KP_DTYPE_INT64:
    case KP_DTYPE_UINT64:
        printf("currently not support printf feature on 64-bits integer.");
        break;
    default:
        printf("print nef single tensor information quantization parameters in fail: invalid KneronKNE_DataType_enum_t\n");
    }
}

void helper_print_model_info(kp_model_nef_descriptor_t *pModel_desc)
{
    printf("\n");
    printf("this NEF contains %d model(s):\n", pModel_desc->num_models);
    for (int i = 0; i < pModel_desc->num_models; i++)
    {
        printf("[%d] model ID = %d\n", i + 1, pModel_desc->models[i].id);
        printf("    model raw output size = %d\n", pModel_desc->models[i].max_raw_out_size);

        printf("    model input:\n");
        for (int node_idx = 0; node_idx < pModel_desc->models[i].input_nodes_num; node_idx++) {
            kp_tensor_descriptor_t *tensor_desc                                             = &(pModel_desc->models[i].input_nodes[node_idx]);
            kp_tensor_shape_info_t* tensor_shape_info                                       = &(tensor_desc->tensor_shape_info);
            kp_tensor_shape_info_v1_t* tensor_shape_info_v1                                 = &(tensor_shape_info->tensor_shape_info_data.v1);
            kp_tensor_shape_info_v2_t* tensor_shape_info_v2                                 = &(tensor_shape_info->tensor_shape_info_data.v2);
            kp_quantization_parameters_t* quantization_parameters                           = &(tensor_desc->quantization_parameters);
            kp_quantization_parameters_v1_t* quantization_parameters_v1                     = NULL;
            kp_quantized_fixed_point_descriptor_t* quantized_fixed_point_descriptor_list    = NULL;

            printf("        [%d] input name = %s\n", node_idx, tensor_desc->name);
            printf("        [%d] input shape info version: %u\n", node_idx, tensor_shape_info->version);
            if (KP_MODEL_TENSOR_SHAPE_INFO_VERSION_1 == tensor_shape_info->version) {
                printf("        [%d] input shape npu: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v1->shape_npu_len; i++) {
                    printf(" %u", tensor_shape_info_v1->shape_npu[i]);

                    if (i < tensor_shape_info_v1->shape_npu_len - 1)
                        printf(",");
                }
                printf("]\n");

                printf("        [%d] input shape onnx: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v1->shape_onnx_len; i++) {
                    printf(" %u", tensor_shape_info_v1->shape_onnx[i]);
                    if (i < tensor_shape_info_v1->shape_onnx_len - 1)
                        printf(",");
                }
                printf("]\n");

                printf("        [%d] input axis_permutation_onnx_to_npu: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v1->axis_permutation_len; i++) {
                    printf(" %u", tensor_shape_info_v1->axis_permutation_onnx_to_npu[i]);
                    if (i < tensor_shape_info_v1->axis_permutation_len - 1)
                        printf(",");
                }
                printf("]\n");
            } else if (KP_MODEL_TENSOR_SHAPE_INFO_VERSION_2 == tensor_shape_info->version) {
                printf("        [%d] input shape: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v2->shape_len; i++) {
                    printf(" %u", tensor_shape_info_v2->shape[i]);

                    if (i < tensor_shape_info_v2->shape_len - 1)
                        printf(",");
                }
                printf("]\n");

                printf("        [%d] input stride_onnx: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v2->shape_len; i++) {
                    printf(" %u", tensor_shape_info_v2->stride_onnx[i]);
                    if (i < tensor_shape_info_v2->shape_len - 1)
                        printf(",");
                }
                printf("]\n");

                printf("        [%d] input stride_npu: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v2->shape_len; i++) {
                    printf(" %u", tensor_shape_info_v2->stride_onnx[i]);
                    if (i < tensor_shape_info_v2->shape_len - 1)
                        printf(",");
                }
                printf("]\n");
            } else {
                printf("        [%d] print nef single tensor information shape info in fail: invalid kp_model_tensor_shape_info_version_t\n", node_idx);
            }

            printf("        [%d] data layout format = %s\n", node_idx, helper_kp_model_tensor_data_layout_to_string(tensor_desc->data_layout));
            printf("        [%d] quantization parameters version = %u\n", node_idx, quantization_parameters->version);

            if (KP_MODEL_QUANTIZATION_PARAMS_VERSION_1 == quantization_parameters->version) {
                quantization_parameters_v1              = &(quantization_parameters->quantization_parameters_data.v1);
                quantized_fixed_point_descriptor_list   = quantization_parameters_v1->quantized_fixed_point_descriptor;

                printf("        [%d] quantization axis = %u\n", node_idx, quantization_parameters_v1->quantized_axis);

                for (int quant_info_idx = 0; quant_info_idx < quantization_parameters_v1->quantized_fixed_point_descriptor_num; quant_info_idx++) {
                    printf("        [%d] quantization parameters [%d] = ", node_idx, quant_info_idx);
                    helper_print_kp_quantized_fixed_point_descriptor(&(quantized_fixed_point_descriptor_list[quant_info_idx]));
                }
            } else {
                printf("        [%d] print nef single input tensor information quantization parameters in fail: invalid kp_model_tensor_shape_info_version_t\n", node_idx);
            }
        }

        printf("    model output:\n");
        for (int node_idx = 0; node_idx < pModel_desc->models[i].output_nodes_num; node_idx++) {
            kp_tensor_descriptor_t *tensor_desc                                             = &(pModel_desc->models[i].output_nodes[node_idx]);
            kp_tensor_shape_info_t* tensor_shape_info                                       = &(tensor_desc->tensor_shape_info);
            kp_tensor_shape_info_v1_t* tensor_shape_info_v1                                 = &(tensor_shape_info->tensor_shape_info_data.v1);
            kp_tensor_shape_info_v2_t* tensor_shape_info_v2                                 = &(tensor_shape_info->tensor_shape_info_data.v2);
            kp_quantization_parameters_t* quantization_parameters                           = &(tensor_desc->quantization_parameters);
            kp_quantization_parameters_v1_t* quantization_parameters_v1                     = NULL;
            kp_quantized_fixed_point_descriptor_t* quantized_fixed_point_descriptor_list    = NULL;

            printf("        [%d] output name = %s\n", node_idx, tensor_desc->name);
            printf("        [%d] output shape info version: %u\n", node_idx, tensor_shape_info->version);
            if (KP_MODEL_TENSOR_SHAPE_INFO_VERSION_1 == tensor_shape_info->version) {
                printf("        [%d] output shape npu: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v1->shape_npu_len; i++) {
                    printf(" %u", tensor_shape_info_v1->shape_npu[i]);

                    if (i < tensor_shape_info_v1->shape_npu_len - 1)
                        printf(",");
                }
                printf("]\n");

                printf("        [%d] output shape onnx: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v1->shape_onnx_len; i++) {
                    printf(" %u", tensor_shape_info_v1->shape_onnx[i]);
                    if (i < tensor_shape_info_v1->shape_onnx_len - 1)
                        printf(",");
                }
                printf("]\n");

                printf("        [%d] output axis_permutation_onnx_to_npu: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v1->axis_permutation_len; i++) {
                    printf(" %u", tensor_shape_info_v1->axis_permutation_onnx_to_npu[i]);
                    if (i < tensor_shape_info_v1->axis_permutation_len - 1)
                        printf(",");
                }
                printf("]\n");
            } else if (KP_MODEL_TENSOR_SHAPE_INFO_VERSION_2 == tensor_shape_info->version) {
                printf("        [%d] output shape: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v2->shape_len; i++) {
                    printf(" %u", tensor_shape_info_v2->shape[i]);

                    if (i < tensor_shape_info_v2->shape_len - 1)
                        printf(",");
                }
                printf("]\n");

                printf("        [%d] output stride_onnx: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v2->shape_len; i++) {
                    printf(" %u", tensor_shape_info_v2->stride_onnx[i]);
                    if (i < tensor_shape_info_v2->shape_len - 1)
                        printf(",");
                }
                printf("]\n");

                printf("        [%d] output stride_npu: [", node_idx);
                for (int i = 0; i < tensor_shape_info_v2->shape_len; i++) {
                    printf(" %u", tensor_shape_info_v2->stride_onnx[i]);
                    if (i < tensor_shape_info_v2->shape_len - 1)
                        printf(",");
                }
                printf("]\n");
            } else {
                printf("        [%d] print nef single tensor information quantization parameters in fail: invalid kp_model_tensor_shape_info_version_t\n", node_idx);
            }


            printf("        [%d] data layout format = %s\n", node_idx, helper_kp_model_tensor_data_layout_to_string(tensor_desc->data_layout));
            printf("        [%d] quantization parameters version = %u\n", node_idx, quantization_parameters->version);

            if (KP_MODEL_QUANTIZATION_PARAMS_VERSION_1 == quantization_parameters->version) {
                quantization_parameters_v1              = &(quantization_parameters->quantization_parameters_data.v1);
                quantized_fixed_point_descriptor_list   = quantization_parameters_v1->quantized_fixed_point_descriptor;

                printf("        [%d] quantization axis = %u\n", node_idx, quantization_parameters_v1->quantized_axis);

                for (int quant_info_idx = 0; quant_info_idx < quantization_parameters_v1->quantized_fixed_point_descriptor_num; quant_info_idx++) {
                    printf("        [%d] quantization parameters [%d] = ", node_idx, quant_info_idx);
                    helper_print_kp_quantized_fixed_point_descriptor(&(quantized_fixed_point_descriptor_list[quant_info_idx]));
                }
            } else {
                printf("        [%d] print nef single output tensor information quantization parameters in fail: invalid kp_model_tensor_shape_info_version_t\n", node_idx);
            }
        }

        printf("\n");
    }
}

void helper_get_device_usb_speed_by_port_id(kp_devices_list_t *devices_list, int port_id, int *link_speed)
{
    kp_device_descriptor_t *dev_descp;

    if (((0 < devices_list->num_dev)) && (0 == port_id))
    {
        dev_descp = &(devices_list->device[0]);
        *link_speed = dev_descp->link_speed;
        return;
    }
    else
    {
        for (int i = 0; i < devices_list->num_dev; i++)
        {
            dev_descp = &(devices_list->device[i]);

            if (port_id == dev_descp->port_id)
            {
                *link_speed = dev_descp->link_speed;
                return;
            }
        }

        printf("\033[91m[error] no kneron device found.\033[0m\n");
        *link_speed = KP_USB_SPEED_UNKNOWN;
    }
}

void helper_bounding_box_stabilization(uint32_t box_count_last, kp_bounding_box_t *boxes_last,
                                       uint32_t box_count_latest, kp_bounding_box_t *boxes_latest,
                                       uint32_t *box_count_stabilized, kp_bounding_box_t *boxes_stabilized,
                                       int pixel_offset, float box_score_threshold)
{
    bool box_used[YOLO_GOOD_BOX_MAX] = {false};

    uint32_t count = 0;

    for (uint32_t i = 0; i < box_count_latest; i++)
    {
        kp_bounding_box_t *box = &boxes_latest[i];

        for (uint32_t k = 0; k < box_count_last; k++)
        {
            if (box_used[k])
                continue;

            kp_bounding_box_t *obox = &boxes_last[k];

            if (box->class_num == obox->class_num &&
                abs((int)(box->x1 - obox->x1)) < pixel_offset &&
                abs((int)(box->x2 - obox->x2)) < pixel_offset &&
                abs((int)(box->y1 - obox->y1)) < pixel_offset &&
                abs((int)(box->y2 - obox->y2)) < pixel_offset)
            {
                box->x1 = obox->x1;
                box->x2 = obox->x2;
                box->y1 = obox->y1;
                box->y2 = obox->y2;
                box->score = fmax(box->score, obox->score);
                box_used[k] = true;
            }
        }

        if (box->score < box_score_threshold)
            continue;

        memcpy((void *)&boxes_stabilized[count], (void *)box, sizeof(kp_bounding_box_t));
        count++;
    }

    *box_count_stabilized = count;
}

void helper_convert_rgb888_to_bmp(const char *out_bmp_path, uint32_t width, uint32_t height, uint8_t *rgb888_buf)
{
    unsigned int bytes_per_pixel_bmp = 3; // This is constant (also for RAW8) for all supported raw formats
    int pixel_data_size = 0;
    int padding_byte_num = 0;

    // length of bmp image width in byte should be aligned with 4
    if (width * height % 4 != 0)
        padding_byte_num = 4 - width * bytes_per_pixel_bmp % 4;

    // pixel_data is used to store pixel data of bmp image
    pixel_data_size = (width * bytes_per_pixel_bmp + padding_byte_num) * height; // image width of bmp file should be a multiple of 4
    unsigned char *pixel_data = (unsigned char *)malloc(pixel_data_size);
    if(NULL == pixel_data) {
        printf("Error! malloc for image buffer failed\n");
        goto err;
    }

    for (uint32_t i = 0; i < height; i++)
    {
        for (uint32_t j = 0; j < width; j++)
        {
            int bmp_y = (int)(height - i - 1); // row order of bmp is from bottom to top
            int bmp_pixel_pos = (width * 3 + padding_byte_num) * bmp_y + j * 3;
            int rgb888_pos = width * 3 * i + j * 3;
            pixel_data[bmp_pixel_pos]     = rgb888_buf[rgb888_pos + 2]; // B
            pixel_data[bmp_pixel_pos + 1] = rgb888_buf[rgb888_pos + 1]; // G
            pixel_data[bmp_pixel_pos + 2] = rgb888_buf[rgb888_pos];     // R
        }
    }

    dump_bmp_file_from_bmp_pixel_data(out_bmp_path, width, height, pixel_data);

err:

    free(pixel_data);

    return;
}

int helper_get_quantization_parameters_v1_information(kp_quantization_parameters_v1_t* quantization_parameters_v1, int quantized_fixed_point_descriptor_idx, int *radix, float *scale)
{
    if ((NULL == quantization_parameters_v1) ||
        (NULL == radix) ||
        (NULL == scale)) {
        printf("error: NULL pointer input parameter\n");
        return KP_ERROR_INVALID_PARAM_12;
    }

    if (quantized_fixed_point_descriptor_idx >= quantization_parameters_v1->quantized_fixed_point_descriptor_num) {
        printf("error: index of quantized_fixed_point_descriptor out of range\n");
        return KP_ERROR_INVALID_PARAM_12;
    }

    kp_quantized_fixed_point_descriptor_t *quantized_fixed_point_descriptor = &(quantization_parameters_v1->quantized_fixed_point_descriptor[quantized_fixed_point_descriptor_idx]);

    *radix = quantized_fixed_point_descriptor->radix;

    switch (quantized_fixed_point_descriptor->scale_dtype)
    {
    case KP_DTYPE_INT8:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_int8;
        break;
    case KP_DTYPE_INT16:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_int16;
        break;
    case KP_DTYPE_INT32:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_int32;
        break;
    case KP_DTYPE_UINT8:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_uint8;
        break;
    case KP_DTYPE_UINT16:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_uint16;
        break;
    case KP_DTYPE_UINT32:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_uint32;
        break;
    case KP_DTYPE_FLOAT32:
        *scale = (float)quantized_fixed_point_descriptor->scale.scale_float32;
        break;
    default:
        printf("error: get invalide KneronKNE_DataType_enum_t ...\n");
        return KP_ERROR_INVALID_MODEL_21;
    }

    return KP_SUCCESS;
}

int helper_convert_onnx_data_to_npu_data(kp_tensor_descriptor_t *tensor_descriptor, float *onnx_data_buf, int32_t onnx_data_buf_size, int8_t **npu_data_buf, int32_t *npu_data_buf_size)
{
    int status                                                  = KP_SUCCESS;

    kp_tensor_shape_info_v2_t *tensor_shape_info                = NULL;

    uint32_t npu_data_layout                                    = KP_MODEL_TENSOR_DATA_LAYOUT_UNKNOWN;

    kp_quantization_parameters_v1_t *quantization_parameters_v1 = NULL;
    float quantization_max_value                                = 0;
    float quantization_min_value                                = 0;
    int32_t radix                                               = 0;
    float scale                                                 = 0;
    float quantization_factor                                   = 0;
    int quantized_axis_stride                                   = 0;

    int32_t *onnx_data_shape_index                              = NULL;
    int32_t onnx_data_buf_offset                                = 0;
    int32_t npu_data_buf_offset                                 = 0;

    int max_npu_stride_axis                                     = 0;
    uint32_t max_npu_stride                                     = 0;
    uint16_t npu_data_element_u16b                              = 0;
    uint16_t npu_data_element_i16b                              = 0;
    uint16_t npu_data_high_bit_offset                           = 16;

    if (NULL == tensor_descriptor ||
        NULL == onnx_data_buf ||
        0 == onnx_data_buf_size ||
        NULL == npu_data_buf ||
        NULL == npu_data_buf_size) {
        printf("convert onnx data to npu data fail: NULL pointer input parameters ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    if (KP_MODEL_TENSOR_SHAPE_INFO_VERSION_2 != tensor_descriptor->tensor_shape_info.version) {
        printf("convert onnx data to npu data fail: only support KP_MODEL_TENSOR_SHAPE_INFO_VERSION_2 tensor shape ...\n");
        status = KP_ERROR_INVALID_PARAM_12;
        goto FUNC_OUT;
    }

    quantization_parameters_v1  = &(tensor_descriptor->quantization_parameters.quantization_parameters_data.v1);
    tensor_shape_info           = &(tensor_descriptor->tensor_shape_info.tensor_shape_info_data.v2);
    npu_data_layout             = tensor_descriptor->data_layout;

    /* input data quantization */
    {
        // get clip value by data layout
        switch (npu_data_layout)
        {
        case KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B:
        case KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B:
        case KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B:
        case KP_MODEL_TENSOR_DATA_LAYOUT_RAW_8B:
            quantization_max_value = 127.0f;
            quantization_min_value = -128.0f;
            break;
        case KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B:
        case KP_MODEL_TENSOR_DATA_LAYOUT_RAW_16B:
        case KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8BHL:
        case KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8BHL:
        case KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8BHL:
            quantization_max_value = 32766.0f;
            quantization_min_value = -32768.0f;
            break;
        case KP_MODEL_TENSOR_DATA_LAYOUT_RAW_FLOAT:
            quantization_max_value = FLT_MAX;
            quantization_min_value = FLT_MIN;
            break;
        default:
            printf("error: get invalide data layout ...\n");
            status = KP_ERROR_INVALID_MODEL_21;
            goto FUNC_OUT;
        }

        // toolchain calculate the radix value from input data (after normalization), and set it into NEF model.
        // NPU will divide input data "2^radix" automatically, so, we have to scaling the input data here due to this reason.
        if (1 == quantization_parameters_v1->quantized_fixed_point_descriptor_num) {
            // layer-wise quantize
            radix = 0;
            scale = 0;
            if (KP_SUCCESS != helper_get_quantization_parameters_v1_information(quantization_parameters_v1, 0, &radix, &scale)) {
                printf("error: get invalide KneronKNE_DataType_enum_t ...\n");
                status = KP_ERROR_INVALID_MODEL_21;
                goto FUNC_OUT;
            }

            quantization_factor = powf(2, (float)radix) * scale;

            for (onnx_data_buf_offset = 0; onnx_data_buf_offset < onnx_data_buf_size; onnx_data_buf_offset++) {
                onnx_data_buf[onnx_data_buf_offset] = (float)kneron_round(onnx_data_buf[onnx_data_buf_offset] * quantization_factor);
                onnx_data_buf[onnx_data_buf_offset] = MAX(quantization_min_value, MIN(onnx_data_buf[onnx_data_buf_offset], quantization_max_value));
            }
        } else if (1 < quantization_parameters_v1->quantized_fixed_point_descriptor_num) {
            // channel-wise quantize
            quantized_axis_stride = 1;
            for (int axis = 0; axis < tensor_descriptor->tensor_shape_info.tensor_shape_info_data.v2.shape_len; axis++) {
                if (axis != quantization_parameters_v1->quantized_axis) {
                    quantized_axis_stride *= tensor_descriptor->tensor_shape_info.tensor_shape_info_data.v2.shape[axis];
                }
            }

            onnx_data_buf_offset = 0;
            for (int quantized_fixed_point_descriptor_idx = 0; quantized_fixed_point_descriptor_idx < quantization_parameters_v1->quantized_fixed_point_descriptor_num; quantized_fixed_point_descriptor_idx++) {
                radix = 0;
                scale = 0;

                if (KP_SUCCESS != helper_get_quantization_parameters_v1_information(quantization_parameters_v1, quantized_fixed_point_descriptor_idx, &radix, &scale)) {
                    printf("error: get invalide KneronKNE_DataType_enum_t ...\n");
                    status = KP_ERROR_INVALID_MODEL_21;
                    goto FUNC_OUT;
                }

                quantization_factor = powf(2, (float)radix) * scale;

                for (int quantized_axis_offset = 0; quantized_axis_offset < quantized_axis_stride; quantized_axis_offset++) {
                    onnx_data_buf[onnx_data_buf_offset] = (float)kneron_round(onnx_data_buf[onnx_data_buf_offset] * quantization_factor);
                    onnx_data_buf[onnx_data_buf_offset] = MAX(quantization_min_value, MIN(onnx_data_buf[onnx_data_buf_offset], quantization_max_value));
                    onnx_data_buf_offset += 1;
                }
            }
        } else {
            printf("error: get invalide number of quantized_fixed_point_descriptor ...\n");
            status = KP_ERROR_INVALID_MODEL_21;
            goto FUNC_OUT;
        }
    }

    /* re-layout the data to fit NPU data layout format */
    {
        for (int axis = 0; axis < tensor_shape_info->shape_len; axis++) {
            if (tensor_shape_info->stride_npu[axis] > max_npu_stride) {
                max_npu_stride      = tensor_shape_info->stride_npu[axis];
                max_npu_stride_axis = axis;
            }
        }

        onnx_data_shape_index   = calloc(tensor_shape_info->shape_len, sizeof(int32_t));
        *npu_data_buf_size      = tensor_shape_info->shape[max_npu_stride_axis] * max_npu_stride;

        if (NULL == onnx_data_shape_index) {
            printf("error: malloc working buffer onnx_data_shape_index fail ...\n");
            status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
            goto FUNC_OUT;
        }

        switch (npu_data_layout)
        {
        case KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8B:
        case KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8B:
        case KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8B:
        case KP_MODEL_TENSOR_DATA_LAYOUT_RAW_8B:
            *npu_data_buf_size  = *npu_data_buf_size * sizeof(int8_t);
            *npu_data_buf       = realloc(*npu_data_buf, *npu_data_buf_size);

            if (NULL == *npu_data_buf) {
                printf("error: malloc working buffer npu_data_buf fail ...\n");
                status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
                goto FUNC_OUT;
            }

            memset(*npu_data_buf, 0, *npu_data_buf_size);

            while (true) {
                onnx_data_buf_offset    = 0;
                npu_data_buf_offset     = 0;

                for (int32_t axis = 0; axis < tensor_shape_info->shape_len; axis++) {
                    onnx_data_buf_offset    += onnx_data_shape_index[axis] * tensor_shape_info->stride_onnx[axis];
                    npu_data_buf_offset     += onnx_data_shape_index[axis] * tensor_shape_info->stride_npu[axis];
                }

                ((int8_t *)*npu_data_buf)[npu_data_buf_offset] = (int8_t)onnx_data_buf[onnx_data_buf_offset];

                for (int32_t axis = tensor_shape_info->shape_len - 1; axis >= 0; axis--) {
                    onnx_data_shape_index[axis]++;
                    if (onnx_data_shape_index[axis] == tensor_shape_info->shape[axis]) {
                        if (axis == 0)
                            break;

                        onnx_data_shape_index[axis] = 0;
                        continue;
                    } else {
                        break;
                    }
                }

                if (onnx_data_shape_index[0] == tensor_shape_info->shape[0]) {
                    break;
                }
            }
            break;
        case KP_MODEL_TENSOR_DATA_LAYOUT_8W1C16B:
        case KP_MODEL_TENSOR_DATA_LAYOUT_RAW_16B:
            *npu_data_buf_size  = *npu_data_buf_size * sizeof(int16_t);
            *npu_data_buf       = realloc(*npu_data_buf, *npu_data_buf_size);

            if (NULL == *npu_data_buf) {
                printf("error: malloc working buffer npu_data_buf fail ...\n");
                status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
                goto FUNC_OUT;
            }

            memset(*npu_data_buf, 0, *npu_data_buf_size);

            while (true) {
                onnx_data_buf_offset    = 0;
                npu_data_buf_offset     = 0;

                for (int32_t axis = 0; axis < tensor_shape_info->shape_len; axis++) {
                    onnx_data_buf_offset    += onnx_data_shape_index[axis] * tensor_shape_info->stride_onnx[axis];
                    npu_data_buf_offset     += onnx_data_shape_index[axis] * tensor_shape_info->stride_npu[axis];
                }

                npu_data_element_i16b                               = (int16_t)onnx_data_buf[onnx_data_buf_offset];
                npu_data_element_u16b                               = *((uint16_t *)(&npu_data_element_i16b));
                ((uint16_t *)*npu_data_buf)[npu_data_buf_offset]    = (npu_data_element_u16b & 0xfffeu);

                for (int32_t axis = tensor_shape_info->shape_len - 1; axis >= 0; axis--) {
                    onnx_data_shape_index[axis]++;
                    if (onnx_data_shape_index[axis] == tensor_shape_info->shape[axis]) {
                        if (axis == 0)
                            break;

                        onnx_data_shape_index[axis] = 0;
                        continue;
                    } else {
                        break;
                    }
                }

                if (onnx_data_shape_index[0] == tensor_shape_info->shape[0])
                    break;
            }
            break;
        case KP_MODEL_TENSOR_DATA_LAYOUT_4W4C8BHL:
        case KP_MODEL_TENSOR_DATA_LAYOUT_1W16C8BHL:
        case KP_MODEL_TENSOR_DATA_LAYOUT_16W1C8BHL:
            *npu_data_buf_size  = *npu_data_buf_size * sizeof(uint16_t);
            *npu_data_buf       = realloc(*npu_data_buf, *npu_data_buf_size);

            if (NULL == *npu_data_buf) {
                printf("error: malloc working buffer npu_data_buf fail ...\n");
                status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
                goto FUNC_OUT;
            }

            memset(*npu_data_buf, 0, *npu_data_buf_size);

            while (true) {
                onnx_data_buf_offset    = 0;
                npu_data_buf_offset     = 0;

                for (int32_t axis = 0; axis < tensor_shape_info->shape_len; axis++) {
                    onnx_data_buf_offset    += onnx_data_shape_index[axis] * tensor_shape_info->stride_onnx[axis];
                    npu_data_buf_offset     += onnx_data_shape_index[axis] * tensor_shape_info->stride_npu[axis];
                }

                /* npu_data_buf_offset = (npu_data_buf_offset / 16) * 32 + (npu_data_buf_offset % 16) */
                npu_data_buf_offset = ((npu_data_buf_offset >> 4) << 5) + (npu_data_buf_offset & 15u);

                npu_data_element_i16b                                                       = (int16_t)onnx_data_buf[onnx_data_buf_offset];
                npu_data_element_u16b                                                       = ((*((uint16_t *)(&npu_data_element_i16b))) >> 1);
                ((uint8_t *)*npu_data_buf)[npu_data_buf_offset]                             = (uint8_t)(npu_data_element_u16b & 0x007fu);
                ((uint8_t *)*npu_data_buf)[npu_data_buf_offset + npu_data_high_bit_offset]  = (uint8_t)((npu_data_element_u16b >> 7) & 0x00ffu);

                for (int32_t axis = tensor_shape_info->shape_len - 1; axis >= 0; axis--) {
                    onnx_data_shape_index[axis]++;
                    if (onnx_data_shape_index[axis] == tensor_shape_info->shape[axis]) {
                        if (axis == 0)
                            break;

                        onnx_data_shape_index[axis] = 0;
                        continue;
                    } else {
                        break;
                    }
                }

                if (onnx_data_shape_index[0] == tensor_shape_info->shape[0])
                    break;
            }
            break;
        case KP_MODEL_TENSOR_DATA_LAYOUT_RAW_FLOAT:
            *npu_data_buf_size  = *npu_data_buf_size * sizeof(float);
            *npu_data_buf        = realloc(*npu_data_buf, *npu_data_buf_size);

            if (NULL == *npu_data_buf) {
                printf("error: malloc working buffer npu_data_buf fail ...\n");
                status = KP_ERROR_MEMORY_ALLOCATION_FAILURE_9;
                goto FUNC_OUT;
            }

            while (true) {
                onnx_data_buf_offset    = 0;
                npu_data_buf_offset     = 0;

                for (int32_t axis = 0; axis < tensor_shape_info->shape_len; axis++) {
                    onnx_data_buf_offset    += onnx_data_shape_index[axis] * tensor_shape_info->stride_onnx[axis];
                    npu_data_buf_offset     += onnx_data_shape_index[axis] * tensor_shape_info->stride_npu[axis];
                }

                ((float *)*npu_data_buf)[npu_data_buf_offset] = (float)onnx_data_buf[onnx_data_buf_offset];

                for (int32_t axis = tensor_shape_info->shape_len - 1; axis >= 0; axis--) {
                    onnx_data_shape_index[axis]++;
                    if (onnx_data_shape_index[axis] == tensor_shape_info->shape[axis]) {
                        if (axis == 0)
                            break;

                        onnx_data_shape_index[axis] = 0;
                        continue;
                    } else {
                        break;
                    }
                }

                if (onnx_data_shape_index[0] == tensor_shape_info->shape[0]) {
                    break;
                }
            }
            break;
        }
    }

FUNC_OUT:
    if (NULL != onnx_data_shape_index)
        free(onnx_data_shape_index);

    return status;
}
