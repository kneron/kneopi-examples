/* Copyright (c) 2020 Kneron, Inc. All Rights Reserved.
 *
 * The information contained herein is property of Kneron, Inc.
 * Terms and conditions of usage are described in detail in Kneron
 * STANDARD SOFTWARE LICENSE AGREEMENT.
 *
 * Licensees are granted free, non-transferable use of the information.
 * NO WARRANTY of ANY KIND is provided. This heading must NOT be removed
 * from the file.
 */

/******************************************************************************
*  Filename:
*  ---------
*  ipc.h
*
*  Description:
*  ------------
*
*
******************************************************************************/

#ifndef _IPC_H_
#define _IPC_H_

/******************************************************************************
Head Block of The File
******************************************************************************/

#include <stdint.h>
#include <stdbool.h>
#include "base.h"

#if defined(KL720) && !defined(PY_WRAPPER)
#include "jpeg_intf.h"
#include "tof_intf.h"
#include "stereo_depth_init.h"
#else
#ifndef BOOLEAN_DEFINED
#ifndef boolean
typedef char  boolean;
#endif // boolean
#define BOOLEAN_DEFINED
#endif // BOOLEAN_DEFINED
#endif

#if defined(KL520)
#ifdef USE_64
typedef uint64_t kdp_size_t;
#else
typedef uint32_t kdp_size_t;
#endif // USE_64
#endif // KL520

//----------------------------

#define SCPU2NCPU_ID                        ('s'<<24 | 'c'<<16 | 'p'<<8 | 'u')
#define NCPU2SCPU_ID                        ('n'<<24 | 'c'<<16 | 'p'<<8 | 'u')

#if defined(KL730)
#define MULTI_MODEL_MAX                     50                                  /* Max active models in memory */
#else
#define MULTI_MODEL_MAX                     16                                  /* Max active models in memory */
#endif

#define IPC_IMAGE_ACTIVE_MAX                2                                   /* Max active images for NCPU/NPU */
#define IPC_IMAGE_MAX                       2                                   /* Max cycled buffer for images */
#define IPC_MODEL_MAX                       (MULTI_MODEL_MAX * IPC_IMAGE_ACTIVE_MAX)

/* Image process cmd_flags set by scpu */
#define IMAGE_STATE_INACTIVE                0
#define IMAGE_STATE_ACTIVE                  1
#define IMAGE_STATE_NPU_DONE                2
#define IMAGE_STATE_DONE                    3
#define IMAGE_STATE_JPEG_ENC_DONE           4
#define IMAGE_STATE_JPEG_DEC_DONE           5
#define IMAGE_STATE_ERR_DSP_BUSY            6
#define IMAGE_STATE_JPEG_ENC_FAIL           7
#define IMAGE_STATE_JPEG_DEC_FAIL           8
#define IMAGE_STATE_RECEIVING               9                                   /* need check with mozart firmware */
#define IMAGE_STATE_TOF_DEC_DONE            10
#define IMAGE_STATE_TOF_DEC_FAIL            11
#define IMAGE_STATE_TOF_CALC_IR_BRIGHT_DONE 12
#define IMAGE_STATE_TOF_CALC_IR_BRIGHT_FAIL 13
#define IMAGE_STATE_STEREO_DEPTH_DONE       14
#define IMAGE_STATE_STEREO_DEPTH_FAIL       15

/* Image process status set by ncpu */
#define IMAGE_STATE_IDLE                    0
#define IMAGE_STATE_NPU_BUSY                1
#define IMAGE_STATE_POST_PROCESSING         IMAGE_STATE_NPU_DONE
#define IMAGE_STATE_POST_PROCESSING_DONE    IMAGE_STATE_DONE
#define IMAGE_STATE_TIMEOUT                 (7)
#define IMAGE_STATE_PREPROC_ERROR           (-1)
#define IMAGE_STATE_NPU_ERROR               (-2)
#define IMAGE_STATE_POSTPROC_ERROR          (-3)

/* NPU error code (sync with kp_struct.h) */
#define IMAGE_STATE_NCPU_ERR_BEGIN          200
#define IMAGE_STATE_NCPU_INVALID_IMAGE      201

/* normalization control */
#define IMAGE_FORMAT_SUB128                 BIT31                               /* 1: sub 128 for each value */
#define IMAGE_FORMAT_RIGHT_SHIFT_ONE_BIT    BIT22                               /* 1: right shift for 1-bit (normalization)*/

/* cv rotate control */
#define IMAGE_FORMAT_ROT_MASK               (BIT30 | BIT29)
#define IMAGE_FORMAT_ROT_SHIFT              29

/*  -- setting values of ROT -- */
#define IMAGE_FORMAT_ROT_CLOCKWISE          0x01                                /* ROT 90  */
#define IMAGE_FORMAT_ROT_COUNTER_CLOCKWISE  0x02                                /* ROT 270 */
#define IMAGE_FORMAT_ROT_180                0x03                                /* TODO, ROT 180 */

/* flow control */
#define IMAGE_FORMAT_PARALLEL_PROC          BIT27                               /* 1: parallel execution of NPU and NCPU */
#define IMAGE_FORMAT_NOT_KEEP_RATIO         BIT26                               /* TODO, duplicated */

#define IMAGE_FORMAT_MODEL_AGE_GENDER       BIT24

#define IMAGE_FORMAT_DEFINED_PAD            BIT23                               /* Customized padding */

/* scale/crop control */
#define IMAGE_FORMAT_NO_UPSCALING           BIT25                               /* 1: no upscale, only downscale is supported */
                                                                                /* 0: both upscale and downscale are supported */

#define IMAGE_FORMAT_SYMMETRIC_PADDING      BIT21                               /* 1: symmetic padding; */
                                                                                /* 0: corner padding */
#define IMAGE_FORMAT_PAD_SHIFT              21

#define IMAGE_FORMAT_CHANGE_ASPECT_RATIO    BIT20                               /* 1: scale without padding; */
                                                                                /* 0: scale with padding */

/* flow control - 2 */
#define IMAGE_FORMAT_BYPASS_PRE             BIT19                               /* 1: bypass pre-process */
#define IMAGE_FORMAT_BYPASS_NPU_OP          BIT18                               /* 1: bypass NPU OP */
#define IMAGE_FORMAT_BYPASS_CPU_OP          BIT17                               /* 1: bypass CPU OP */

#define IMAGE_FORMAT_BYPASS_POST            BIT16                               /* 1: bypass post-process (output NPU result directly) */
#define IMAGE_FORMAT_RAW_OUTPUT             BIT28                               /* 1: bypass post-process (include meta data for data parsing )*/


/* supported image foramts            BIT7 - BIT0 */
#define IMAGE_FORMAT_NPU                    0x00FF                              /* settings: NPU_FORMAT_XXXX */

/*********************************************
 *  settings for IMAGE_FORMAT_NPU
 *********************************************/
#define NPU_FORMAT_RGBA8888                 0x00
#define NPU_FORMAT_YUV422                   0x10                                /* similiar to Y0CBY1CR */
#define NPU_FORMAT_NIR                      0x20

/* Determine the exact format with the data byte sequence in DDR memory:
 * [lowest byte]...[highest byte]
 */
#define NPU_FORMAT_YCBCR422                 0x30
#define NPU_FORMAT_YCBCR422_CRY1CBY0        0x30
#define NPU_FORMAT_YCBCR422_CBY1CRY0        0x31
#define NPU_FORMAT_YCBCR422_Y1CRY0CB        0x32
#define NPU_FORMAT_YCBCR422_Y1CBY0CR        0x33
#define NPU_FORMAT_YCBCR422_CRY0CBY1        0x34
#define NPU_FORMAT_YCBCR422_CBY0CRY1        0x35
#define NPU_FORMAT_YCBCR422_Y0CRY1CB        0x36
#define NPU_FORMAT_YCBCR422_Y0CBY1CR        0x37                                /* Y0CbY1CrY2CbY3Cr... */

#define NPU_FORMAT_YUV444                   0x40
#define NPU_FORMAT_YCBCR444                 0x50
#define NPU_FORMAT_RGB565                   0x60
#define NPU_FORMAT_YUV420                   0x70
// ------------------------------------------

#define MAX_CNN_NODES                       45                                  /* NetputNode, CPU nodes, Out Nodes, etc */
#define MAX_OUT_NODES                       40                                  /* max Out Nodes */

#define MAX_INT_FOR_ALIGN                   0x10000000
#define NCPU_CLOCK_CNT_PER_MS               500000

#define KP_DEBUG_BUF_SIZE                   (8 * 1024 * 1024)                   // FIXME, max is 1920x1080 RGB8888

#define MAX_PARAMS_LEN                      40                                  /* uint32_t */

#if defined(KL730)
#define MAX_INPUT_NODE_COUNT                100
#else
#define MAX_INPUT_NODE_COUNT                5
#endif

/*50k log buffer*/
#define MAX_LOG_LENGTH                      256
#define LOG_QUEUE_NUM                       200
#define FLAG_LOGGER_SCPU_IN                 (1U << 0)
#define FLAG_LOGGER_NCPU_IN                 (1U << 1)

/******************************************************************************
Declaration of Enum
******************************************************************************/

/* scpu_to_ncpu: cmd */
typedef enum {
    CMD_INVALID,
    CMD_INIT,
    CMD_RUN_NPU,
    CMD_SLEEP_NPU,
    CMD_JPEG_ENCODE,
    CMD_JPEG_DECODE,
    CMD_CROP_RESIZE,
    CMD_TOF_DECODE,
    CMD_TOF_DECODE_DUAL,
    CMD_TOF_CALC_IR_BRIGHT,
    CMD_STEREO_DEPTH_FUSION,
    CMD_SCPU_NCPU_TOTAL,
    CMD_ALIGN_32 = MAX_INT_FOR_ALIGN,
} scpu_ncpu_cmd_t;

/* in every IPC interrupt triggered by NCPU, SCPU check in_comm_p to see the data type */
typedef enum {
    NCPU_REQUEST_NEW_IMG = 1,
    NCPU_EXEC_RESULT,
    MSG_ALIGN_32 = MAX_INT_FOR_ALIGN,
} ncpu_scpu_ipc_msg_type_t;

/* overall SCPU/DSP status*/
typedef enum {
    NCPU_STS_ERROR = -1,
    NCPU_STS_READY = 0,                                                         /* DSP is ready to run new task */
    NCPU_STS_BUSY,                                                              /* one of CNN/JPEG ENC/JPEG DEC is running, cannot accept new  task now */
    NCPU_STS_INVALID_PARAM,                                                     /* invalid IPC parameters */
    STS_ALIGN_32 = MAX_INT_FOR_ALIGN,
} ncpu_status_t;

typedef enum {
    POST_PROC_FAIL = -1,
    POST_PROC_SUCCESS = 0,
    RET_ALIGN_32 = MAX_INT_FOR_ALIGN,
} post_proc_return_sts_t;

typedef enum {
    NCPU_NONE_RESULT = -1,
    NCPU_POSTPROC_RESULT = 1,
    NCPU_JPEG_ENC_RESULT,
    NCPU_JPEG_DEC_RESULT,
    NCPU_CROP_RESIZE_RESULT,
    NCPU_TOF_DEC_RESULT,
    NCPU_IR_BRIGHT_RESULT,
    NCPU_STEREO_DEPTH_RESULT,
    NCPU_RESULT_TYPE_MAX,
    RES_ALIGN_32 = MAX_INT_FOR_ALIGN,
} NCPU_TO_SCPU_RESULT_TYPE;

typedef enum {
    CROP_RESIZE_OPERATION_FAILED = -1,                                          /* Failure in doing operation */
    CROP_RESIZE_OPERATION_SUCCESS = 1,                                          /* Operation Succeded */
    CROP_RESIZE_OPERATION_INVALID_PARM,                                         /* Inavlid parameter provided */
    RESIZE_ALIGN_32 = MAX_INT_FOR_ALIGN,
} crop_resize_oper_sts_t;

enum {
    LOGGER_SCPU_IN = 0,
    LOGGER_NCPU_IN,
    LOGGER_OUT,
    LOGGER_TOTAL
};

/******************************************************************************
Declaration of data structure
******************************************************************************/
// Sec 5: structure, uniou, enum, linked list
/* Model structure */
typedef struct kdp_model_s {
    uint32_t                                model_type;                         /* Model type, defined in model_type.h */
    uint32_t                                model_version;                      /* Model version */

    uint32_t                                input_mem_addr;                     /* Input in memory */
    int32_t                                 input_mem_len;

    uint32_t                                output_mem_addr;                    /* Output in memory */
    int32_t                                 output_mem_len;

    uint32_t                                buf_addr;                           /* Working buffer */
    int32_t                                 buf_len;

    uint32_t                                cmd_mem_addr;                       /* command.bin in memory */
    int32_t                                 cmd_mem_len;

    uint32_t                                weight_mem_addr;                    /* weight.bin in memory */
    int32_t                                 weight_mem_len;

    uint32_t                                setup_mem_addr;                     /* setup.bin in memory */
    int32_t                                 setup_mem_len;
} kdp_model_t;

typedef struct kdp_model_s kdp_model_info_t;

/* Result structure of a model */
typedef struct result_buf_s {
    int32_t                                 model_id;
    uintptr_t                               result_mem_addr;
    int32_t                                 result_mem_len;
    int32_t                                 result_ret_len;
} result_buf_t;

/* Parameter structure of a raw image */
typedef struct parameter_s {
    /* Crop parameters or other purposes */
    int                                     crop_top;
    int                                     crop_bottom;
    int                                     crop_left;
    int                                     crop_right;

    /* Pad parameters or other purposes */
    int                                     pad_top;
    int                                     pad_bottom;
    int                                     pad_left;
    int                                     pad_right;

    float                                   scale_width;
    float                                   scale_height;

#if defined(KL630) || defined(KL730)
    /* IE driver padding mode for pre-processing
     0:change aspect ratio(no padding).
     1:keep aspect ratio, allow scaling, pad corner(right or bottom)
     2:keep aspect ratio, allow scaling, pad center(top/down or right/left)
     3:keep aspect ratio, not allow scaling, pad corner(right or bottom)
     4:keep aspect ratio, not allow scaling, pad center(top/down, or right/left)
    */
    uint32_t                                ie_pad_mode;
    int32_t                                 angle;
    int                                     flip_face;
#endif
} parameter_t;

typedef struct kdp_img_info_s {
    /* input image in memory */
    uintptr_t                               image_mem_addr;
    int32_t                                 image_mem_len;

    /* raw image dimensions */
    uint32_t                                input_row;
    uint32_t                                input_col;
    uint32_t                                input_channel;

    /* Raw image format and pre-process flags
     * refer to dsp_img_fmt_t
     * bit-31: = 1 : subtract 128
     * bit 30:29 00: no rotation; 01: rotate clockwise; 10: rotate counter clockwise; 11: reserved
     * bit 7:0: format
     */
    uint32_t                                format;

    /* Parameter structure */
    struct parameter_s                      params_s;
} kdp_img_info_t;

struct kdp_img_cfg {
    uint32_t                                num_image;
    kdp_img_info_t                          image_list[MAX_INPUT_NODE_COUNT];
    uint32_t                                inf_format;
    uint32_t                                image_buf_active_index;             /* scpu_to_ncpu->active_img_index */
};

struct kdp_crop_box_s {
    int32_t                                 top;
    int32_t                                 bottom;
    int32_t                                 left;
    int32_t                                 right;
};

struct kdp_pad_value_s {
    int32_t                                 pad_top;
    int32_t                                 pad_bottom;
    int32_t                                 pad_left;
    int32_t                                 pad_right;
};

typedef struct {
    uint32_t                                w;
    uint32_t                                h;
    uint32_t                                c;
} img_dim_t;

/* Raw image structure */
typedef struct kdp_img_raw_s {
    int                                     state;                              /* Image state: 1 = active, 0 = inactive */
    int                                     seq_num;                            /* Image sequence number */
    int                                     ref_idx;                            /* Image ref index */
    uint32_t                                num_image;                          /* Number of raw images */
    kdp_img_info_t                          image_list[MAX_INPUT_NODE_COUNT];   /* List of raw images */

    /* Parallel and raw output flags
     * refer to dsp_img_fmt_t
     */
    uint32_t                                inf_format;
    uint32_t                                pre_proc_config[MAX_PARAMS_LEN];    /* User define data for ncpu/npu pre-processing configuration */
    uint32_t                                post_proc_config[MAX_PARAMS_LEN];   /* User define data for ncpu/npu post-processing configuration */

    struct result_buf_s                     result;

    /* Test: SCPU total */
    uint32_t                                tick_start;
    uint32_t                                tick_end;
    uint32_t                                tick_got_ncpu_ack;

    /* Test: NCPU processes */
    uint32_t                                tick_start_parse;
    uint32_t                                tick_end_parse;
    uint32_t                                tick_start_inproc;
    uint32_t                                tick_end_inproc;
    uint32_t                                tick_start_pre;
    uint32_t                                tick_end_pre;
    uint32_t                                tick_start_npu;
    uint32_t                                tick_cnn_interrupt_rvd;
    uint32_t                                tick_end_npu;
    uint32_t                                tick_start_post;
    uint32_t                                tick_end_post;
    uint32_t                                tick_start_dram_copy;
    uint32_t                                tick_end_dram_copy;
    uint32_t                                tick_rslt_got_scpu_ack;
    uint32_t                                tick_ncpu_img_req;
    uint32_t                                tick_ncpu_img_ack;
    uint32_t                                tick_last_img_req;
} kdp_img_raw_t;

/* Image result structure */
typedef struct kdp_img_result_s {
#if defined(KL520)
    int                                     status;                             /* Processing status: 2 = done, 1 = running, 0 = unused */
#elif defined(KL720) || defined(KL630) || defined(KL730)
    post_proc_return_sts_t                  status  __attribute__((aligned (4)));
#endif

    int                                     seq_num;                            /* Image sequence number */

    /* result memory addr */
    //dummy information
    uintptr_t                               result_mem_addr;
} kdp_img_result_t;

typedef struct {
    uint32_t                                fmt;

    parameter_t                             param;

    img_dim_t                               src_dim;
    img_dim_t                               dst_dim;

    uintptr_t                               src_addr;
    uint32_t                                src_data_len;

    uintptr_t                               dst_addr;
    uint32_t                                dst_buf_size;
    uint32_t                                dst_filled_len;

    int32_t                                 seq_num;

    int32_t                                 bUseHwInproc;                       /* 1: use NPU HW inproc; 0: use DSP SW solution */
    uintptr_t                               tmp_buf_addr;                       /* this tmp_buf_addr is needed for SW crop/resize, not for HW inproc */
} crop_resize_param_t;

typedef struct {
    int32_t                                 rslt_type;                          /* NCPU_TO_SCPU_RESULT_TYPE */

    crop_resize_oper_sts_t                  sts     __attribute__((aligned (4)));

    uintptr_t                               out_addr;                           /* output buf */
    uint32_t                                out_len;                            /* output buf length */

    int32_t                                 seq_num;
} crop_resize_result_t;

/* Structure of nCPU->sCPU IPC Message data */
typedef struct ncpu_to_scpu_req_img_s {
    int32_t                                 bHandledByScpu;
    int32_t                                 ipc_type;                           /* ncpu_scpu_ipc_msg_type_t */
    int32_t                                 sts;                                /* ncpu_status_t */
} ncpu_to_scpu_req_img_t;

typedef struct {
    int                                     model_id;
    uint32_t                                tick_before_preprocess;
    uint32_t                                sum_ticks_preprocess;

    uint32_t                                tick_before_inference;
    uint32_t                                sum_ticks_inference;

    uint32_t                                tick_before_postprocess;
    uint32_t                                sum_ticks_postprocess;

    uint32_t                                tick_before_cpu_op;
    uint32_t                                sum_ticks_cpu_op;
    uint32_t                                sum_cpu_op_count;

    uint32_t                                sum_frame_count;
} kp_model_profile_t;

typedef struct {
    int                                     model_id;
    uint32_t                                cycle_before_preprocess;
    uint64_t                                sum_cycles_preprocess;

    uint32_t                                cycle_before_inference;
    uint64_t                                sum_cycles_inference;

    uint32_t                                cycle_before_postprocess;
    uint64_t                                sum_cycles_postprocess;

    uint32_t                                cycle_before_cpu_op;
    uint64_t                                sum_cycles_cpu_op;
    uint64_t                                sum_cpu_op_count;

    uint32_t                                sum_frame_count;
} kp_model_profile_cycle_t;

/* Structure of sCPU->nCPU Message */
typedef struct scpu_to_ncpu_s {
    uint32_t                                id;                                 /* = 'scpu' */
    volatile uint32_t                       bNcpuReceived;
    uint32_t                                cmd;                                /* scpu_ncpu_cmd_t */

    /*
     * debug control flags (dbg.h):
     *   bits 19-16: scpu debug level
     *   bits 03-00: ncpu debug level
     */
    uint32_t                                debug_flags;
    uint32_t                                kp_dbg_checkpoinots;
    uint32_t                                kp_dbg_enable_profile;              /* 1: enable, 0: disable */

    int32_t                                 active_img_index_rgb_liveness;      /* Used in KL520 only */

    /* Active models in memory and running */
    int32_t                                 num_models;                         /* sually, num_models=1 (only one active model) */
    struct kdp_model_s                      models[IPC_MODEL_MAX];              /* to save active modelInfo */
    int32_t                                 models_type[IPC_MODEL_MAX];         /* to save model type */
    int32_t                                 model_slot_index;

    /* working buffer in case in-proc is necessary, raw img will copy to here for in-proc,
       in-proc output will be placed in the input mem address in setup.bin */
    uintptr_t                               input_mem_addr2;
    int32_t                                 input_mem_len2;

    uintptr_t                               output_mem_addr2;                   /* Memory for DME */
    int32_t                                 output_mem_len2;

    uintptr_t                               output_mem_addr3;                   /* Memory for post processing (shared) */

    kdp_img_raw_t*                          pRawimg;                            /* SCPU need to alloc for every image */
    uintptr_t                               ncpu_img_req_msg_addr;              /* ncpu_to_scpu_req_img_t*, SCPU always get result from here */

    uintptr_t                               log_buf_base;
    int32_t                                 log_buf_len;

    /* support features extension or for standalone non-cnn features */
    void*                                   pExtInParam;                        /* pointer to extended parameter data structure */
    int32_t                                 nLenExtInParam;                     /* length of extended parameter data structure */

    void*                                   pExtOutRslt;                        /* pointer to extended feature result data structure */
    int32_t                                 nLenExtOutRslt;                     /* Length of extended feature result data */

    void*                                   pCpuNodeBuffer;                     /* pointer to working buffer for Cpu Node */
    int32_t                                 nLenCpuNodeBuffer;                  /* Length of working buffer for Cpu Node */

    struct kdp_img_raw_s                    raw_images[IPC_IMAGE_MAX];          /* Raw image information */

    uint32_t                                active_img_index;                   /* Used in KL520 only */
    uintptr_t                               inproc_mem_addr;                    /* Memory for pre processing command */

    uintptr_t                               output_mem_addr4;                   /* Memory for post processing parameters */

    void*                                   kp_dbg_buffer;                      /* Debug used buffer */

    kp_model_profile_t                      kp_model_profile_records[MULTI_MODEL_MAX];
} scpu_to_ncpu_t;

/* Structure of nCPU->sCPU IPC Message data */
typedef struct ncpu_to_scpu_postproc_result_s {
    int32_t                                 model_slot_index;                   /* RUN which model for this image */
    kdp_img_result_t                        img_result;
    uintptr_t                               OrigRawImgAddr;
} ncpu_to_scpu_postproc_result_t;

/* Structure of nCPU->sCPU Message */
typedef struct ncpu_to_scpu_s {
    volatile boolean                        print_log;
    uint8_t*                                p_log_buf_base;
    uint32_t                                id;                                 /* = 'ncpu' */
    int32_t                                 bHandledByScpu;
    ncpu_scpu_ipc_msg_type_t                ipc_type    __attribute__((aligned (4)));   /* ncpu_scpu_ipc_msg_type_t */
    ncpu_status_t                           sts         __attribute__((aligned (4)));   /* overall NCPU/DSP status */
    NCPU_TO_SCPU_RESULT_TYPE                out_type    __attribute__((aligned (4)));

    union {
        ncpu_to_scpu_postproc_result_t      postproc;
#if defined(KL720) && !defined(PY_WRAPPER)
        jpeg_encode_result_t                jpeg_encode;
        jpeg_decode_result_t                jpeg_decode;
        tof_decode_result_t                 tof_decode;
        ir_bright_result_t                  tof_ir_bright;
        crop_resize_result_t                crop_resize;
#endif
    } result;

    uint32_t                                extRsltAddr;

    ncpu_to_scpu_req_img_t                  req_img;
    volatile int32_t                        kp_dbg_status;
} ncpu_to_scpu_result_t;

typedef struct {
    volatile boolean                        init_done;
    volatile boolean                        willing[LOGGER_TOTAL];
    volatile uint8_t                        w_idx;
    volatile uint8_t                        r_idx;
    volatile uint8_t                        turn;
    uint8_t*                                p_msg;
} logger_mgt_t;

#endif //_IPC_H_
