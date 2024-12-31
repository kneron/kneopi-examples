/**
 * @file      ncpu_gen_struct.h
 * @brief     NCPU generic data structures used for API
 * @copyright (c) 2020-2023 Kneron Inc. All right reserved.
 */

#ifndef __NCPU_GEN_STRUCT_H__
#define __NCPU_GEN_STRUCT_H__

#include <stdint.h>
#include <stdbool.h>
#include "ipc.h"

#if defined(KL720) || defined(KL630)
    #include "model_parser_api.h"
#elif defined(KL730)
    #include "model_parser_api_kne.h"
#endif

/* Helper macros */
#define RAW_INFERENCE_FORMAT(image_p)           (image_p->raw_img_p->inf_format)
#define RAW_IMAGE_NUM_INPUT(image_p)            (image_p->raw_img_p->num_image)
#define RAW_IMAGE_MEM_ADDR(image_p, i)          (image_p->raw_img_p->image_list[i].image_mem_addr)
#define RAW_IMAGE_MEM_LEN(image_p, i)           (image_p->raw_img_p->image_list[i].image_mem_len)
#define RAW_FORMAT(image_p, i)                  (image_p->raw_img_p->image_list[i].format)
#define RAW_INPUT_ROW(image_p, i)               (image_p->raw_img_p->image_list[i].input_row)
#define RAW_INPUT_COL(image_p, i)               (image_p->raw_img_p->image_list[i].input_col)
#define RAW_CROP_TOP(image_p, i)                (image_p->raw_img_p->image_list[i].params_s.crop_top)
#define RAW_CROP_BOTTOM(image_p, i)             (image_p->raw_img_p->image_list[i].params_s.crop_bottom)
#define RAW_CROP_LEFT(image_p, i)               (image_p->raw_img_p->image_list[i].params_s.crop_left)
#define RAW_CROP_RIGHT(image_p, i)              (image_p->raw_img_p->image_list[i].params_s.crop_right)
#define RAW_PAD_TOP(image_p, i)                 (image_p->raw_img_p->image_list[i].params_s.pad_top)
#define RAW_PAD_BOTTOM(image_p, i)              (image_p->raw_img_p->image_list[i].params_s.pad_bottom)
#define RAW_PAD_LEFT(image_p, i)                (image_p->raw_img_p->image_list[i].params_s.pad_left)
#define RAW_PAD_RIGHT(image_p, i)               (image_p->raw_img_p->image_list[i].params_s.pad_right)
#define RAW_SCALE_WIDTH(image_p, i)             (image_p->raw_img_p->image_list[i].params_s.scale_width)
#define RAW_SCALE_HEIGHT(image_p, i)            (image_p->raw_img_p->image_list[i].params_s.scale_height)
#if defined(KL630) || defined(KL730)
#define RAW_FLIP_FACE(image_p, i)               (image_p->raw_img_p->image_list[i].params_s.flip_face)
#endif
#define RAW_PRE_PROC_CONFIG(image_p)            (image_p->raw_img_p->pre_proc_config)
#define RAW_POST_PROC_CONFIG(image_p)           (image_p->raw_img_p->post_proc_config)

#if defined(KL520)
#define RAW_TICK_START_PRE(image_p)             (image_p->raw_img_p->tick_start_pre)
#define RAW_TICK_END_PRE(image_p)               (image_p->raw_img_p->tick_end_pre)
#define RAW_TICK_START_NPU(image_p)             (image_p->raw_img_p->tick_start_npu)
#define RAW_TICK_END_NPU(image_p)               (image_p->raw_img_p->tick_end_npu)
#define RAW_TICK_START_POST(image_p)            (image_p->raw_img_p->tick_start_post)
#define RAW_TICK_END_POST(image_p)              (image_p->raw_img_p->tick_end_post)

#define PREPROC_INPROC_MEM_ADDR(image_p, i)     (image_p->model_preproc.node_p[i].inproc_mem_addr)
#endif
#define PREPROC_INPUT_NUM(image_p)              (image_p->model_preproc.input_num)

#define DIM_INPUT_ROW(image_p, i)               (image_p->model_preproc.node_p[i].dim.input_row)
#define DIM_INPUT_COL(image_p, i)               (image_p->model_preproc.node_p[i].dim.input_col)
#define DIM_INPUT_CH(image_p, i)                (image_p->model_preproc.node_p[i].dim.input_channel)

#define PREPROC_INPUT_MEM_ADDR(image_p, i)      (image_p->model_preproc.node_p[i].input_mem_addr)
#define PREPROC_INPUT_MEM_LEN(image_p, i)       (image_p->model_preproc.node_p[i].input_mem_len)
#define PREPROC_INPUT_MEM_ADDR2(image_p, i)     (image_p->model_preproc.node_p[i].input_mem_addr2)
#define PREPROC_INPUT_MEM_LEN2(image_p, i)      (image_p->model_preproc.node_p[i].input_mem_len2)
#define PREPROC_INPUT_RADIX(image_p, i)         (image_p->model_preproc.node_p[i].input_radix)
#define PREPROC_INPUT_FORMAT(image_p, i)        (image_p->model_preproc.node_p[i].input_format)
#define PREPROC_PARAMS_P(image_p, i)            (image_p->model_preproc.node_p[i].params_p)
#define PREPROC_INPUT_TENSOR_LIST(image_p)      (image_p->model_preproc.tensors_p)

#define POSTPROC_OUTPUT_NUM(image_p)            (image_p->postproc.output_num)
#define POSTPROC_OUTPUT_FORMAT(image_p)         (image_p->postproc.output_format)
#define POSTPROC_OUTPUT_MEM_ADDR(image_p)       (image_p->postproc.output_mem_addr)
#define POSTPROC_OUTPUT_MEM_LEN(image_p)        (image_p->postproc.output_mem_len)
#define POSTPROC_RESULT_MEM_ADDR(image_p)       (image_p->postproc.result_mem_addr)
#define POSTPROC_RESULT_MEM_LEN(image_p)        (image_p->postproc.result_mem_len)
#define POSTPROC_RESULT_MODEL_ID(image_p)       (image_p->postproc.model_id)
#define POSTPROC_PARAMS_P(image_p)              (image_p->postproc.params_p)
#define POSTPROC_OUTPUT_TENSOR_LIST(image_p)    (image_p->postproc.tensors_p)
#define POSTPROC_RAW_RESULT_MEM_ADDR(image_p)   (image_p->postproc.raw_result_mem_addr)
#if defined(KL520)
#define POSTPROC_OUTPUT_MEM_ADDR2(image_p)      (image_p->postproc.output_mem_addr2)
#define POSTPROC_OUTPUT_MEM_LEN2(image_p)       (image_p->postproc.output_mem_len2)
#endif
#define POSTPROC_OUTPUT_MEM_ADDR3(image_p)      (image_p->postproc.output_mem_addr3)
#define POSTPROC_OUTPUT_MEM_ADDR4(image_p)      (image_p->postproc.output_mem_addr4)

#if defined(KL520)
#define OUT_NODE_ADDR_PARALLEL(out_p, image_p) \
    (OUT_NODE_ADDR(out_p) + POSTPROC_OUTPUT_MEM_ADDR(image_p) - MODEL_OUTPUT_MEM_ADDR(image_p))

#define CPU_OP_NODE(image_p)                    (image_p->cpu_op.node_p)
#define CPU_OP_NODE_OP_TYPE(image_p)            (image_p->cpu_op.node_p->op_type)
#define CPU_OP_NODE_INPUT_COL(image_p)          (image_p->cpu_op.node_p->in_num_col)
#define CPU_OP_NODE_INPUT_ROW(image_p)          (image_p->cpu_op.node_p->in_num_row)
#define CPU_OP_NODE_INPUT_CH(image_p)           (image_p->cpu_op.node_p->in_num_ch)
#define CPU_OP_NODE_INPUT_ADDR(image_p)         (image_p->cpu_op.node_p->input_datanodes[0].node_list[0].addr)
#define CPU_OP_NODE_OUTPUT_COL(image_p)         (image_p->cpu_op.node_p->out_num_col)
#define CPU_OP_NODE_OUTPUT_ROW(image_p)         (image_p->cpu_op.node_p->out_num_row)
#define CPU_OP_NODE_OUTPUT_CH(image_p)          (image_p->cpu_op.node_p->out_num_ch)
#define CPU_OP_NODE_OUTPUT_ADDR(image_p)        (image_p->cpu_op.node_p->output_datanode.node_list[0].addr)
#endif

/*****************************************************************
 * tensor information struct
 *****************************************************************/

/**
 * @brief address mode for tensor NPU data address.
 */
typedef enum
{
    NGS_MODEL_TENSOR_ADDRESS_MODE_UNKNOWN           = 0,                            /**< unknow address mode */
    NGS_MODEL_TENSOR_ADDRESS_MODE_ABSOLUTE          = 1,                            /**< absolute mode */
    NGS_MODEL_TENSOR_ADDRESS_MODE_RELATIVE          = 2                             /**< relative mode */
} ngs_model_tensor_address_mode_t;

/**
 * @brief shape information version for tensors.
 */
typedef enum
{
    NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_UNKNOWN     = 0,                            /**< unknow version */
    NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_1           = 1,                            /**< version 1 - for KL520, KL720 and KL630 */
    NGS_MODEL_TENSOR_SHAPE_INFO_VERSION_2           = 2                             /**< version 2 - for KL730 */
} ngs_model_tensor_shape_info_version_t;

/**
 * @brief quantization parameters version for tensors.
 */
typedef enum
{
    NGS_MODEL_QUANTIZATION_PARAMS_VERSION_UNKNOWN   = 0,                            /**< unknow version */
    NGS_MODEL_QUANTIZATION_PARAMS_VERSION_1         = 1,                            /**< version 1 - for KL520, KL720, KL630 and KL730 */
} ngs_quantization_parameters_version_t;

/**
 * @brief enum for kneron data type
 */
typedef enum
{
    NGS_DTYPE_UNKNOWN                               = 0,                            /**< unknown data type */
    NGS_DTYPE_INT8                                  = 1,                            /**< represent one scale value by int8_t data type */
    NGS_DTYPE_INT16                                 = 2,                            /**< represent one scale value by int16_t data type */
    NGS_DTYPE_INT32                                 = 3,                            /**< represent one scale value by int32_t data type */
    NGS_DTYPE_INT64                                 = 4,                            /**< represent one scale value by int64_t data type */
    NGS_DTYPE_UINT8                                 = 5,                            /**< represent one scale value by uint8_t data type */
    NGS_DTYPE_UINT16                                = 6,                            /**< represent one scale value by uint16_t data type */
    NGS_DTYPE_UINT32                                = 7,                            /**< represent one scale value by uint32_t data type */
    NGS_DTYPE_UINT64                                = 8,                            /**< represent one scale value by uint64_t data type */
    NGS_DTYPE_FLOAT32                               = 9,                            /**< represent one scale value by float32 data type */
    NGS_DTYPE_BFLOAT16                              = 10,                           /**< represent one scale value by bfloat16 data type (store in uint16_t 2 bytes) */
    NGS_DTYPE_DOUBLE64                              = 11                            /**< represent one scale value by double64 data type */
} ngs_dtype_t;

/**
 * @brief a scale of node
 */
typedef union
{
    int8_t scale_int8;                                                              /**< scale of node in data type int8 */
    int16_t scale_int16;                                                            /**< scale of node in data type int16 */
    int32_t scale_int32;                                                            /**< scale of node in data type int32 */
    int64_t scale_int64;                                                            /**< scale of node in data type int64 */
    uint8_t scale_uint8;                                                            /**< scale of node in data type uint8 */
    uint16_t scale_uint16;                                                          /**< scale of node in data type uint16 */
    uint32_t scale_uint32;                                                          /**< scale of node in data type uint32 */
    uint64_t scale_uint64;                                                          /**< scale of node in data type uint64 */
    uint16_t scale_bfloat16;                                                        /**< scale of node in data type bfloat16 (store in uint16_t 2 bytes) */
    float scale_float32;                                                            /**< scale of node in data type float32 */
    double scale_double64;                                                          /**< scale of node in data type double64 */
} ngs_scale_t;

/**
 * @brief a basic descriptor for a fixed-point quantization information
 */
typedef struct
{
    int32_t                                 radix;                                  /**< radix of node */
    uint32_t                                scale_dtype;                            /**< datatype of scale (ref. ngs_dtype_t) */
    ngs_scale_t                             scale;                                  /**< scale of node */
} __attribute__((packed, aligned(4))) ngs_quantized_fixed_point_descriptor_t;

/**
 * @brief a basic descriptor for quantization parameters (version 1)
 */
typedef struct
{
    uint32_t                                quantized_axis;                         /**< the axis along which the fixed-point quantization information performed */
    uint32_t                                quantized_fixed_point_descriptor_num;   /**< numbers of fixed-point quantization information */
    ngs_quantized_fixed_point_descriptor_t* quantized_fixed_point_descriptor;       /**< array of fixed-point quantization information */
} __attribute__((packed, aligned(4))) ngs_quantization_parameters_v1_t;

/**
 * @brief a basic descriptor for quantization parameters (version 1)
 */
typedef union
{
    ngs_quantization_parameters_v1_t        v1;                                     /**< quantization parameters - version 1 */
} ngs_quantization_parameters_data_t;

/**
 * @brief a basic descriptor for quantization parameters
 */
typedef struct
{
    uint32_t                                version;                                /**< quantization parameters version (ref. ngs_model_tensor_shape_info_version_t) */
    ngs_quantization_parameters_data_t      quantization_parameters_data;           /**< quantization parameters data */
} __attribute__((packed, aligned(4))) ngs_quantization_parameters_t;

/**
 * @brief a shape descriptor for a tensor (version 1)
 */
typedef struct
{
    uint32_t                                shape_npu_len;                          /**< length of npu shape (Default value: 4) */
    int32_t*                                shape_npu;                              /**< npu shape (Default dimension order: BxCxHxW) */
    uint32_t                                shape_onnx_len;                         /**< length of onnx shape */
    int32_t*                                shape_onnx;                             /**< onnx shape */
    uint32_t                                axis_permutation_len;                   /**< length of remap axis permutation */
    int32_t*                                axis_permutation_onnx_to_npu;           /**< remap axis permutation from onnx to npu shape (shape_intrp_dim) */
} __attribute__((packed, aligned(4))) ngs_tensor_shape_info_v1_t;

/**
 * @brief a shape descriptor for a tensor (version 2)
 */
typedef struct
{
    uint32_t                                shape_len;                              /**< length of shape */
    int32_t*                                shape;                                  /**< shape */
    uint32_t*                               stride_onnx;                            /**< data access stride of ONNX (in scalar) */
    uint32_t*                               stride_npu;                             /**< data access stride of NPU (in scalar) */
} __attribute__((packed, aligned(4))) ngs_tensor_shape_info_v2_t;

/**
 * @brief a shape descriptor data for a tensor
 */
typedef union
{
    ngs_tensor_shape_info_v1_t              v1;                                     /**< shape information - version 1 */
    ngs_tensor_shape_info_v2_t              v2;                                     /**< shape information - version 2 */
} ngs_tensor_shape_info_data_t;

/**
 * @brief a basic descriptor for a tensor in model
 */
typedef struct
{
    uint32_t                                version;                                /**< shape information version (ref. ngs_model_tensor_shape_info_version_t) */
    ngs_tensor_shape_info_data_t            tensor_shape_info_data;                 /**< shape information data */
} __attribute__((packed, aligned(4))) ngs_tensor_shape_info_t;

/**
 * @brief a basic descriptor for a tensor in model
 */
typedef struct
{
    uintptr_t                               base_pointer;                           /**< pointer to NPU raw data of the specified output tensor */
    uint64_t                                base_pointer_offset;                    /**< specified output tensor NPU data relative offset from the NPU output address (only served in relative address mode) */
    uint32_t                                base_pointer_address_mode;              /**< if the base pointer is in relative address mode or not (ref. ngs_model_tensor_address_mode_t) */
    uint32_t                                index;                                  /**< index of node */
    uint32_t                                data_layout;                            /**< npu memory layout (ref. ngs_model_tensor_data_layout_t) */
    ngs_tensor_shape_info_t                 tensor_shape_info;                      /**< shape information */
    ngs_quantization_parameters_t           quantization_parameters;                /**< quantization parameters */
} __attribute__((packed, aligned(4))) ngs_tensor_t;

/*****************************************************************
 * Legacy setup.bin model information
 *****************************************************************/

/* Type of Operations */
#define FW_IN_NODE                              0
#define FW_CPU_NODE                             1
#define FW_OUTPUT_NODE                          2
#define FW_DATA_NODE                            3
#define FW_SUPER_NODE                           4
#define FW_NETWORK_INPUT_NODE                   5

/* Structure of CNN Header in setup.bin */
typedef struct CNN_Header {
#if defined(KL520)
    uint32_t                            crc;
    uint32_t                            version;
    uint32_t                            key_offset;
    uint32_t                            model_type;
    uint32_t                            app_type;
    uint32_t                            dram_start;
    uint32_t                            dram_size;
    uint32_t                            input_row;
    uint32_t                            input_col;
    uint32_t                            input_channel;
    uint32_t                            cmd_start;
    uint32_t                            cmd_size;
    uint32_t                            weight_start;
    uint32_t                            weight_size;
    uint32_t                            input_start;
    uint32_t                            input_size;
    int32_t                             input_radix;
    uint32_t                            output_nums;
#elif defined(KL720)
    uint32_t                            crc;
    uint32_t                            version;
    uint32_t                            key_offset;
    uint32_t                            model_type;
    uint32_t                            application_type;
    uint32_t                            dram_start;
    uint32_t                            dram_size;
    uint32_t                            cmd_start;
    uint32_t                            cmd_size;
    uint32_t                            weight_start;
    uint32_t                            weight_size;
    uint32_t                            input_start;
    uint32_t                            input_size;
    uint32_t                            input_num;
    uint32_t                            output_num;
#endif
} CNN_Header;

/* Structure of Input Operation */
// node_id = 0
typedef struct In_Node {
    uint32_t                            node_id;
    uint32_t                            next_npu;
} In_Node;

/* Structures of Data Nodes */
// node_id = 4
typedef struct Super_Node {
    uint32_t                            node_id;
    uintptr_t                           addr;
    uint32_t                            row_start;
    uint32_t                            col_start;
    uint32_t                            ch_start;
    uint32_t                            row_length;
    uint32_t                            col_length;
    uint32_t                            ch_length;
} Super_Node;

// node_id = 3
typedef struct Data_Node {
    uint32_t                            node_id;
    uint32_t                            supernum;
    uint32_t                            data_format;
    int32_t                             data_radix;
    float                               data_scale;
    uint32_t                            row_start;
    uint32_t                            col_start;
    uint32_t                            ch_start;
    uint32_t                            row_length;
    uint32_t                            col_length;
    uint32_t                            ch_length;
#if defined(KL520)
    Super_Node                          node_list[1];
#else
    Super_Node *                        node_list;
#endif
} Data_Node;

/* Structure of CPU Operation */
// node_id = 1
typedef struct CPU_Node {
    uint32_t                            node_id;
    uint32_t                            input_datanode_num;
    uint32_t                            op_type;
    uint32_t                            in_num_row;
    uint32_t                            in_num_col;
    uint32_t                            in_num_ch;
    uint32_t                            out_num_row;
    uint32_t                            out_num_col;
    uint32_t                            out_num_ch;
    uint32_t                            h_pad;
    uint32_t                            w_pad;
    uint32_t                            kernel_h;
    uint32_t                            kernel_w;
    uint32_t                            stride_h;
    uint32_t                            stride_w;
#if defined(KL520)
    Data_Node                           output_datanode;
    Data_Node                           input_datanodes[1];
#else
    Data_Node *                         output_datanode;
    Data_Node *                         input_datanodes;
#endif
} CPU_Node;

/* Structure of Output Operation */
// node_id = 2
typedef struct Out_Node {
    uint32_t                            node_id;
    uint32_t                            supernum;
    uint32_t                            data_format;
    uint32_t                            row_start;
    uint32_t                            col_start;
    uint32_t                            ch_start;
    uint32_t                            row_length;
    uint32_t                            col_length;
    uint32_t                            ch_length;
    uint32_t                            output_index;
    int32_t                             output_radix;
    float                               output_scale;
#if defined(KL520)
    Super_Node                          node_list[1];
#else
    Super_Node *                        node_list;
#endif
} Out_Node;

// node_id = 5
typedef struct NetInput_Node {
    uint32_t                            node_id;
    uint32_t                            input_index;
    uint32_t                            input_format;
    uint32_t                            input_row;
    uint32_t                            input_col;
    uint32_t                            input_channel;
    uint32_t                            input_start;
    uint32_t                            input_size;
    int32_t                             input_radix;
} NetInput_Node;

typedef struct Operation_Node {
    uint32_t                            node_id;
    uint32_t                            buffer_index;
    In_Node *                           inN;
    Out_Node *                          outN;
    CPU_Node *                          cpuN;
    NetInput_Node *                     netinN;
    struct Operation_Node *             next;
} Operation_Node;

/* Structure of setup.bin file */
struct setup_struct_s {
    CNN_Header                          header;

#if defined(KL520)
    union {
        In_Node                         in_nd;
        Out_Node                        out_nd;
        CPU_Node                        cpu_nd;
    } nodes[1];
#endif
};

/* CNN input dimensions */
struct kdp_model_dim_s {
    uint32_t                            input_row;
    uint32_t                            input_col;
    uint32_t                            input_channel;
};

/* Structure of kdp_pre_proc_s */
typedef struct kdp_pre_proc_s {
    uintptr_t                           input_mem_addr;                 /* input image in memory for NPU */
    int32_t                             input_mem_len;

    uintptr_t                           input_mem_addr2;                /* Input working buffers for NPU */
    int32_t                             input_mem_len2;

    uintptr_t                           inproc_mem_addr;                /* data memory for inproc array (used in KL520 only) */

    int32_t                             input_radix;                    /* number of bits for input fraction */
    uint32_t                            input_format;                   /* input data format from NPU*/

    void *                              params_p;                       /* pre-process parameters for the model */

    struct kdp_model_dim_s              dim;                            /* Model dimension */
} kdp_pre_proc_t;

struct kdp_model_pre_proc_s {
    uint32_t                            input_num;                      /* number of input node */
    kdp_pre_proc_t*                     node_p;                      /* Pre process struct list */
    ngs_tensor_t*                       tensors_p;                      /* output tenor information list */
};

/* Structure of kdp_post_proc_s */
struct kdp_post_proc_s {
    uint32_t                            output_num;                     /* output number */

    uintptr_t                           output_mem_addr;                /* output data memory from NPU */
    int32_t                             output_mem_len;

    uintptr_t                           result_mem_addr;                /* result data memory from post processing */
    int32_t                             result_mem_len;

    uintptr_t                           output_mem_addr2;               /* 2nd output data memory for parallel processing (used in KL520 only) */
    uint32_t                            output_mem_len2;

    uintptr_t                           output_mem_addr3;               /* data memory for FD SSR Anchors */
    uintptr_t                           output_mem_addr4;               /* data memory for CenterNet or shared */

    uint32_t                            output_format;                  /* output data format from NPU
                                                                         *     BIT(0): =0, 8-bits
                                                                         *             =1, 16-bits
                                                                         */

    ngs_tensor_t*                       tensors_p;                      /* output tenor information list */
    Out_Node *                          node_p;                         /* output node parameter (legacy output tensor information) */
    void *                              params_p;                       /* post-process parameters for the model */
};

/* Structure of kdp_cpu_op_s (used in KL520 only) */
struct kdp_cpu_op_s {
    CPU_Node*                           node_p;                         /* cpu op node parameter */
};

#if defined(KL720) || defined(KL630)
/* Parsed model setup.bin information (legacy) */
typedef struct {
    int                                 nTotalNodes;
    CNN_Header*                         pSetupHead;
    Operation_Node*                     pNodeHead;
    kdp_model_t                         oModel;
    int                                 total_nodes;
    int                                 current_node_id;
    uintptr_t                           pNodePositions[MAX_CNN_NODES];
} parsed_fw_model_t;


/*****************************************************************
 * Flatbuffer setup.bin model information
 *****************************************************************/

/* Parsed model setup.bin information (flatbuffer) */
typedef struct {
    session_hdl_t                       parser_session_hdl;
    node_hdl_t                          in_node_hdl;
    node_hdl_t                          out_node_hdl;
    node_hdl_t                          cpu_node_hdl;
    node_hdl_t                          const_node_hdl;
} parsed_fw_model_flatbuffer_t;
#endif

/*****************************************************************
 * Main NCPU inference flow information
 *****************************************************************/

/* KDP image structure */
typedef struct kdp_image_s {
    /* a NCPU copy of raw image struct in in_comm_p */
    struct kdp_img_raw_s*               raw_img_p;

#if defined(KL520)
    struct kdp_model_s*                 model_p;
#elif defined(KL720) || defined(KL630)
    /**
     * To compatible legacy/flatbuffer version setup.bin information
     *
     * is_flatbuffer:           false - legacy. true - flatbuffer.
     * pParsedModel:            for legacy setup.bin model. (NULL when is_flatbuffer is true)
     * pParsedModelFlatbuffer:  for flatbuffer setup.bin model. (NULL when is_flatbuffer is false)
     */
    bool                                is_flatbuffer;
    parsed_fw_model_t*                  pParsedModel;
    parsed_fw_model_flatbuffer_t*       pParsedModelFlatbuffer;
#elif defined(KL730)
    /**
     * The KL730 KNE model handler
     */
    session_hdl_t                       parser_session_hdl;
    mdl_hdl_t                           mdl_hdl;
#endif

    int                                 model_id;
    int                                 slot_idx;
    char*                               setup_mem_p;

    void*                               pExtParam;
    int32_t                             nLenExtParam;

    /* Model dimension and Pre process struct */
    struct kdp_model_pre_proc_s         model_preproc;

    /* Post process struct */
    struct kdp_post_proc_s              postproc;

    struct kdp_cpu_op_s                 cpu_op;                         /* CPU operation struct (used in KL520 only) */
} kdp_image_t;


#if defined(KL520)
typedef void (*pre_post_fn)(void *dst_p, void *src_p, int size);
#elif defined(KL720)
#define    TOTAL_STANDALONE_MODULE      6

typedef int (*dsp_ftr_fn)(void  *pIn,  void  *pOut);

typedef struct {
    int                                 standalone_cmd;
    dsp_ftr_fn                          fn;
} dsp_ftr_node_t;
#endif

#if defined(KL520) || defined(KL720)
#define MAX_MODEL_REGISTRATIONS         50

/* Prototypes for pre/post callback functions */
typedef int (*cnn_pre_fn)(struct kdp_image_s *pKdpImage, uint32_t index);
typedef int (*cnn_post_fn)(struct kdp_image_s *pKdpImage);

typedef struct {
    int         model_id;
    cnn_pre_fn  fn;
} model_pre_func_t;

typedef struct {
    int         model_id;
    cnn_post_fn fn;
} model_post_func_t;
#endif

#endif    /* __NCPU_GEN_STRUCT_H__ */
