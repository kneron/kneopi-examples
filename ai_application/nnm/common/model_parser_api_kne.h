#ifndef __MODEL_PARSER_API_H__
#define __MODEL_PARSER_API_H__
//#error __MODEL_PARSER_API_H__
/* --------------------------------------------------------------------------
 * Copyright (c) 2022 Kneron Inc. All rights reserved.
 *
 *      Name:    model_parser_api.h
 *      Purpose: Kneron model parser API for NEF v2.0
 *
 *---------------------------------------------------------------------------*/

#include <stdint.h>
#include <stdbool.h>

typedef enum {
    DEFAULT_TGT,
    KL730_TGT,
    KL540_TGT,
    KL630_TGT,
} mdl_target_t;

typedef enum {
    MDL_NPU_CORE,
    MDL_CPU_CORE,
    MDL_LOOP_CORE,
    MDL_NPU_CPU_CORE,
} mdl_core_type_t;

typedef enum {
    MDL_SEG_DEFAULT,
    MDL_SEG_INPUT,
    MDL_SEG_OUTPUT,
    MDL_SEG_WORK,
    MDL_SEG_WT,
    MDL_SEQ_CONST_INPUT,
    MDL_SEG_CPU_TEMP,
    TOTAL_SEGS
} mdl_seg_type_t;

typedef enum {
    EM_None = 1,
    EM_CUSTOM,
    EM_KNNum,
} mdl_encrypt_mode_t;

typedef enum {
    DRAM_FMT_UNKNOWN = -1,
    /* conv format */
    DRAM_FMT_1W16C8B_CH_COMPACT     = 0,
    DRAM_FMT_1W16C8BHL_CH_COMPACT   = 1,
    DRAM_FMT_4W4C8B                 = 2,
    DRAM_FMT_4W4C8BHL               = 3,
    DRAM_FMT_16W1C8B                = 4,
    DRAM_FMT_16W1C8BHL              = 5,
    DRAM_FMT_8W1C16B                = 6,
    /* psum data format */
    DRAM_FMT_PS_1W16C24B            = 7,
    /* conv format */
    DRAM_FMT_1W16C8B                = 8,
    DRAM_FMT_1W16C8BHL              = 9,
    /* inproc format */
    DRAM_FMT_HW4C8B_KEEP_A          = 10,
    DRAM_FMT_HW4C8B_DROP_A          = 11,
    DRAM_FMT_HW1C8B                 = 12,
    DRAM_FMT_HW1C16B_LE             = 13,
    DRAM_FMT_HW1C16B_BE             = 14,
    /* row format */
    DRAM_FMT_RAW8B                  = 100,
    DRAM_FMT_RAW16B                 = 101,
    DRAM_FMT_RAW_FLOAT              = 102
} mdl_dram_data_fmt_t;

typedef enum {
    DT_None,
    DT_Int8,
    DT_Int16,
    DT_Int32,
    DT_Int64,
    DT_UInt8,
    DT_UInt16,
    DT_Uint32,
    DT_UInt64,
    DT_Float,
    DT_Bfloat16,
    DT_Double
} mdl_data_type_t;

typedef enum {
    NOT_EXECUTED,
    DONE,
} mdl_core_state_t;

typedef enum {
    DMA_TYPE_UNKNOWN,
    DMA_TYPE_RDMA,
    DMA_TYPE_WDMA,
} dma_type_t;

typedef struct {
    uint8_t  bank;
    uint8_t  DMAType;       //dma_type_t
    uint8_t  association;   //mdl_seg_type_t
} DmaBankInfo_t;

typedef struct {
    uint32_t batch;
    uint32_t ch;
    uint32_t h;
    uint32_t w;
} mdl_shape_t;

#if 0
typedef struct {
    float   scale;
    int32_t radix;
} mdl_quant_factor_t;
#endif

typedef struct {
    int32_t *radix_p;
    float   *scale_p;
    uint32_t scale_count;
    mdl_data_type_t scale_type;
} mdl_quant_param_t;

typedef struct {
    uintptr_t      entry;
    uint32_t       len;
    mdl_seg_type_t seg;
} mdl_buffer_t;

typedef struct {
    int32_t   *pDep;
    int32_t   total_dep;
    int32_t   *pParent;
    int32_t   total_parent;
    int32_t   *pChildren;
    int32_t   total_children;
} core_dep_tbl_t;

typedef struct fw_info_s {
    int32_t               model_id;
    uint32_t              input_size;
    uint32_t              output_size;
    uint32_t              work_size;
    uint32_t              cpu_temp_size;
} buf_info_t;

typedef void * node_hdl_t;

typedef void * core_hdl_t;
typedef void * loop_hdl_t;
typedef void * tensor_hdl_t;
typedef void * cpu_operator_hdl_t;
typedef void * session_hdl_t;
typedef void * mdl_hdl_t;
typedef void * param_hdl_t;
typedef void * const_data_hdl_t;
typedef void * input_order_hdl_t;
typedef void * dependency_hdl_t;
typedef void * npu_eu_param_hdl_t;
typedef void * cpu_eu_param_hdl_t;
typedef void * loop_eu_param_hdl_t;

#define DMA_BANK_INFO_MAX_SETS     8

/*********************************************************************************
                     Global Functions
**********************************************************************************/

/**
Setup one parsing session

[In]: kne_addr - address of the nef data

Return: session handle; NULL - fail
*/
session_hdl_t kne_parser_open(uintptr_t kne_addr);

/**
Close one parsing session

[In]: session - session handle

Return: 0 - success; -1 - fail
*/
int kne_parser_close(session_hdl_t session);

/**
query the hardware target supported by this KNE binary

[In]: s_hdl - the session handle

Return: supported hardware target ID
*/
mdl_target_t kne_parser_get_target_id(session_hdl_t s_hdl);

/**
query how many models available in the loaded NEF
[In]: session - session handle

Return: total model number; -1 - fail
*/
int kne_parser_get_model_number(session_hdl_t session);

/**
get a model_id list in the loaded NEF

[In]: session - session handle
[Out]: model_list - pre-allocated array address, will be filled with model_id list

Return: total model number
*/
int kne_parser_get_model_list(session_hdl_t session, int *model_list);

/**
Retrieve out the model handle by model type

[In]: s_hdl - session handle
[In]: model_type - model ID defined in model_type.h

Return: model handle; NULL - fail
*/
mdl_hdl_t kne_parser_get_model(session_hdl_t s_hdl, uint32_t model_type);

/**
Retrieve out the buffer_info applicable for all models in this KNE.
The returned buffer sizes are max sizes across all embedded models,
it can be used for every model to run with it.

[In]: s_hdl - session handle
[In]: pInfo - point to the buf_info_t obj

Return: void
*/
void kne_parser_get_kne_buf_info(session_hdl_t s_hdl, buf_info_t *pInfo);

/**
Initialize one model

[In]: s_hdl - session handle
[In]: model_type - model id

Return: model handle; NULL - fail
*/
mdl_hdl_t kne_parser_mdl_init(session_hdl_t s_hdl, int model_type);

/**
DeInitialize one model, release related resources

[In]: hdl - model handle

Return: 0 - success; -1 - fail
*/
int kne_parser_mdl_deinit(mdl_hdl_t hdl);

/*********************************************************************************
                      Model Common APIs
**********************************************************************************/

/**
query if the model specified is initialized

[In]: mdl_hdl - the model handle

Return: TRUE for intialized
*/
bool mdl_parse_is_initialized(mdl_hdl_t mdl_hdl);

/**
query the model type id of this model

[In]: mdl_hdl - the model handle

Return: model ID
*/
uint32_t mdl_parse_get_model_id(mdl_hdl_t mdl_hdl);

/**
query the model type id of this weight reference model

[In]: mdl_hdl - the model handle

Return: weight reference model ID
*/
int32_t mdl_parse_get_weight_ref_model_id(mdl_hdl_t mdl_hdl);

/**
query the encryption mode of this model

[In]: mdl_hdl - the model handle

Return: encryption mode
*/
mdl_encrypt_mode_t mdl_parse_get_ency_mode(mdl_hdl_t mdl_hdl);

/**
obtain encryption key of this model

[In]: mdl_hdl - the model handle
[Out]: key - point to encryption key vector flatbuffers_uint8_vec_t

Return: key length in bytes
*/
int mdl_parse_get_ency_key(mdl_hdl_t mdl_hdl, uintptr_t *key);

/**
query run level of this model

[In]: mdl_hdl - the model handle

Return: run level
*/
int32_t mdl_parse_get_run_level(mdl_hdl_t mdl_hdl);

/**
query total number of this model, includes main core(s) + sub_core(s)

[In]: mdl_hdl - the model handle

Return: total number of cores
*/
int32_t mdl_parse_get_total_cores(mdl_hdl_t mdl_hdl);

/**
query total core number of this model

[In]: mdl_hdl - the model handle

Return: total number of main cores
*/
int32_t mdl_parse_get_main_cores(mdl_hdl_t mdl_hdl);

/**
retrieve the index specified main core of this model

[In]: mdl_hdl - the model handle
[In]: idx - main core index

Return: point to main core obj
*/
core_hdl_t mdl_parse_get_main_core(mdl_hdl_t mdl_hdl, uint32_t idx);

/**
determine if a core is a sub-core inside a loop core
[In]: mdl_hdl - the model handle
[In]: core_id - sub-core id

Return: TRUE if it is a sub-core
*/
bool mdl_parse_is_sub_core(mdl_hdl_t mdl_hdl, uint32_t core_id);

/**
query the Tensor total count for input tensor in the specified model

[In]: mdl_hdl - the model handle

Return: Tensor count(0 - inputs Tensor does not existing)
*/
int mdl_parse_get_input_tensor_cnt(mdl_hdl_t mdl_hdl);

/**
query the Tensor total count for output tensor in the specified model

[In]: mdl_hdl - the model handle

Return: Tensor count(0 - outputs Tensor does not existing)
*/
int mdl_parse_get_output_tensor_cnt(mdl_hdl_t mdl_hdl);

/**
query the Tensor total count for const input tensor in the specified model

[In]: mdl_hdl - the model handle

Return: vector len(0 - const input vector does not existing)
*/
int mdl_parse_get_const_input_vec_cnt(mdl_hdl_t mdl_hdl);

/**
retrieve input tensor vector in the specified model

[In]: mdl_hdl - the model handle

Return: input tensor vector handle; NULL for none
*/
tensor_hdl_t mdl_parse_get_input_tensor_hdl(mdl_hdl_t mdl_hdl);

/**
retrieve output tensor vector

[In]: mdl_hdl - the model handle in the specified model

Return: output tensor vector handle; NULL for none
*/
tensor_hdl_t mdl_parse_get_output_tensor_hdl(mdl_hdl_t mdl_hdl);


/**
retrieve input tensor vector in the specified model

[In]: mdl_hdl - the model handle

Return: input tensor vector handle; NULL for none
*/
tensor_hdl_t mdl_parse_get_inputs_tensor_hdl(mdl_hdl_t mdl_hdl);

/**
retrieve output tensor vector

[In]: mdl_hdl - the model handle in the specified model

Return: output tensor vector handle; NULL for none
*/
tensor_hdl_t mdl_parse_get_outputs_tensor_hdl(mdl_hdl_t mdl_hdl);
/**
retrieve const input tensor vector in the specified model

[In]: mdl_hdl - the model handle

Return: const input vector address; NULL for none
*/
uintptr_t mdl_parse_get_const_input_addr(mdl_hdl_t mdl_hdl);

/**
determine if the current core is the final NPU core in this model

[In]: mdl_hdl - the model handle
[In]: core_id - the current core ID

Return: true if the current core is the last NPU core
*/
bool mdl_parse_if_last_npu_core(mdl_hdl_t mdl_hdl, int core_id);

/**
query the total elements of DmaBankInfo vector of this model.

[In]: mdl_hdl - the model handle

Return: DmaBankInfo vector len
*/
int mdl_parse_get_dma_bank_info_len(mdl_hdl_t mdl_hdl);

/**
query DmaBankInfo of this model

[In]: mdl_hdl - the model handle
[Out]: pBankInfo - point to a DmaBankInfo_t array pre-allocated

Return: void
*/
void mdl_parse_get_dma_bank_info(mdl_hdl_t mdl_hdl, DmaBankInfo_t *pBankInfo);

/**
Retrieve out the model buffer_info
The returned buffer sizes are fit and more accurate for the specified model only;
in contrast, kne_parser_get_kne_buf_info returned buffers may be bigger than actual needed.

[In]: s_hdl - session handle
[Out]: pInfo - point to the buf_info_t obj

Return: void
*/
void mdl_parser_get_model_buf_info(mdl_hdl_t mdl_hdl, buf_info_t *pInfo);

/*********************************************************************************
                     Tensor Utility APIs
**********************************************************************************/

/**
query the NPU input or output data format in DRAM for the specified tensor table

[In]: hdl - handle of the tensor table
[In]: idx - index of Tensor vector

Return: NPU data format
*/
mdl_dram_data_fmt_t mdl_tensor_get_fmt(tensor_hdl_t hdl, int idx);

/**
query the dimension of the specified tensor

[In]: hdl - handle of the tensor table
[In]: idx - index of Tensor vector

Return: dimension
*/
int32_t mdl_tensor_get_ndim(tensor_hdl_t hdl, int idx);

/**
query the channel axis of the specified tensor

[In]: hdl - handle of the tensor table
[In]: idx - index of Tensor vector

Return: channel axis
*/
int32_t mdl_tensor_get_channel_axis(tensor_hdl_t hdl, int idx);

/**
query the size of the data type

[In]: kne_data_type_enum - KNE data type

Return: data type size
*/
int32_t mdl_tensor_get_datatype_size(uint32_t kne_data_type_enum);

/**
query the valid size and aligned size in pixel of the specified tensor

[In]: hdl - handle of the tensor table
[In]: idx - index of Tensor vector
[In]: valid_size_p - valid size variable address, can be NULL for no-return
[In]: align_size_p - align size variable address, can be NULL for no-return

Return: size values will be filled to the variable passed in
*/
void mdl_tensor_get_size_in_pixel(tensor_hdl_t hdl, int idx, uint32_t *valid_size_p, uint32_t *align_size_p);

void mdl_tensor_get_start_addr_in_pixel(tensor_hdl_t hdl, int idx, uint32_t *start_addr);
/**
query the NPU output data shape (b,c,h,w) for the specified tensor table

[In]: hdl - handle of the tensor table
[In]: idx - index of Tensor vector
[Out]: point to the Shape data (mdl_shape_t)

Return: 0 - sucess; -1 - fail
*/
int mdl_tensor_get_shape(tensor_hdl_t hdl, int idx, mdl_shape_t *pShape);

/**
query the NPU output data shape in dynamic length for the specified tensor table

[In]: hdl - handle of the tensor table
[In]: idx - index of Tensor vector
[Out]: point to the Shape data
[Out]: length the Shape data

Return: 0 - sucess; -1 - fail
*/
int mdl_tensor_get_shape_dynamic(tensor_hdl_t hdl, int idx, int32_t **pShape, uint32_t *pShape_len);

/**
query the compact stride and aligned stride of the specified tensor

[In]: hdl - handle of the tensor table
[In]: idx - index of Tensor vector
[Out]: point to return value of stride_compact vector len
[Out]: stride_compact_p - point to an uint32_t array for containing return value
[Out]: point to return value of stride_aligned vector len
[Out]: stride_aligned_p - point to an uint32_t array for containing return value

Return: void
*/
void mdl_tensor_get_stride(tensor_hdl_t hdl, int idx, uint32_t *stride_aligned_len, uint32_t **stride_aligned_p);

//void mdl_tensor_get_stride(tensor_hdl_t hdl, int idx, uint32_t *stride_compact_len,
//                           uint32_t **stride_compact_p, uint32_t *stride_aligned_len, uint32_t **stride_aligned_p);

/**
query the NPU output data address for the specified tensor table

[In]: mdl_hdl - model handle
[In]: hdl - handle of the tensor table
[In]: idx - index of Tensor vector
[In]: in_seg - Input segment address
[In]: out_seg - Output segment address
[In]: wk_seg - Work segment address
[In]: cpu_temp_seg - cpu_temp segment address
[In]: wt_seg - weight segment address
[In]: const_input_seg - constant input segment address

Return: the address of the NPU data
*/
uintptr_t mdl_tensor_get_out_addr(mdl_hdl_t mdl_hdl, tensor_hdl_t hdl, int idx,
                                  uintptr_t in_seg, uintptr_t out_seg,
                                  uintptr_t wk_seg, uintptr_t cpu_temp_seq,
                                  uintptr_t wt_seg, uintptr_t const_input_seg);

/**
query quantization information (radix_count, scale_count and scale_type) for the specified tensor

[In]: hdl - handle of the tensor vector
[In]: idx - index of Tensor vector
[Out]: radix_count_p - point to the variable containing radix_count
[Out]: scale_count_p - point to the variable containing scale_count
[Out]: scale_type_p - point to the variable containning DataType

Return: the information of quantization will be filled in passed in varibles
*/
void mdl_tensor_get_quant_info(tensor_hdl_t hdl, int idx,
                               uint32_t *radix_count_p,
                               uint32_t *scale_count_p, mdl_data_type_t *scale_type_p);

/**
query the quantization radix array and scale array addresses

[In]: hdl - handle of the tensor table
[In]: t_idx - index of Tensor vector
[Out]: p_radix_p - point to the address of radix array
[Out]: p_scale_p - point to the address of scale array

Return: void
*/
void mdl_tensor_get_quant_vector(tensor_hdl_t hdl, int t_idx,
                                 int8_t **p_radix_p, uint8_t **p_scale_p);


/*********************************************************************************
                     Core Utility APIs
**********************************************************************************/

/**
query the ID for the specified core

[In]: hdl - handle of the core

Return: core ID
*/
int mdl_core_get_coreID(core_hdl_t hdl);

/**
query the core type for the specified core

[In]: hdl - handle of the core

Return: core type
*/
mdl_core_type_t mdl_core_get_coreType(core_hdl_t hdl);

/**
query the core state for the specified core

[In]: hdl - handle of the core

Return: core state
*/
mdl_core_state_t mdl_core_get_coreState(core_hdl_t hdl);

/**
get the input tensor vector len in the specified core

[In]: core_hdl - the core handle

Return: input tensor len
*/
int mdl_core_get_input_tensor_len(core_hdl_t core_hdl);

/**
retrieve input tensor vector in the specified core

[In]: core_hdl - the core handle

Return: input tensor vector handle; NULL for none
*/
tensor_hdl_t mdl_core_get_input_tensor_hdl(core_hdl_t core_hdl);

/**
get the output tensor vector len in the specified core

[In]: core_hdl - the core handle

Return: output tensor vector len
*/
int mdl_core_get_output_tensor_len(core_hdl_t core_hdl);

/**
retrieve output tensor vector in the specified core

[In]: core_hdl - the core handle

Return: output tensor vector handle; NULL for none
*/
tensor_hdl_t mdl_core_get_output_tensor_hdl(core_hdl_t core_hdl);

/**
retrieve dependencies in the specified core

[In]: core_hdl - the core handle
[Out]: dep_array_p - point to a pre-allocated int array for containing dependencies.

Return: dependency vector len
*/
int  mdl_core_get_dependencies(core_hdl_t core_hdl, int **dep_array_p);

/**
retrieve parents in the specified core

[In]: core_hdl - the core handle
[Out]: parent_array_p - point to a pre-allocated int array for containing parents.

Return: parent vector len
*/
int  mdl_core_get_parents(core_hdl_t core_hdl, int **parent_array_p);

/**
retrieve children in the specified core

[In]: core_hdl - the core handle
[Out]: child_array_p - point to a pre-allocated int array for containing children.

Return: children vector len
*/
int mdl_core_get_children(core_hdl_t core_hdl, int **child_array_p);

/**
get the NPU core parameters of the specified core

[In]: core_hdl - the core handle
[Out]: points to cmd_addr/cmd_len/wt_addr/wt_len

Return: 0 - success; -1 - fail
*/
int mdl_core_get_npu_parameters(core_hdl_t core_hdl, uintptr_t *pCmdAddr,
                                uint32_t *pCmdLen, uintptr_t *pWtAddr, uint32_t *pWtLen);

/**
get the CPU operator parameter handle of the specified core

[In]: core_hdl - the core handle

Return: CPU operator handle; NULL for fail
*/
cpu_operator_hdl_t mdl_core_get_cpu_operator_hdl(core_hdl_t core_hdl);

/**
get the LOOP handle of the specified core

[In]: core_hdl - the core handle

Return: LOOP handle for further processing; NULL for failure
*/
loop_hdl_t mdl_core_get_loop_hdl(core_hdl_t core_hdl);

/**
Inquery the core obj size

Return: core obj size
*/
int mdl_core_get_core_obj_size(void);

/*********************************************************************************
                     LOOP core Utility APIs
**********************************************************************************/

/**
inquery the interation of the LOOP core

[In]: loop_hdl - loop core handle

Return: iteration number
*/
int mdl_loop_get_iteration(loop_hdl_t loop_hdl);

/**
inquery the conditional variable address of the LOOP core

[In]: loop_hdl - loop core handle
[In]: in_seg - Input segment address
[In]: out_seg - Output segment address
[In]: wk_seg - Work segment address
[In]: cpu_temp_seg - cpu_temp segment address
[In]: wt_seg - weight segment address
[In]: const_input_seg - constant input segment address

Return: address of the conditinal variable
*/
uintptr_t mdl_loop_get_condition_var_addr(loop_hdl_t loop_hdl,
                                          uintptr_t in_seg, uintptr_t out_seg,
                                          uintptr_t wk_seg, uintptr_t cpu_temp_seq,
                                          uintptr_t wt_seg, uintptr_t const_input_seg);

/**
inquery the NPU parameters for loop core initialization

[In]: loop_hdl - loop core handle
[Out]: points to cmd_virt_addr/cmd_len/wt_virt_addr/wt_len

Return: void
*/
void mdl_loop_get_init_npu_param(loop_hdl_t loop_hdl, uintptr_t *cmd_virt_addr,
                                 uint32_t *cmd_len, uintptr_t *wt_virt_addr, uint32_t *wt_len);

/**
inquery the total number of subCores included in the specified LOOP core

[In]: loop_hdl - loop core handle

Return: total subCore number
*/
int mdl_loop_get_total_subCores(loop_hdl_t loop_hdl);

/**
retrieve the subcore in a LOOP core

[In]: loop_hdl - loop core handle
[In]: index of the subcore in the core vector
[out]: pCore: point to core obj which is pre-allocated with size from mdl_core_get_core_obj_size()
      client needs to maintain this block memory to be sure no memory leak

Return: 0 for success; -1 for failure
*/
int mdl_loop_get_subCore(loop_hdl_t loop_hdl, int idx, core_hdl_t *pCore);

/*********************************************************************************
                     CPU Utility APIs
**********************************************************************************/

/**
query the cpu operator count for the specified cpu operator table

[In]: hdl - handle of the cpu operator

Return: total count of cpu operators
*/
int mdl_cpu_get_operator_count(cpu_operator_hdl_t hdl);

/**
query the operator information for the specified cpu operator

[In]: hdl - handle of the cpu operator
[In]: n_idx - index of the cpu operator
[Out]: opcode - point to retrived operator opcode
[Out]: oper_name - point to operator name string address
[Out]: union_type_id - point to union Kneron_BuiltinOptions_union_type_t type, used to
       match out the parameter type
[Out]: union_type_name - point to the type name string address

Return: the handle to obtain parameter options
*/
param_hdl_t mdl_cpu_get_operator_info(cpu_operator_hdl_t hdl, int n_idx, uint32_t *opcode,
                                      char **oper_name, uint32_t *union_type_id, char **union_type_name);

/**
obtain the cpu operator parameters

[In]: pHdl: point to handle (return value of mdl_cpu_get_cpu_operator_info())
[In]: opcode: operator code got from call to mdl_cpu_get_cpu_operator_info()
[Out]: point to the point of operator parameter struct instance

Return: 0 - success; -1 - fail
*/
int mdl_cpu_get_params(param_hdl_t *pHdl, uint32_t opcode, void **pParam);

/**
query the length of the inputs Tensor vector for the specified cpu node

[In]: hdl - handle of the cpu operator
[In]: n_idx - index of the cpu node

Return: length of inputs tensor
*/
int mdl_cpu_get_inputs_tensor_len(cpu_operator_hdl_t hdl, int n_idx);

/**
obtain the inputs tensor vector table for the specified cpu node

[In]: hdl - handle of the cpu operator
[In]: n_idx - index of the cpu node

Return: handle to the inputs tensor table (KNE_Tensor_table_t), as
     the handle to access Tensor parameters via Tensor Utility APIs
*/
tensor_hdl_t mdl_cpu_get_inputs_tensor_hdl(cpu_operator_hdl_t hdl, int n_idx);

/**
query the length of the outputs Tensor vector for the specified cpu node

[In]: hdl - handle of the cpu operator
[In]: n_idx - index of the cpu node

Return: length of outputs tensor
*/
int mdl_cpu_get_outputs_tensor_len(cpu_operator_hdl_t hdl, int n_idx);

/**
obtain the outputs tensor vector table for the specified cpu node

[In]: hdl - handle of the cpu operator
[In]: n_idx - index of the cpu node

Return: handle to the outputs tensor table, as the handle to
        access Tensor parameters via Tensor Utility APIs
*/
tensor_hdl_t mdl_cpu_get_outputs_tensor_hdl(cpu_operator_hdl_t hdl, int n_idx);

/**
query the length of the weights vector for the specified cpu node

[In]: hdl - handle of the cpu operator
[In]: n_idx - index of the cpu node

Return: length of weight vector
*/
int mdl_cpu_get_weights_const_len(cpu_operator_hdl_t hdl, int n_idx);

/**
obtain the outputs tensor vector table for the specified cpu node

[In]: hdl - handle of the cpu operator
[In]: n_idx - index of the cpu node

Return: handle to the outputs tensor table, as the handle to
        access Tensor parameters via Tensor Utility APIs
*/
const_data_hdl_t mdl_cpu_get_weights_const_hdl(cpu_operator_hdl_t hdl, int n_idx);

/**
query the length of the input_order vector for the specified cpu node

[In]: hdl - handle of the cpu operator
[In]: n_idx - index of the cpu node

Return: length of input_order vector
*/
int mdl_cpu_get_input_order_len(cpu_operator_hdl_t hdl, int n_idx);

/**
obtain the input_order vector table for the specified cpu node

[In]: hdl - handle of the cpu operator
[In]: n_idx - index of the cpu node

Return: handle to the outputs tensor table, as the handle to
        access Tensor parameters via Tensor Utility APIs
*/
input_order_hdl_t mdl_cpu_get_input_order_hdl(cpu_operator_hdl_t hdl, int n_idx);

/*********************************************************************************
                     Const Data Utility APIs
**********************************************************************************/
/**
query the data type of the index specified const data element

[In]: hdl - handle of the const data vector
[In]: c_idx - index in consta vector

Return: data type
*/
mdl_data_type_t mdl_const_get_data_type(const_data_hdl_t hdl, int c_idx);

/**
query the Const data shape (b,c,h,w) for the specified element

[In]: hdl - handle of the const data vector
[In]: idx - index in consta vector
[Out]: point to the Shape array (int array)

Return: 0 - sucess; -1 - fail
*/
int mdl_const_get_shape(const_data_hdl_t hdl, int idx, uintptr_t *pShape);

/**
query the const data vector len and vector start address of the index specified const data element

[In]: hdl - handle of the const data vector
[In]: c_idx - index in consta vector
[Out]: pLen - point to variable for return vector len
[Out]: pVecAddr - point to the point of the variable for return vector start address

Return: void
*/
void mdl_const_get_const_values(const_data_hdl_t hdl, int c_idx, int *pLen, uint8_t **pVecAddr);

/*********************************************************************************
                    CPU Input Order Utility APIs
**********************************************************************************/

/**
query the const data vector len and vector start address of the index specified const data element

[In]: hdl - handle of the input_order vector
[In]: idx - index in input_order vector

Return: input type of the specify CPU input index
*/
int8_t mdl_input_order_get_input_type(input_order_hdl_t hdl, int idx);

/*********************************************************************************
                     NPU_CPU core Utility APIs
**********************************************************************************/

/**
inquery the NPU parameters of NPU_CPU core

[In]: core_hdl - NPU_CPU core handle
[Out]: points to cmd_phy_addr/cmd_len/wt_phy_addr/wt_len

Return: void
*/
void mdl_npu_cpu_get_npu_param(core_hdl_t core_hdl, uintptr_t *cmd_phy_addr,
               uint32_t *cmd_len, uintptr_t *wt_phy_addr, uint32_t *wt_len);

/**
inquery the total number of subCpuCores included in the specified NPU_CPU core

[In]: core_hdl - core handle

Return: total subCpuCore number
*/
int mdl_npu_cpu_get_total_subCpuCores(core_hdl_t core_hdl);

/**
retrieve the subCpuCore in a NPU_CPU core

[In]: core_hdl - core handle
[In]: index of the subcore in the core vector
[out]: pSubCpuCore: point to core obj which is pre-allocated with size from mdl_core_get_core_obj_size()
      client needs to maintain this block memory to be sure no memory leak

Return: subCore ID (>0); -1 for failure
*/
int mdl_npu_cpu_get_subCore(core_hdl_t core_hdl, int idx, core_hdl_t *pSubCpuCore);

#endif  //__MODEL_PARSER_API_H__
