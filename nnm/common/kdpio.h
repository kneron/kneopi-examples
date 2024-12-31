/*
 * Kneron Header for KDP
 *
 * Copyright (C) 2018-2023 Kneron, Inc. All rights reserved.
 *
 */

#ifndef __KDPIO_H_
#define __KDPIO_H_

#include "ncpu_gen_struct.h"

/* Internal helper macros */
#define POSTPROC_OUT_NODE(image_p)          (image_p->postproc.node_p)
#define POSTPROC_OUT_NODE_COL(image_p)      (image_p->postproc.node_p->col_length)
#define POSTPROC_OUT_NODE_ROW(image_p)      (image_p->postproc.node_p->row_length)
#define POSTPROC_OUT_NODE_CH(image_p)       (image_p->postproc.node_p->ch_length)
#define POSTPROC_OUT_NODE_RADIX(image_p)    (image_p->postproc.node_p->output_radix)
#define POSTPROC_OUT_NODE_SCALE(image_p)    (image_p->postproc.node_p->output_scale)
#if defined(KL520)
#define POSTPROC_OUT_NODE_ADDR(image_p)     (image_p->postproc.node_p->node_list[0].addr)
#else
#define POSTPROC_OUT_NODE_ADDR(image_p)     (image_p->postproc.node_p->node_list->addr)
#endif

#define OUT_NODE_COL(out_p)                 (out_p->col_length)
#define OUT_NODE_ROW(out_p)                 (out_p->row_length)
#define OUT_NODE_CH(out_p)                  (out_p->ch_length)
#define OUT_NODE_RADIX(out_p)               (out_p->output_radix)
#define OUT_NODE_SCALE(out_p)               (out_p->output_scale)
#if defined(KL520)
#define OUT_NODE_ADDR(out_p)                (out_p->node_list[0].addr)
#else
#define OUT_NODE_ADDR(out_p)                (out_p->node_list->addr)
#endif

#if defined(KL520)
#define MODEL_P(image_p)                    (image_p->model_p)
#else
#define PARSED_MODEL_P(image_p)             (image_p->pParsedModel)
#define MODEL_P(image_p)                    (&image_p->pParsedModel->oModel)
#endif
#define MODEL_ID(image_p)                   (image_p->model_id)
#define MODEL_SETUP_MEM_P(image_p)          (image_p->setup_mem_p)
#define MODEL_CMD_MEM_ADDR(image_p)         (MODEL_P(image_p)->cmd_mem_addr)
#define MODEL_CMD_MEM_LEN(image_p)          (MODEL_P(image_p)->cmd_mem_len)
#define MODEL_WEIGHT_MEM_ADDR(image_p)      (MODEL_P(image_p)->weight_mem_addr)
#define MODEL_BUF_ADDR(image_p)             (MODEL_P(image_p)->buf_addr)
#define MODEL_SETUP_MEM_ADDR(image_p)       (MODEL_P(image_p)->setup_mem_addr)
#define MODEL_INPUT_MEM_ADDR(image_p)       (MODEL_P(image_p)->input_mem_addr)
#define MODEL_INPUT_MEM_LEN(image_p)        (MODEL_P(image_p)->input_mem_len)
#define MODEL_OUTPUT_MEM_ADDR(image_p)      (MODEL_P(image_p)->output_mem_addr)
#define MODEL_OUTPUT_MEM_LEN(image_p)       (MODEL_P(image_p)->output_mem_len)

/* Return code */
#define RET_ERROR                           -1
#define RET_NO_ERROR                        0
#define RET_NEXT_PRE_PROC                   1
#define RET_NEXT_NPU                        2
#define RET_NEXT_CPU                        3
#define RET_NEXT_POST_PROC                  4

#endif
