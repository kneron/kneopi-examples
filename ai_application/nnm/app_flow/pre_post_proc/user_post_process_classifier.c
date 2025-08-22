/*
 * Kneron Example classifier Post-Processing.
 *
 * Copyright (C) 2023 Kneron, Inc. All rights reserved.
 *
 */
#include <math.h>

#include "base.h"
#include "ipc.h"
#include "kdpio.h"
#include "user_post_process_classifier.h"
#include "user_utils.h"

/******************************************************************
 * local define values
*******************************************************************/
// N/A

/******************************************************************
 * local struct defined
*******************************************************************/
// N/A

/******************************************************************
 * local variable initialization
*******************************************************************/
// N/A

/******************************************************************
 * local util function
*******************************************************************/
static int float_comparator(float a, float b) {
    float diff = a - b;

    if (diff < 0)
        return 1;
    else if (diff > 0)
        return -1;
    return 0;
}

static int comparator_callback(const void *pa, const void *pb)
{
    float a, b;

    a = ((struct ex_classifier_result_s *)pa)->score;
    b = ((struct ex_classifier_result_s *)pb)->score;

    return float_comparator(a, b);
}

/******************************************************************
 * main function
*******************************************************************/
int user_post_classifier_top_n(int model_id, struct kdp_image_s *image_p)
{
    (void)model_id;

    /* get result buffer */
    struct ex_classifier_top_n_result_s *result_p = (struct ex_classifier_top_n_result_s *)(POSTPROC_RESULT_MEM_ADDR(image_p));

    /* initialize */
    ngs_tensor_t *tensor                                            = NULL;
    ngs_quantization_parameters_v1_t *quantization_parameters_v1    = NULL;
    int32_t *shape                                                  = NULL;
    float div                                                       = 0;
    float scale                                                     = 0;
    float data                                                      = 0;
    float scaled_data                                               = 0;

    uint32_t scalar_index_list[2]                                   = {0};
    uint32_t scalar_index_list_len                                  = 2;

    float sum                                                       = 0.0;
    float max_scaled_data                                           = -INFINITY;
    double offset                                                   = 0.0;

    struct ex_classifier_result_s *top_n_results                    = NULL;

    /* get output node */
    if (0 != ex_get_output_tensor(image_p, 0, &tensor))
        goto FUNC_OUT;

    /* get quantization params */
    quantization_parameters_v1 = ex_get_tensor_quantization_parameters_v1(tensor);
    if (NULL == quantization_parameters_v1)
        goto FUNC_OUT;

    /* get shape */
    shape = ex_get_tensor_shape(tensor);
    if (NULL == shape)
        goto FUNC_OUT;

    /* calculate quantization params */
    div     = ex_pow2(quantization_parameters_v1->quantized_fixed_point_descriptor[0].radix);
    scale   = quantization_parameters_v1->quantized_fixed_point_descriptor[0].scale.scale_float32;
    scale   = 1.0f / (div * scale);

    /* setting number of category */
    result_p->top_n_num = shape[1];

    /* retrieve ouput feature map & do post process */
    /**
     * safe implementation to avoid NaN result
     * ref 1: https://stackoverflow.com/questions/9906136/implementation-of-a-softmax-activation-function-for-neural-networks
     * ref 2: https://codereview.stackexchange.com/questions/180467/implementing-softmax-in-c
     */
    top_n_results = result_p->top_n_results;

    for (int ch = 0; ch < shape[1]; ch++) {
        scalar_index_list[0]        = 0;
        scalar_index_list[1]        = ch;
        data                        = (float)*ex_get_scalar_int8(tensor, scalar_index_list, scalar_index_list_len);
        scaled_data                 = ex_do_div_scale_optim(data, scale);

        top_n_results[ch].class_num = ch;
        top_n_results[ch].score     = scaled_data;

        if (max_scaled_data < scaled_data) {
            max_scaled_data = scaled_data;
        }
    }

    for (int ch = 0; ch < shape[1]; ch++) {
        sum += expf(top_n_results[ch].score - max_scaled_data);
    }

    offset = (double)max_scaled_data + log(sum);
    for (int ch = 0; ch < shape[1]; ch++) {
        top_n_results[ch].score = expf(top_n_results[ch].score - offset);
    }

    qsort(top_n_results, result_p->top_n_num, sizeof(struct ex_classifier_result_s), comparator_callback);

FUNC_OUT:
    return sizeof(struct ex_classifier_top_n_result_s);
}
