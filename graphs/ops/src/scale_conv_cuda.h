int scale_conv_forward_cuda(THCudaTensor *input,
                             THCudaTensor *weight, /*THCudaTensor * bias, */
                             THCudaTensor *scale, THCudaTensor *output,
                             THCudaTensor *columns, THCudaTensor *ones, int kW,
                             int kH, int dW, int dH, int padW, int padH,
                             int dilationW, int dilationH,
                             int deformable_group, int im2col_step);

int scale_conv_backward_input_cuda(
    THCudaTensor *input, THCudaTensor *scale, THCudaTensor *gradOutput,
    THCudaTensor *gradInput, THCudaTensor *gradScale, THCudaTensor *weight,
    THCudaTensor *columns, int kW, int kH, int dW, int dH, int padW, int padH,
    int dilationW, int dilationH, int deformable_group, int im2col_step);

int scale_conv_backward_parameters_cuda(
    THCudaTensor *input, THCudaTensor *scale, THCudaTensor *gradOutput,
    THCudaTensor *gradWeight, /*THCudaTensor *gradBias, */
    THCudaTensor *columns, THCudaTensor *ones, int kW, int kH, int dW, int dH,
    int padW, int padH, int dilationW, int dilationH, int deformable_group,
    float scale_, int im2col_step);
