int scale_conv_forward(THFloatTensor *input, THFloatTensor *scale,
                        THFloatTensor *output);
int scale_conv_backward(THFloatTensor *grad_output, THFloatTensor *grad_input,
                         THFloatTensor *grad_scale);