#include <TH/TH.h>

int scale_conv_forward(THFloatTensor *input, THFloatTensor *scale,
                        THFloatTensor *output)
{
  // if (!THFloatTensor_isSameSizeAs(input1, input2))
    // return 0;
  // THFloatTensor_resizeAs(output, input);
  // THFloatTensor_cadd(output, input1, 1.0, input2);
  return 1;
}

int scale_conv_backward(THFloatTensor *grad_output, THFloatTensor *grad_input,
                         THFloatTensor *grad_scale)
{
  // THFloatTensor_resizeAs(grad_input, grad_output);
  // THFloatTensor_fill(grad_input, 1);
  return 1;
}