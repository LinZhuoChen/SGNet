#ifndef SCALE_CONV_CUDA_KERNEL_H_
#define SCALE_CONV_CUDA_KERNEL_H_
void scale_im2col(cudaStream_t stream, const float *data_im,
                       const float *data_scale, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs,
                       const int deformable_group, float *data_col);

void scale_col2im(cudaStream_t stream, const float *data_col,
                       const float *data_scale, const int channels,
                       const int height, const int width, const int ksize_h,
                       const int ksize_w, const int pad_h, const int pad_w,
                       const int stride_h, const int stride_w,
                       const int dilation_h, const int dilation_w,
                       const int parallel_imgs,
                       const int deformable_group, float *grad_im);

void scale_col2im_coord(cudaStream_t stream, const float *data_col,
                             const float *data_im, const float *data_scale,
                             const int channels, const int height,
                             const int width, const int ksize_h,
                             const int ksize_w, const int pad_h,
                             const int pad_w, const int stride_h,
                             const int stride_w, const int dilation_h,
                             const int dilation_w, const int parallel_imgs,
                             const int deformable_group, float *grad_scale);
#endif