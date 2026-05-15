#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

void launchSobelKernel(const float* d_frame1, const float* d_frame2, 
                       float* d_Ix, float* d_Iy, float* d_It,
                       int width, int height, 
                       cudaStream_t stream = 0);

void launchLucasKanadeKernel(const float* d_Ix, const float* d_Iy, const float* d_It,
                             float* d_u, float* d_v,
                             int width, int height, 
                             cudaStream_t stream = 0);

// THÊM MỚI: Hàm thu nhỏ ảnh và làm mờ (Downsample + Gaussian Blur)
void launchPyramidKernel(const float* d_input, float* d_output,
                         int in_width, int in_height,
                         int out_width, int out_height,
                         cudaStream_t stream = 0);

#endif // CUDA_KERNELS_CUH