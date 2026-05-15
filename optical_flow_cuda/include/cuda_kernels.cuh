#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

// Đã thêm d_frame2 vào tham số
void launchSobelKernel(const float* d_frame1, const float* d_frame2, 
                       float* d_Ix, float* d_Iy, float* d_It,
                       int width, int height, 
                       cudaStream_t stream = 0);

#endif // CUDA_KERNELS_CUH