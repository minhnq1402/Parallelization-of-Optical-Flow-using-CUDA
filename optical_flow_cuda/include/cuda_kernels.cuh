#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>

// Hàm wrapper để C++ gọi kernel tính Gradient (Sobel) trên GPU
void launchSobelKernel(const float* d_input_frame, 
                       float* d_Ix, float* d_Iy, float* d_It,
                       int width, int height, 
                       cudaStream_t stream = 0);

#endif // CUDA_KERNELS_CUH