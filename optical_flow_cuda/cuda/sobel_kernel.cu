#include "cuda_kernels.cuh"
#include <stdio.h>

// 1. Kernel thực thi trên GPU
__global__ void sobel_naive_kernel(const float* input, float* Ix, float* Iy, float* It, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // TODO: Tích chập Sobel ở Giai đoạn 2
}

// 2. Hàm Wrapper thực thi trên CPU để gọi Kernel
void launchSobelKernel(const float* d_input_frame, float* d_Ix, float* d_Iy, float* d_It, int width, int height, cudaStream_t stream) {
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("[Device] Launching Sobel Kernel voi Grid(%d, %d), Block(%d, %d)\n", 
           numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);

    // Bắn kernel xuống GPU
    sobel_naive_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_input_frame, d_Ix, d_Iy, d_It, width, height);
    
    // Bắt lỗi đồng bộ (Sync) ngay lập tức
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("[Device] LỖI CUDA: %s\n", cudaGetErrorString(err));
    }
}