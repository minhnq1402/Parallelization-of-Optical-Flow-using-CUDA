#include "cuda_kernels.cuh"
#include <stdio.h>

// Ma trận Gaussian 5x5 chuẩn (đã được làm phẳng thành mảng 1D 25 phần tử)
__constant__ float d_gaussian_kernel[25] = {
    1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256,
    4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
    6.0f/256, 24.0f/256, 36.0f/256, 24.0f/256, 6.0f/256,
    4.0f/256, 16.0f/256, 24.0f/256, 16.0f/256, 4.0f/256,
    1.0f/256,  4.0f/256,  6.0f/256,  4.0f/256, 1.0f/256
};

__global__ void pyramid_downsample_kernel(const float* input, float* output, 
                                          int in_width, int in_height, 
                                          int out_width, int out_height) {
    // Tọa độ trên ảnh đích (ảnh đã thu nhỏ)
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= out_width || out_y >= out_height) return;

    // Tọa độ tương ứng trên ảnh gốc (nhân 2)
    int in_x = out_x * 2;
    int in_y = out_y * 2;

    float pixel_value = 0.0f;
    int weight_idx = 0;

    // Quét vùng 5x5 trên ảnh gốc
    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            // Xử lý kẹp viền (Clamp to edge)
            int sample_x = min(max(in_x + dx, 0), in_width - 1);
            int sample_y = min(max(in_y + dy, 0), in_height - 1);
            
            pixel_value += input[sample_y * in_width + sample_x] * d_gaussian_kernel[weight_idx];
            weight_idx++;
        }
    }

    // Ghi vào ảnh đích
    output[out_y * out_width + out_x] = pixel_value;
}

void launchPyramidKernel(const float* d_input, float* d_output,
                         int in_width, int in_height,
                         int out_width, int out_height,
                         cudaStream_t stream) {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((out_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (out_height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    pyramid_downsample_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_input, d_output, in_width, in_height, out_width, out_height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[Device] CUDA Error in Pyramid Kernel: %s\n", cudaGetErrorString(err));
    }
}