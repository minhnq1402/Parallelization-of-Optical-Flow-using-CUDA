#include "cuda_kernels.cuh"
#include <stdio.h>

__global__ void lucas_kanade_naive_kernel(const float* Ix, const float* Iy, const float* It,
                                          float* u, float* v, 
                                          int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    // Bán kính cửa sổ 21x21 -> radius = 10
    int radius = 10;
    
    // Các biến lưu tổng của ma trận AtA và Atb
    float sum_Ixx = 0.0f, sum_Iyy = 0.0f, sum_Ixy = 0.0f;
    float sum_Ixt = 0.0f, sum_Iyt = 0.0f;

    // Quét qua cửa sổ 21x21 xung quanh pixel hiện tại
    for (int wy = -radius; wy <= radius; ++wy) {
        for (int wx = -radius; wx <= radius; ++wx) {
            // Xử lý viền bằng cách kẹp tọa độ (clamp to edge)
            int nx = min(max(x + wx, 0), width - 1);
            int ny = min(max(y + wy, 0), height - 1);
            
            int idx = ny * width + nx;

            float ix = Ix[idx];
            float iy = Iy[idx];
            float it = It[idx];

            sum_Ixx += ix * ix;
            sum_Iyy += iy * iy;
            sum_Ixy += ix * iy;
            sum_Ixt += ix * it;
            sum_Iyt += iy * it;
        }
    }

    // Giải hệ phương trình 2x2 bằng quy tắc Cramer
    // [ sum_Ixx  sum_Ixy ] [ u ] = [ -sum_Ixt ]
    // [ sum_Ixy  sum_Iyy ] [ v ] = [ -sum_Iyt ]
    
    float det = sum_Ixx * sum_Iyy - sum_Ixy * sum_Ixy;
    int out_idx = y * width + x;

    // Kiểm tra định thức: Nếu vùng ảnh quá trơn (det rất nhỏ), không thể nghịch đảo
    if (det > 1e-5f) {
        u[out_idx] = (-sum_Iyy * sum_Ixt + sum_Ixy * sum_Iyt) / det;
        v[out_idx] = ( sum_Ixy * sum_Ixt - sum_Ixx * sum_Iyt) / det;
    } else {
        u[out_idx] = 0.0f;
        v[out_idx] = 0.0f;
    }
}

void launchLucasKanadeKernel(const float* d_Ix, const float* d_Iy, const float* d_It,
                             float* d_u, float* d_v,
                             int width, int height, cudaStream_t stream) {
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    lucas_kanade_naive_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_Ix, d_Iy, d_It, d_u, d_v, width, height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[Device] CUDA Error in LK Kernel: %s\n", cudaGetErrorString(err));
    }
}