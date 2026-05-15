#include "cuda_kernels.cuh"
#include <stdio.h>

__global__ void sobel_naive_kernel(const float* frame1, const float* frame2, 
                                   float* Ix, float* Iy, float* It, 
                                   int width, int height) {
    // 1. Lấy tọa độ pixel 2D mà thread này đảm nhận
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // 2. Bỏ qua các pixel ở viền (vì Sobel 3x3 cần đọc lân cận)
    if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1) {
        // Gán bằng 0 cho an toàn nếu nó là viền
        if (x < width && y < height) {
            int idx = y * width + x;
            Ix[idx] = 0.0f;
            Iy[idx] = 0.0f;
            It[idx] = 0.0f;
        }
        return;
    }

    int idx = y * width + x;

    // 3. Đọc vùng 3x3 lân cận từ Global Memory (Rất chậm, nhưng dễ debug)
    float p00 = frame1[(y - 1) * width + (x - 1)]; float p01 = frame1[(y - 1) * width + x]; float p02 = frame1[(y - 1) * width + (x + 1)];
    float p10 = frame1[y * width + (x - 1)];       float p11 = frame1[y * width + x];       float p12 = frame1[y * width + (x + 1)];
    float p20 = frame1[(y + 1) * width + (x - 1)]; float p21 = frame1[(y + 1) * width + x]; float p22 = frame1[(y + 1) * width + (x + 1)];

    // 4. Tích chập tính Ix (Sobel X: Nhấn mạnh viền dọc)
    float val_Ix = -1.0f * p00 + 1.0f * p02 
                   -2.0f * p10 + 2.0f * p12 
                   -1.0f * p20 + 1.0f * p22;

    // 5. Tích chập tính Iy (Sobel Y: Nhấn mạnh viền ngang)
    float val_Iy = -1.0f * p00 - 2.0f * p01 - 1.0f * p02 
                   +1.0f * p20 + 2.0f * p21 + 1.0f * p22;

    // 6. Tính It (Đạo hàm thời gian giữa 2 frame)
    float val_It = frame2[idx] - frame1[idx];

    // 7. Ghi kết quả lại ra Global Memory
    Ix[idx] = val_Ix;
    Iy[idx] = val_Iy;
    It[idx] = val_It;
}

void launchSobelKernel(const float* d_frame1, const float* d_frame2, 
                       float* d_Ix, float* d_Iy, float* d_It, 
                       int width, int height, cudaStream_t stream) {
    
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    sobel_naive_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(d_frame1, d_frame2, d_Ix, d_Iy, d_It, width, height);
    
    // Bắt lỗi ngầm
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[Device] CUDA Error in Sobel Kernel: %s\n", cudaGetErrorString(err));
    }
}