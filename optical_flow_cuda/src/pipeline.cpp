#include "core_pipeline.h"
#include "cuda_kernels.cuh"
#include <iostream>

void processOpticalFlow(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& flow_u, cv::Mat& flow_v) {
    int width = frame1.cols;
    int height = frame1.rows;
    size_t size_bytes = width * height * sizeof(float);
    
    std::cout << "[Host] Cap phat VRAM va tinh Sobel Gradient..." << std::endl;

    // Khai báo con trỏ trên GPU
    float *d_frame1, *d_frame2, *d_Ix, *d_Iy, *d_It;

    // Cấp phát bộ nhớ
    cudaMalloc(&d_frame1, size_bytes);
    cudaMalloc(&d_frame2, size_bytes);
    cudaMalloc(&d_Ix, size_bytes);
    cudaMalloc(&d_Iy, size_bytes);
    cudaMalloc(&d_It, size_bytes);

    // Copy dữ liệu từ RAM xuống VRAM (Host to Device)
    cudaMemcpy(d_frame1, frame1.ptr<float>(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_frame2, frame2.ptr<float>(), size_bytes, cudaMemcpyHostToDevice);

    // Bắn lệnh chạy Kernel
    launchSobelKernel(d_frame1, d_frame2, d_Ix, d_Iy, d_It, width, height, 0);

    // Chờ GPU tính xong
    cudaDeviceSynchronize();

    // DỌN DẸP BỘ NHỚ (KHÔNG ĐƯỢC QUÊN)
    cudaFree(d_frame1);
    cudaFree(d_frame2);
    cudaFree(d_Ix);
    cudaFree(d_Iy);
    cudaFree(d_It);

    std::cout << "[Host] Hoan thanh tinh Gradient." << std::endl;
}