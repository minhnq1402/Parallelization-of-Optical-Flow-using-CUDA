#include "core_pipeline.h"
#include "cuda_kernels.cuh"
#include <iostream>

void processOpticalFlow(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& flow_u, cv::Mat& flow_v) {
    int width = frame1.cols;
    int height = frame1.rows;
    size_t size_bytes = width * height * sizeof(float);
    
    // Kích thước của ảnh ở Level 1 (thu nhỏ một nửa)
    int width_L1 = width / 2;
    int height_L1 = height / 2;
    size_t size_bytes_L1 = width_L1 * height_L1 * sizeof(float);

    std::cout << "[Host] Khoi dong Pipeline (" << width << "x" << height << ")" << std::endl;

    float *d_f1, *d_f2, *d_f1_L1, *d_f2_L1;
    // ... (Khai báo các biến d_Ix, d_Iy, d_It, d_u, d_v tương tự như bản cũ)
    float *d_Ix, *d_Iy, *d_It, *d_u, *d_v;

    // Cấp phát VRAM
    cudaMalloc(&d_f1, size_bytes); cudaMalloc(&d_f2, size_bytes);
    cudaMalloc(&d_f1_L1, size_bytes_L1); cudaMalloc(&d_f2_L1, size_bytes_L1); // VRAM cho ảnh thu nhỏ
    cudaMalloc(&d_Ix, size_bytes_L1); cudaMalloc(&d_Iy, size_bytes_L1); cudaMalloc(&d_It, size_bytes_L1);
    cudaMalloc(&d_u, size_bytes_L1); cudaMalloc(&d_v, size_bytes_L1);

    // Upload
    cudaMemcpy(d_f1, frame1.ptr<float>(), size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f2, frame2.ptr<float>(), size_bytes, cudaMemcpyHostToDevice);

    std::cout << "[Host] -> Xay dung Pyramid (Downsample 50%)..." << std::endl;
    launchPyramidKernel(d_f1, d_f1_L1, width, height, width_L1, height_L1, 0);
    launchPyramidKernel(d_f2, d_f2_L1, width, height, width_L1, height_L1, 0);

    // Chạy Sobel và LK trên ảnh ĐÃ THU NHỎ (Level 1)
    std::cout << "[Host] -> Chay Sobel tren Level 1..." << std::endl;
    launchSobelKernel(d_f1_L1, d_f2_L1, d_Ix, d_Iy, d_It, width_L1, height_L1, 0);

    std::cout << "[Host] -> Chay Lucas-Kanade tren Level 1..." << std::endl;
    launchLucasKanadeKernel(d_Ix, d_Iy, d_It, d_u, d_v, width_L1, height_L1, 0);

    cudaDeviceSynchronize();

    // Dọn dẹp
    cudaFree(d_f1); cudaFree(d_f2); cudaFree(d_f1_L1); cudaFree(d_f2_L1);
    cudaFree(d_Ix); cudaFree(d_Iy); cudaFree(d_It); cudaFree(d_u); cudaFree(d_v);

    std::cout << "[Host] Hoan thanh toan bo pipeline Naive." << std::endl;
}