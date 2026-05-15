#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "core_pipeline.h"
#include "cuda_kernels.cuh"

// Hàm phụ trợ đọc file nhị phân
void readBin(const std::string& filename, std::vector<float>& data) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Khong tim thay file: " + filename);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
}

int main() {
    std::cout << "=== UNIT TEST: SOBEL KERNEL ===" << std::endl;
    int width = 100, height = 100;
    int N = width * height;

    // 1. Cấp phát RAM (Host)
    std::vector<float> h_frame1(N), h_frame2(N);
    std::vector<float> h_Ix_gt(N), h_Iy_gt(N), h_It_gt(N);
    std::vector<float> h_Ix_gpu(N, 0), h_Iy_gpu(N, 0), h_It_gpu(N, 0);

    // 2. Đọc Ground Truth từ Python
    try {
        readBin("data/frame1.bin", h_frame1);
        readBin("data/frame2.bin", h_frame2);
        readBin("data/Ix_gt.bin", h_Ix_gt);
        readBin("data/Iy_gt.bin", h_Iy_gt);
        readBin("data/It_gt.bin", h_It_gt);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\nChay script Python truoc nhe!" << std::endl;
        return -1;
    }

    // 3. Cấp phát VRAM (Device)
    float *d_f1, *d_f2, *d_Ix, *d_Iy, *d_It;
    cudaMalloc(&d_f1, N * sizeof(float)); cudaMalloc(&d_f2, N * sizeof(float));
    cudaMalloc(&d_Ix, N * sizeof(float)); cudaMalloc(&d_Iy, N * sizeof(float)); cudaMalloc(&d_It, N * sizeof(float));

    // 4. Copy xuống GPU và tính toán
    cudaMemcpy(d_f1, h_frame1.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f2, h_frame2.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    
    launchSobelKernel(d_f1, d_f2, d_Ix, d_Iy, d_It, width, height);
    cudaDeviceSynchronize();

    // 5. Copy kết quả về CPU
    cudaMemcpy(h_Ix_gpu.data(), d_Ix, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Iy_gpu.data(), d_Iy, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_It_gpu.data(), d_It, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 6. SO SÁNH SAI SỐ (Max Error)
    float max_err_x = 0, max_err_y = 0, max_err_t = 0;
    // Bỏ qua viền 1 pixel vì ta không tính Sobel ở viền
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int i = y * width + x;
            max_err_x = std::max(max_err_x, std::abs(h_Ix_gpu[i] - h_Ix_gt[i]));
            max_err_y = std::max(max_err_y, std::abs(h_Iy_gpu[i] - h_Iy_gt[i]));
            max_err_t = std::max(max_err_t, std::abs(h_It_gpu[i] - h_It_gt[i]));
        }
    }

    std::cout << "Max Error Ix: " << max_err_x << std::endl;
    std::cout << "Max Error Iy: " << max_err_y << std::endl;
    std::cout << "Max Error It: " << max_err_t << std::endl;

    if (max_err_x < 1e-4 && max_err_y < 1e-4 && max_err_t < 1e-4) {
        std::cout << ">>> [PASS] CUDA Sobel Kernel tinh toan CHINH XAC!" << std::endl;
    } else {
        std::cout << ">>> [FAIL] Co sai so lon. Kiem tra lai logic Kernel!" << std::endl;
    }

    // Dọn dẹp
    cudaFree(d_f1); cudaFree(d_f2); cudaFree(d_Ix); cudaFree(d_Iy); cudaFree(d_It);
    return 0;
}