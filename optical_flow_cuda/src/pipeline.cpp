#include "core_pipeline.h"
#include "cuda_kernels.cuh"
#include <iostream>

void processOpticalFlow(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& flow_u, cv::Mat& flow_v) {
    int width = frame1.cols;
    int height = frame1.rows;
    
    std::cout << "[Host] Bat dau xu ly Optical Flow cho frame: " 
              << width << "x" << height << std::endl;

    // TODO Giai doan 3: cudaMalloc, cudaMemcpy tai day

    // Gọi thử hàm Wrapper của CUDA (Truyền nullptr để test Linker trước)
    // CẢNH BÁO: Chạy thật với nullptr sẽ gây Segmentation Fault. 
    // Giai đoạn này chỉ in log để check luồng.
    std::cout << "[Host] Dang goi CUDA Kernel..." << std::endl;
    
    // Giả lập gọi kernel với size 10x10
    launchSobelKernel(nullptr, nullptr, nullptr, nullptr, 10, 10);

    std::cout << "[Host] Hoan thanh luong Optical Flow." << std::endl;
}