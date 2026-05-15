#include <iostream>
#include <opencv2/opencv.hpp>
#include "core_pipeline.h"

int main() {
    std::cout << "=== KHOI DONG HE THONG OPTICAL FLOW ===" << std::endl;

    // Tạo 2 ma trận ảnh xám (Grayscale, float 32-bit) giả lập
    cv::Mat dummy_frame1 = cv::Mat::zeros(1080, 1920, CV_32FC1);
    cv::Mat dummy_frame2 = cv::Mat::zeros(1080, 1920, CV_32FC1);
    
    cv::Mat flow_u = cv::Mat::zeros(1080, 1920, CV_32FC1);
    cv::Mat flow_v = cv::Mat::zeros(1080, 1920, CV_32FC1);

    // Chạy pipeline
    processOpticalFlow(dummy_frame1, dummy_frame2, flow_u, flow_v);

    std::cout << "=== SETUP THANH CONG ===" << std::endl;
    return 0;
}