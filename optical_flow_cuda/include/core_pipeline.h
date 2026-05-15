#ifndef CORE_PIPELINE_H
#define CORE_PIPELINE_H

#include <opencv2/opencv.hpp>

// Hàm chính điều phối toàn bộ luồng Optical Flow
// Nhận vào 2 ảnh xám (grayscale) và trả về 2 ma trận vector u, v
void processOpticalFlow(const cv::Mat& frame1, const cv::Mat& frame2, cv::Mat& flow_u, cv::Mat& flow_v);

#endif // CORE_PIPELINE_H