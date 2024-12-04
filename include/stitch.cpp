#include "stitch.hpp"

using namespace cv;

cv::Mat Stitch_Custom::stitch(
            cv::Mat front, 
            cv::Mat right_front,
            cv::Mat right_back,
            cv::Mat back,
            cv::Mat left_back,
            cv::Mat left_front) {

    std::cout<< "test" << std::endl;
     return Mat();
}


cv::Mat Stitch_Custom::hconcat(const vector<Mat>& images) {
    // 确保所有图像的高度相同
    int height = -1;
    for (const auto& img : images) {
        if (height == -1) height = img.rows;
        else if (img.rows != height) {
            cerr << "Error: All images must have the same height." << endl;
            return Mat();
        }
    }

    // 计算总宽度
    int total_width = 0;
    for (const auto& img : images) {
        total_width += img.cols;
    }

    // 创建结果图像
    Mat result(height, total_width, images[0].type());

    // 将所有图像复制到结果图像中
    int current_x = 0;
    for (const auto& img : images) {
        img.copyTo(result(Rect(current_x, 0, img.cols, img.rows)));
        current_x += img.cols;
    }

    return result;
}


