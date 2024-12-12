#include <iostream>  
#include <chrono>
#include "include/stitch.hpp"

using namespace cv;
using namespace std;


int main(int argc, char *argv[])
{
    
    string front_path = "./images/resize_10.jpg";       // 前
    string right_front_path = "./images/resize_12.jpg"; // 右前 
    string right_back_path = "./images/resize_13.jpg";  // 右后
    // string back_path = "./images/resize_14.jpg";        // 后 
    // string left_back_path = "./images/resize_15.jpg";   // 左后 
    // string left_front_path  = "./images/resize_16.jpg"; // 左前

    auto  stitch_cls = new Stitch_Custom();

    std::vector<cv::Mat> images_vec;     

    images_vec.push_back(cv::imread(front_path));
    images_vec.push_back(cv::imread(right_front_path));
    images_vec.push_back(cv::imread(right_back_path));
    // images_vec.push_back(cv::imread(back_path));
    // images_vec.push_back(cv::imread(left_back_path));
    // images_vec.push_back(cv::imread(left_front_path));

    // images_vec 可以是任意数量的图片，因为是水平拼接，要保证图片的高度一致
    cv::Mat result = stitch_cls->hconcat(images_vec);
    imwrite("result.jpg", result);

}



