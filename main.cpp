#include <iostream>  
#include <chrono>
#include "include/stitch.hpp"

using namespace cv;
using namespace std;


int main_v2() {

    string front_path = "./images/resize_10.jpg";       // 前
    string right_front_path = "./images/resize_12.jpg"; // 右前 
    string right_back_path = "./images/resize_13.jpg";  // 右后
    string back_path = "./images/resize_14.jpg";        // 后 
    string left_back_path = "./images/resize_15.jpg";   // 左后 
    string left_front_path  = "./images/resize_16.jpg"; // 左前

    cv::Mat img_1 = cv::imread(front_path);
    cv::Mat img_2 = cv::imread(right_front_path);
    cv::Mat img_3 = cv::imread(right_back_path);
    cv::Mat img_4 = cv::imread(back_path);
    cv::Mat img_5 = cv::imread(left_back_path);
    cv::Mat img_6 = cv::imread(left_front_path);

    // ImgInfo imginfo_1(img_1, 0, 0, 640, 480, 0);
    // ImgInfo imginfo_2(img_2, 0, 0, 640, 480, 50);
    // ImgInfo imginfo_3(img_3, 0, 0, 640, 480, 50);
    // ImgInfo imginfo_4(img_4, 0, 0, 640, 480, -30);
    // ImgInfo imginfo_5(img_5, 0, 0, 640, 480, -80);
    // ImgInfo imginfo_6(img_6, 0, 0, 640, 480, 0);

    /* 参数说明
        ImgInfo imginfo(
                    img,         //原始未裁剪图片
                    top_left_x,  // 裁剪位置的左上角 x
                    top_left_y,  // 裁剪位置的左上角 y
                    bottome_right_x, // 裁剪位置的右下角 x
                    bottome_right_y, // 裁剪位置的右下角 y
                    up_offset     // 按顺序，相邻的两张图片中，后面的图片相对于前面图片的上下偏移像素
                                 向上偏移为正数， 向下偏移为 负数
                    );
    */
    
    ImgInfo imginfo_1(img_1, 10, 10, 400, 400, 0);
    ImgInfo imginfo_2(img_2, 100, 100, 640, 480, 50);
    ImgInfo imginfo_3(img_3, 20, 20, 340, 480, 20);
    ImgInfo imginfo_4(img_4, 0, 0, 300, 300, 30);
    ImgInfo imginfo_5(img_5, 0, 0, 250, 350, -50);
    ImgInfo imginfo_6(img_6, 70, 70, 400, 480, -10);

    std::vector<ImgInfo> imginfo_vec;

    imginfo_vec.push_back(imginfo_1);
    imginfo_vec.push_back(imginfo_2);
    imginfo_vec.push_back(imginfo_3);
    imginfo_vec.push_back(imginfo_4);
    imginfo_vec.push_back(imginfo_5);
    imginfo_vec.push_back(imginfo_6);

    auto  stitch_cls = new Stitch_Custom();
    cv::Mat result = stitch_cls->stitch_hard(imginfo_vec);
    imwrite("result.jpg", result);

    return 0;

}


int main(int argc, char *argv[])
{
    main_v2();
    // string front_path = "./images/resize_10.jpg";       // 前
    // string right_front_path = "./images/resize_12.jpg"; // 右前 
    // string right_back_path = "./images/resize_13.jpg";  // 右后
    // // string back_path = "./images/resize_14.jpg";        // 后 
    // // string left_back_path = "./images/resize_15.jpg";   // 左后 
    // // string left_front_path  = "./images/resize_16.jpg"; // 左前

    // std::vector<cv::Mat> images_vec;     

    // images_vec.push_back(cv::imread(front_path));
    // images_vec.push_back(cv::imread(right_front_path));
    // images_vec.push_back(cv::imread(right_back_path));
    // // images_vec.push_back(cv::imread(back_path));
    // // images_vec.push_back(cv::imread(left_back_path));
    // // images_vec.push_back(cv::imread(left_front_path));

    // // images_vec 可以是任意数量的图片，因为是水平拼接，要保证图片的高度一致

    // auto  stitch_cls = new Stitch_Custom();
    // cv::Mat result = stitch_cls->hconcat(images_vec);
    // imwrite("result.jpg", result);

}



