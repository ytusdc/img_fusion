#include <iostream>  
#include <chrono>
#include "include/stitch.hpp"

using namespace cv;
using namespace std;



/* 参数说明
    ImgInfo imginfo(
                img,         //原始未裁剪图片
				img_crop     // 旋转+裁剪后最终图片
                top_left_x,  // 旋转后，裁剪位置的左上角 x
                top_left_y,  // 旋转后，裁剪位置的左上角 y
                bottome_right_x, // 旋转后，裁剪位置的右下角 x
                bottome_right_y, // 旋转后，裁剪位置的右下角 y
                up_offset        // 旋转+裁剪后，按顺序，相邻的两张图片中，后面的图片相对于前面图片的上下偏移像素
                                向上偏移为正数， 向下偏移为 负数
				int rotate_angle; // 旋转角度，围绕图片中心旋转，正数为向左旋转，负数为向右旋转
                );
*/

cv::Mat stitch_rotate() {

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


    
    ImgInfo imginfo_1(img_1, 10, 10, 400, 400, 0, 0);
    ImgInfo imginfo_2(img_2, 100, 100, 640, 480, 50, 0);
    ImgInfo imginfo_3(img_3, 20, 20, 340, 480, 20, 0);
    ImgInfo imginfo_4(img_4, 0, 0, 300, 200, 20, 0);
    ImgInfo imginfo_5(img_5, 0, 0, 250, 350, -50, 0);
    ImgInfo imginfo_6(img_6, 70, 70, 400, 480, -10, 0);

    std::vector<ImgInfo> imginfo_vec;

    imginfo_vec.push_back(imginfo_1);
    imginfo_vec.push_back(imginfo_2);
    imginfo_vec.push_back(imginfo_3);
    imginfo_vec.push_back(imginfo_4);
    imginfo_vec.push_back(imginfo_5);
    imginfo_vec.push_back(imginfo_6);

	auto  stitch_cls = new Stitch_Custom();
	cv::Mat result;

    bool ret = stitch_cls->stitch_rotate(imginfo_vec, result);

	if (ret) {
		imwrite("result.jpg", result);
	}else {
		std::cout<< "error, please check" << std::endl;
		std::exit(EXIT_FAILURE);
	}
    return result;
}


int main(int argc, char *argv[])
{
    cv::Mat result_mat = stitch_rotate();


    // 以下部分是裁剪拼接的图像部分，如不需要可以注释掉
    // 裁剪坐标不要超过图片边界，否则会报错
    
    // 裁剪位置的左上角坐标
    int top_left_x=0;
    int top_left_y=0;
    //裁剪位置的右下角坐标
    int bottom_right_x=0; 
    int bottom_right_y=0;

    if (top_left_x==0 && top_left_y==0 && bottom_right_x==0 && bottom_right_y==0){
        return 0;
    }

    int crop_width = bottom_right_x - top_left_x;
    int crop_height = bottom_right_y - top_left_y;

    cv::Rect roi(top_left_x, top_left_y, crop_width, crop_height);
        // 检查矩形是否超出图像边界
    if (roi.x + roi.width > result_mat.cols || roi.y + roi.height > result_mat.rows) {
        throw std::invalid_argument("The ROI is out of the image bounds!");
    }

    // img = image;
    // croppedImage, 根据矩形区域创建ROI
    cv::Mat crop_img = result_mat(roi).clone(); // 使用.clone()创建独立副本
    imwrite("result_crop.jpg", crop_img);
    return 0;
}



