#ifndef STITCH_H_
#define STITCH_H_

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp> 

using namespace std;
using namespace cv;


struct ImgInfo {
    cv::Mat img;
    cv::Mat img_crop;        // 旋转 + 裁剪后最终 的图片
    cv::Point top_left;      // 旋转后，裁剪图片的左上角坐标
    cv::Point bottom_right;  // 旋转后，裁剪图片的右下角坐标
    int up_offset;           // 针对上一张裁剪图片的位置偏移， 正数为向上偏移，负数为向下偏移
    int rotate_angle;               // 旋转角度，围绕图片中心旋转， 正数为向左旋转，负数为向右旋转
    int crop_width;
    int crop_height;
    
    ImgInfo(cv::Mat& image, 
        int top_left_x, 
        int top_left_y, 
        int bottom_right_x, 
        int bottom_right_y,
        int up_pixel,
        int angle ) {
            top_left = cv::Point(top_left_x, top_left_y);
            bottom_right =  cv::Point(bottom_right_x, bottom_right_y);
            up_offset = up_pixel;
            rotate_angle = angle;
            crop_width = bottom_right_x - top_left_x;
            crop_height = bottom_right_y - top_left_y;
            img = image;

            // Rect roi(top_left_x, top_left_y, crop_width, crop_height);

            // // 检查矩形是否超出图像边界
            // if (roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
            //     throw std::invalid_argument("The ROI is out of the image bounds!");
            // }

            // img = image;
            // croppedImage, 根据矩形区域创建ROI
            // img = image(roi).clone(); // 使用.clone()创建独立副本
        }
};

class Stitch_Custom {
    public:
        explicit Stitch_Custom(){};
        ~Stitch_Custom(){};
    public:
        cv::Mat stitch(cv::Mat front, 
                cv::Mat right_front,
                cv::Mat right_back,
                cv::Mat back,
                cv::Mat left_back,
                cv::Mat left_front);
        
        cv::Mat hconcat(const vector<Mat>& images);

        cv::Mat stitch_hard(std::vector<ImgInfo> imginfo_vec);
        cv::Mat rotate(Mat& image, int rotate_angle);
        bool rotate(ImgInfo& image_info);

        bool stitch_rotate(std::vector<ImgInfo> imginfo_vec, cv::Mat& img_result);

        void applyCLAHE(Mat& img);
        void matchBrightness(std::vector<ImgInfo>& imginfo_vec);
        float calculateAverageBrightness(const Mat& img);

    public:
 
        int down_offset_sum = 0;
        int offset_sum = 0;
        int max_height = 0;  // 以第一张图片左上角x为标准，加入偏移后的图片高度
        int sum_width = 0;   // 最终拼接后的图片宽度
};

#endif //STITCH_H_