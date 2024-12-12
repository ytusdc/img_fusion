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
    cv::Point top_left;      // 左上角坐标
    cv::Point bottome_right; // 右下角坐标
    int up_offset;           // 针对上一张图片的位置偏移， 正数为向上偏移，负数为向下偏移
    
    ImgInfo(cv::Mat& image, 
        int top_left_x, int top_left_y, 
        int bottome_right_x, int bottome_right_y,
        int up_pixel) {
            top_left = cv::Point(top_left_x, top_left_y);
            bottome_right =  cv::Point(bottome_right_x, bottome_right_y);
            up_offset = up_pixel;
            crop_width = bottome_right_x - top_left_x;
            crop_height = bottome_right_y - top_left_y;

            Rect roi(top_left_x, top_left_y, crop_width, crop_height);
                // 检查矩形是否超出图像边界
            if (roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
                throw std::invalid_argument("The ROI is out of the image bounds!");
            }

            // img = image;
            // croppedImage, 根据矩形区域创建ROI
            img = image(roi).clone(); // 使用.clone()创建独立副本
        }
    int crop_width;
    int crop_height;
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

    public:
        int max_up_offset_sum = 0;   // 最大向上累计偏移
        int down_offset_sum = 0;
        int offset_sum = 0;
        std::vector<int> img_height_vec; 
        int max_height = 0;  // 以第一张图片左上角x为标准，加入偏移后的图片高度
        int sum_width = 0;   // 最终拼接后的图片宽度
};

#endif //STITCH_H_