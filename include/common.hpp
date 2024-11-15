#ifndef COMMON_HPP
#define COMMON_HPP

#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp> 
#include <iostream>  

using namespace cv;
using namespace std;

typedef struct
{
    Point2f left_top;
    Point2f left_bottom;
    Point2f right_top;
    Point2f right_bottom;
}four_corners_t;

void CalcCorners(const Mat& H, const Mat& src, four_corners_t& corners);
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst, four_corners_t& corners);
void showimg(string name, cv::Mat img);

#endif  // COMMON_HPP