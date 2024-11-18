#include <iostream>  
#include <chrono>
#include "common.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

cv::Mat stitching_surf(cv::Mat& img_left, cv::Mat& img_right);
cv::Mat stitching_orb(cv::Mat& img_left, cv::Mat& img_right);
