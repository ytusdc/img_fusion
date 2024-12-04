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
    
};

#endif //STITCH_H_