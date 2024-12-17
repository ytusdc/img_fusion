#ifndef COMMON_HPP
#define COMMON_HPP

#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp> 
#include <iostream>  

using namespace cv;
using namespace std;

namespace stitch_temp {

int stitch_v1(string filepath);
int stitch_v2(string filepath);

}


#endif  // COMMON_HPP