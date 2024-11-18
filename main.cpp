#include <iostream>  
#include <chrono>
#include "include/stitching.hpp"

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{

    Mat left_img = imread("/home/ytusdc/codes_zkyc/img_fusion/images/left.jpg", 1);    //左图
    Mat right_img = imread("/home/ytusdc/codes_zkyc/img_fusion/images/right.jpg", 1);    //右图

    // cv::Mat mat_dst = stitching_orb(left_img, right_img);

    cv::Mat mat_surf = stitching_orb(left_img, right_img);

    if (mat_surf.empty()) {
        std::cout << "Mat is empty." << std::endl;
        return 0;
    }

    imwrite("dst_surf.jpg", mat_surf);

}



