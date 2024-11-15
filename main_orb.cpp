#include <iostream>  
#include <chrono>
#include "include/common.hpp"

using namespace cv;
using namespace std;

four_corners_t corners;

int main(int argc, char *argv[])
{
    // Mat image01 = imread("t1.jpg", 1);    //右图
    // Mat image02 = imread("t2.jpg", 1);    //左图

    Mat image01 = imread("/home/ytusdc/codes_zkyc/img_fusion/images/right.jpg", 1);    //右图
    Mat image02 = imread("/home/ytusdc/codes_zkyc/img_fusion/images/left.jpg", 1);    //左图

    // imshow("p2", image01);
    // imshow("p1", image02);

    auto start = std::chrono::steady_clock::now();

    //灰度图转换  
    Mat image1, image2;
    cvtColor(image01, image1, cv::COLOR_RGB2GRAY);
    cvtColor(image02, image2, cv::COLOR_RGB2GRAY);


    // //提取特征点    
    // OrbFeatureDetector  surfDetector(3000);  
    // vector<KeyPoint> keyPoint1, keyPoint2;
    // surfDetector.detect(image1, keyPoint1);
    // surfDetector.detect(image2, keyPoint2);

    // //特征点描述，为下边的特征点匹配做准备    
    // OrbDescriptorExtractor  SurfDescriptor;
    // Mat imageDesc1, imageDesc2;
    // SurfDescriptor.compute(image1, keyPoint1, imageDesc1);
    // SurfDescriptor.compute(image2, keyPoint2, imageDesc2);


    //提取特征点    
    cv::Ptr<cv::ORB> orbfDetector = cv::ORB::create();
    vector<KeyPoint> keyPoint1, keyPoint2;
    orbfDetector->detect(image1, keyPoint1);
    orbfDetector->detect(image2, keyPoint2);

    //特征点描述，为下边的特征点匹配做准备    

    Mat imageDesc1, imageDesc2;
    orbfDetector->compute(image1, keyPoint1, imageDesc1);
    orbfDetector->compute(image2, keyPoint2, imageDesc2);

    // 匹配描述符
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    std::vector<cv::DMatch> matches;
    matcher->match(imageDesc1, imageDesc2, matches);

    // 检查匹配点数量
    if (matches.size() < 4) {
        std::cout << "Not enough matches found. Need at least 4 matches." << std::endl;
        return -1;
    }

    // 提取匹配点
    std::vector<cv::Point2f> imagePoints1, imagePoints2;
    for (const auto &match : matches) {
        imagePoints1.push_back(keyPoint1[match.queryIdx].pt);
        imagePoints2.push_back(keyPoint2[match.trainIdx].pt);
    }

    //计算单应性矩阵, 获取图像1到图像2的投影映射矩阵 尺寸为3*3  
    Mat homo = findHomography(imagePoints1, imagePoints2, cv::RANSAC, 5.0);
    // 也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
    //Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
    cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵      

    //计算配准图的四个顶点坐标
    CalcCorners(homo, image01);
    cout << "left_top:" << corners.left_top << endl;
    cout << "left_bottom:" << corners.left_bottom << endl;
    cout << "right_top:" << corners.right_top << endl;
    cout << "right_bottom:" << corners.right_bottom << endl;

    //图像配准  
    Mat imageTransform1, imageTransform2;
    warpPerspective(image01, imageTransform1, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), image02.rows));
    //warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
    // showimg("直接经过透视矩阵变换", imageTransform1);

    // imwrite("trans1.jpg", imageTransform1);


    //创建拼接后的图,需提前计算图的大小
    int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
    int dst_height = image02.rows;

    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);

    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    image02.copyTo(dst(Rect(0, 0, image02.cols, image02.rows)));

    // showimg("b_dst", dst);

    OptimizeSeam(image02, imageTransform1, dst);
    // showimg("dst", dst);
    imwrite("dst_111100.jpg", dst);
    // waitKey();

        // 记录结束时间
    auto end = std::chrono::steady_clock::now();
    // 计算耗时
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    // 输出结果
    std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

    
    return 0;
}



