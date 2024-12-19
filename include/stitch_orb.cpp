#include <iostream>  
#include <chrono>
#include "stitching.hpp"
#include "common.hpp"

/*
https://www.cnblogs.com/skyfsm/p/7411961.html
*/

// 左右图不要传错了，否则报错
cv::Mat stitching_orb(cv::Mat& img_left, cv::Mat& img_right)
{
    auto start = std::chrono::steady_clock::now();

    //灰度图转换  
    Mat image1, image2;
    cvtColor(img_right, image1, cv::COLOR_RGB2GRAY);
    cvtColor(img_left, image2, cv::COLOR_RGB2GRAY);

    // opencv 低版本
    // //提取特征点    
    // OrbFeatureDetector  surfDetector(3000);  
    // vector<KeyPoint> keyPoint1, keyPoint2;
    // surfDetector.detect(image1, keyPoint1);
    // surfDetector.detect(image2, keyPoint2);

    // opencv 高版本
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
        // 创建一个空的Mat对象
        cv::Mat emptyMat;
        return emptyMat;
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

    four_corners_t corners_t;

    //计算配准图的四个顶点坐标
    CalcCorners(homo, img_right, corners_t);
    cout << "left_top:" << corners_t.left_top << endl;
    cout << "left_bottom:" << corners_t.left_bottom << endl;
    cout << "right_top:" << corners_t.right_top << endl;
    cout << "right_bottom:" << corners_t.right_bottom << endl;

    //图像配准  
    Mat imageTransform1, imageTransform2;
    warpPerspective(img_right, imageTransform1, homo, Size(MAX(corners_t.right_top.x, corners_t.right_bottom.x), img_left.rows));
    //warpPerspective(img_right, imageTransform2, adjustMat*homo, Size(img_left.cols*1.3, img_left.rows*1.8));
    // showimg("直接经过透视矩阵变换", imageTransform1);

    // imwrite("trans1.jpg", imageTransform1);

    //创建拼接后的图,需提前计算图的大小
    int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
    int dst_height = img_left.rows;

    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);

    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    img_left.copyTo(dst(Rect(0, 0, img_left.cols, img_left.rows)));

    // showimg("b_dst", dst);

    OptimizeSeam(img_left, imageTransform1, dst, corners_t);
    // showimg("dst", dst);
    // imwrite("dst_111100.jpg", dst);
    // waitKey();

        // 记录结束时间
    auto end = std::chrono::steady_clock::now();
    // 计算耗时
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    // 输出结果
    std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

    return dst;
}


cv::Mat stitching_sift(cv::Mat& img_left, cv::Mat& img_right) {



    //灰度图转换  
    Mat img1, img2;
    cvtColor(img_right, img1, cv::COLOR_RGB2GRAY);
    cvtColor(img_left, img2, cv::COLOR_RGB2GRAY);

    // img1 = img_right;
    // img2 = img_left;

    // img1 = img_left;
    // img2 = img_right;

        // 初始化SIFT检测器和描述符提取器
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
    
    // cv::Ptr<cv::ORB>

    // 检测关键点和计算描述符
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;
    sift->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    // 匹配描述符
    BFMatcher matcher(NORM_L2); // SIFT uses L2 norm
    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // 使用RANSAC找到最佳单应矩阵
    vector<Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    Mat H = findHomography(points1, points2, RANSAC);


    auto start = std::chrono::steady_clock::now();
    // 使用单应矩阵进行图像拼接
    Mat result;
    warpPerspective(img2, result, H, Size(img1.cols + img2.cols, max(img1.rows, img2.rows)));
    Mat half(result, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);

            // 记录结束时间
    auto end = std::chrono::steady_clock::now();
    // 计算耗时
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    // 输出结果
    std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;


    cv::namedWindow("Stitched Image", cv::WINDOW_NORMAL);
    // 显示结果
    imshow("Stitched Image", half);
    waitKey(0);

    return result;
}