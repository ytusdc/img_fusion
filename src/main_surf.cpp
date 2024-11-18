#include <iostream>  
#include <chrono>
#include "include/common.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;


int main(int argc, char *argv[])
{
    Mat image01 = imread("./images/right.jpg", 1);    //右图
    Mat image02 = imread("./images/left.jpg", 1);    //左图
    // imshow("p2", image01);
    // imshow("p1", image02);

    auto start = std::chrono::steady_clock::now();

    //灰度图转换  
    Mat image1, image2;
    cv::cvtColor(image01, image1, cv::COLOR_RGB2GRAY);
    cv::cvtColor(image02, image2, cv::COLOR_RGB2GRAY);


    // //提取特征点    
    // cv::xfeatures2d::SurfFeatureDetector Detector(2000);  
    // vector<KeyPoint> keyPoint1, keyPoint2;
    // Detector.detect(image1, keyPoint1);
    // Detector.detect(image2, keyPoint2);

    // //特征点描述，为下边的特征点匹配做准备    
    // cv::xfeatures2d::SurfDescriptorExtractor Descriptor;
    // Mat imageDesc1, imageDesc2;
    // Descriptor.compute(image1, keyPoint1, imageDesc1);
    // Descriptor.compute(image2, keyPoint2, imageDesc2);


    //提取特征点    
    Ptr<SurfFeatureDetector> Detector = SurfFeatureDetector::create();
    vector<KeyPoint> keyPoint1, keyPoint2;
    Detector->detect(image1, keyPoint1);
    Detector->detect(image2, keyPoint2);

    //特征点描述，为下边的特征点匹配做准备    
    Ptr<SurfDescriptorExtractor> Descriptor= SurfDescriptorExtractor::create();
    Mat imageDesc1, imageDesc2;
    Descriptor->compute(image1, keyPoint1, imageDesc1);
    Descriptor->compute(image2, keyPoint2, imageDesc2);



    FlannBasedMatcher matcher;
    vector<vector<DMatch> > matchePoints;
    vector<DMatch> GoodMatchePoints;

    vector<Mat> train_desc(1, imageDesc1);
    matcher.add(train_desc);
    matcher.train();

    matcher.knnMatch(imageDesc2, matchePoints, 2);
    cout << "total match points: " << matchePoints.size() << endl;

    // Lowe's algorithm,获取优秀匹配点
    for (int i = 0; i < matchePoints.size(); i++)
    {
        if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)
        {
            GoodMatchePoints.push_back(matchePoints[i][0]);
        }
    }

    // Mat first_match;
    // drawMatches(image02, keyPoint2, image01, keyPoint1, GoodMatchePoints, first_match);
    // showimg("first_match ", first_match);

    vector<Point2f> imagePoints1, imagePoints2;

    for (int i = 0; i<GoodMatchePoints.size(); i++)
    {
        imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
        imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
    }

    //获取图像1到图像2的投影映射矩阵 尺寸为3*3  
    Mat homo = findHomography(imagePoints1, imagePoints2, cv::RANSAC);
    //也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
    //Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
    cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵      

    four_corners_t corners_t;
    //计算配准图的四个顶点坐标
    CalcCorners(homo, image01, corners_t);
    cout << "left_top:" << corners_t.left_top << endl;
    cout << "left_bottom:" << corners_t.left_bottom << endl;
    cout << "right_top:" << corners_t.right_top << endl;
    cout << "right_bottom:" << corners_t.right_bottom << endl;

    //图像配准  
    Mat imageTransform1, imageTransform2;
    warpPerspective(image01, imageTransform1, homo, Size(MAX(corners_t.right_top.x, corners_t.right_bottom.x), image02.rows));
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


    OptimizeSeam(image02, imageTransform1, dst, corners_t);

    // showimg("dst", dst);
    imwrite("dst.jpg", dst);

    waitKey();

    // 记录结束时间
    auto end = std::chrono::steady_clock::now();
    // 计算耗时
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    // 输出结果
    std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}
