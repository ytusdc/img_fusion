#include <iostream>  
#include <chrono>
#include "stitching.hpp"
#include "common.hpp"

/*
https://www.cnblogs.com/skyfsm/p/7411961.html
*/

cv::Mat stitching_surf(cv::Mat& img_left, cv::Mat& img_right)
{
    auto start = std::chrono::steady_clock::now();
    //灰度图转换  
    Mat image1, image2;
    cv::cvtColor(img_right, image1, cv::COLOR_RGB2GRAY);
    cv::cvtColor(img_left, image2, cv::COLOR_RGB2GRAY);


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
    // drawMatches(img_left, keyPoint2, img_right, keyPoint1, GoodMatchePoints, first_match);
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
    // imwrite("dst.jpg", dst);

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