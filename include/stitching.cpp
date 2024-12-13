#include <iostream>  
#include <chrono>
#include "stitching.hpp"
#include "common.hpp"

cv::Mat stitching_surf(cv::Mat& img_left, cv::Mat& img_right)
{
    // Mat img_right = imread("./images/right.jpg", 1);    //右图
    // Mat img_left = imread("./images/left.jpg", 1);    //左图
    // imshow("p2", img_right);
    // imshow("p1", img_left);

    auto start = std::chrono::steady_clock::now();

    //灰度图转换  
    Mat image1, image2;
    cv::cvtColor(img_right, image1, cv::COLOR_RGB2GRAY);
    cv::cvtColor(img_left, image2, cv::COLOR_RGB2GRAY);

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

    Mat first_match;
    drawMatches(img_left, keyPoint2, img_right, keyPoint1, GoodMatchePoints, first_match);
    // showimg("first_match ", first_match);

    cv::imwrite("first_match.jpg", first_match);
    // cv::waitKey(0);

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

    imwrite("trans1.jpg", imageTransform1);


    //创建拼接后的图,需提前计算图的大小
    int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
    int dst_height = img_left.rows;

    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);

    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    img_left.copyTo(dst(Rect(0, 0, img_left.cols, img_left.rows)));

    imwrite("temp_dst.jpg", dst);

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


// 左右图不要传错了，否则额报错
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

    // //特征点描述，为下边的特征点匹配做准备    
    // OrbDescriptorExtractor  SurfDescriptor;
    // Mat imageDesc1, imageDesc2;
    // SurfDescriptor.compute(image1, keyPoint1, imageDesc1);
    // SurfDescriptor.compute(image2, keyPoint2, imageDesc2);

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
    // warpPerspective(img_right, imageTransform1, homo, Size(MAX(corners_t.right_top.x, corners_t.right_bottom.x), img_left.rows));
    warpPerspective(img_right, imageTransform1, homo, Size(img_left.cols + img_right.cols, img_left.rows));
    // warpPerspective(img_right, imageTransform2, adjustMat*homo, Size(img_left.cols*1.3, img_left.rows*1.8));
    // showimg("直接经过透视矩阵变换", imageTransform1);

    imwrite("trans1.jpg", imageTransform1);

    //创建拼接后的图,需提前计算图的大小
    int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
    int dst_height = img_left.rows;

    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);

    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    img_left.copyTo(dst(Rect(0, 0, img_left.cols, img_left.rows)));

    // // showimg("b_dst", dst);

    // OptimizeSeam(img_left, imageTransform1, dst, corners_t);
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


    cv::Mat img1 = img_left;
    cv::Mat img2 = img_right;
 
    // 初始化SIFT检测器和描述符
    cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce");
 
    // 检测关键点并计算描述符
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
 
    // 匹配描述符
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);
 
    // 对匹配结果进行排序，并保留前N个匹配项（可选）
    std::sort(matches.begin(), matches.end());
    const int N = 10;  // 保留前N个匹配
 
    // 绘制匹配结果
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Matches", img_matches);
    cv::waitKey(0);
 
    // 找到匹配对之间的最佳单应性变换
    std::vector<cv::Point2f> points1, points2;
    for (size_t i = 0; i < matches.size(); ++i) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }
    cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, 5);
 
    // 使用变换矩阵对图像进行拼接
    cv::Mat result;
    cv::warpPerspective(img1, result, H, cv::Size(img1.cols + img2.cols, img1.rows));
    cv::Mat half(result, cv::Rect(0, 0, img2.cols, img2.rows));
    img2.copyTo(half);

    return half;
}



// 左右图不要传错了，否则报错
cv::Mat stitching_orb_2(cv::Mat& img_left, cv::Mat& img_right)
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

    // //特征点描述，为下边的特征点匹配做准备    
    // OrbDescriptorExtractor  SurfDescriptor;
    // Mat imageDesc1, imageDesc2;
    // SurfDescriptor.compute(image1, keyPoint1, imageDesc1);
    // SurfDescriptor.compute(image2, keyPoint2, imageDesc2);

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


    flann::Index flannIndex(imageDesc1, flann::LshIndexParams(12, 20, 2), cvflann::FLANN_DIST_HAMMING);

    vector<DMatch> GoodMatchePoints;

    Mat macthIndex(imageDesc2.rows, 2, CV_32SC1), matchDistance(imageDesc2.rows, 2, CV_32FC1);
    flannIndex.knnSearch(imageDesc2, macthIndex, matchDistance, 2, flann::SearchParams());

    // Lowe's algorithm,获取优秀匹配点
    for (int i = 0; i < matchDistance.rows; i++)
    {
        if (matchDistance.at<float>(i, 0) < 0.4 * matchDistance.at<float>(i, 1))
        {
            DMatch dmatches(i, macthIndex.at<int>(i, 0), matchDistance.at<float>(i, 0));
            GoodMatchePoints.push_back(dmatches);
        }
    }

    Mat first_match;
    drawMatches(img_left, keyPoint2, img_right, keyPoint1, GoodMatchePoints, first_match);
    // imshow("first_match ", first_match);
    imwrite("first_match.jpg", first_match);



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
    // warpPerspective(img_right, imageTransform1, homo, Size(MAX(corners_t.right_top.x, corners_t.right_bottom.x), img_left.rows));
    warpPerspective(img_right, imageTransform1, homo, Size(img_left.cols + img_right.cols, img_left.rows));
    // warpPerspective(img_right, imageTransform2, adjustMat*homo, Size(img_left.cols*1.3, img_left.rows*1.8));
    // showimg("直接经过透视矩阵变换", imageTransform1);

    imwrite("trans1.jpg", imageTransform1);

    //创建拼接后的图,需提前计算图的大小
    int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
    int dst_height = img_left.rows;

    Mat dst(dst_height, dst_width, CV_8UC3);
    dst.setTo(0);

    imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
    img_left.copyTo(dst(Rect(0, 0, img_left.cols, img_left.rows)));

    // // showimg("b_dst", dst);

    // OptimizeSeam(img_left, imageTransform1, dst, corners_t);
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
