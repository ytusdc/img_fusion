#include <chrono>
#include <iostream>
#include <string>
#include <vector>
// #include <opencv2/opencv.hpp>
// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui.hpp>
// #include <opencv2/stitching.hpp>
// #include <opencv2/features2d/features2d.hpp>
// #include <opencv2/xfeatures2d/nonfree.hpp>

#include <opencv2/opencv.hpp> 
#include <opencv2/highgui/highgui.hpp>   
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp> 

#include "include/stitching.hpp"


using namespace cv;
using namespace std;


// void main_v2(string img_path_1, string img_path_2) {

void main_v2(cv::Mat img1, cv::Mat img2) {


	// Mat img1 = imread("img/img1.png", IMREAD_COLOR);
	// Mat img2 = imread("img/img2.png", IMREAD_COLOR);

    // Mat left_img = imread("/home/ytusdc/codes_zkyc/img_fusion/images/left.jpg", 1);    //左图
    // Mat right_img = imread("/home/ytusdc/codes_zkyc/img_fusion/images/right.jpg", 1);    //右图

    // cv::Mat img1 = cv::imread("./images/resize_12.jpg", IMREAD_COLOR);
    // cv::Mat img2 = cv::imread("./images/resize_13.jpg", IMREAD_COLOR);

    // cv::Mat img1 = cv::imread(img_path_1, IMREAD_COLOR);
    // cv::Mat img2 = cv::imread(img_path_2, IMREAD_COLOR);

    std::vector<Mat> imgs;

    //step 2. sift feature detect
	printf("extract sift features \n");
	std::vector<KeyPoint> keyPoint1, keyPoint2;
	Ptr<Feature2D> siftFeature = cv::SIFT::create(); //The number of best features to retain
 
	siftFeature->detect(img1, keyPoint1);
	siftFeature->detect(img2, keyPoint2);
 
	Mat descor1, descor2;
	siftFeature->compute(img1, keyPoint1, descor1);
	siftFeature->compute(img2, keyPoint2, descor2);
 
    Mat feature_img1, feature_img2;
	drawKeypoints(img1, keyPoint1, feature_img1, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(img2, keyPoint2, feature_img2, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
 
	// imshow("img1", feature_img1);
	// imshow("img2", feature_img2);
    // waitKey(0);


        //step 3. instantiate mathcher
	FlannBasedMatcher matcher;
	std::vector<DMatch> matches;
	matcher.match(descor1, descor2, matches);
	// printff("original match numbers: " + std::to_string(matches.size()));
 
	Mat oriMatchRes;
	drawMatches(img1, keyPoint1, img2, keyPoint2, matches, oriMatchRes, Scalar(0, 255, 0), Scalar::all(-1));
	// imshow("orign match img", oriMatchRes);



        //step 4. select better match
	double sum = 0;
	double maxDist = 0;
	double minDist = 0;
	for (auto &match : matches)
	{
		double dist = match.distance;
		maxDist = max(maxDist, dist);
		minDist = min(minDist, dist);
	}
	// printff("max distance: " + std::to_string(maxDist));
	// printff("min distance: " + std::to_string(minDist));
 
	std::vector<DMatch> goodMatches;
	double threshold = 0.5;
	for (auto &match : matches)
	{
		if (match.distance < threshold * maxDist)
			goodMatches.emplace_back(match);
	}


    // step 5. RANSAC delete mismatched feature points
    //step 5.1 align feature points and convet to float
	std::vector<KeyPoint> R_keypoint01, R_keypoint02;
	for (auto &match : goodMatches)
	{
		R_keypoint01.emplace_back(keyPoint1[match.queryIdx]);
		R_keypoint02.emplace_back(keyPoint2[match.trainIdx]);
	}
	std::vector<Point2f> p01, p02;
	for (int i = 0; i < goodMatches.size(); ++i)
	{
		p01.emplace_back(R_keypoint01[i].pt);
		p02.emplace_back(R_keypoint02[i].pt);
	}
 
	//step 5.2 compute homography
	std::vector<uchar> RansacStatus;
	Mat fundamental = findHomography(p01, p02, RansacStatus, cv::RANSAC);
	Mat dst;
	warpPerspective(img1, dst, fundamental, Size(img1.cols, img1.rows));
	imwrite("epipolarimage.jpg", dst);


 
	//step 5.3  delete mismatched points
	std::vector<KeyPoint> RR_keypoint01, RR_keypoint02;
	std::vector<DMatch> RR_matches;
	int idx = 0;
	for (int i = 0; i < goodMatches.size(); ++i)
	{
		if (RansacStatus[i] != 0)
		{
			RR_keypoint01.emplace_back(R_keypoint01[i]);
			RR_keypoint02.emplace_back(R_keypoint02[i]);
			goodMatches[i].queryIdx = idx;
			goodMatches[i].trainIdx = idx;
			RR_matches.emplace_back(goodMatches[i]);
			++idx;
		}
	}
	// printff("refine match pairs : " + std::to_string(RR_matches.size()));
	Mat imgRRMatches;
	drawMatches(img1, RR_keypoint01, img2, RR_keypoint02, RR_matches, imgRRMatches, Scalar(0, 255, 0), Scalar::all(-1));
	// imshow("final match", imgRRMatches);
    	imwrite("final_match", imgRRMatches);

    //step 6. stitch
	Mat finalImg = dst.clone();
	img2.copyTo(finalImg(Rect(0, 0, img2.cols, img2.rows)));
	// imshow("stitching image", finalImg);

    imwrite("stitchingimage.jpg", finalImg);

    waitKey(0);
    return;
}


int main_3()
{

    cv::Mat img_10 = cv::imread("./images/resize_10.jpg");
    cv::Mat img_11 = cv::imread("./images/resize_11.jpg");
    cv::Mat img_12 = cv::imread("./images/resize_12.jpg");
    cv::Mat img_13 = cv::imread("./images/resize_13.jpg");
    cv::Mat img_14 = cv::imread("./images/resize_14.jpg");
    cv::Mat img_15 = cv::imread("./images/resize_15.jpg");
    cv::Mat img_16 = cv::imread("./images/resize_16.jpg");

    Mat left_img = imread("/home/ytusdc/codes_zkyc/img_fusion/images/left.jpg", 1);    //左图
    Mat right_img = imread("/home/ytusdc/codes_zkyc/img_fusion/images/right.jpg", 1);    //右图

    // Step 1: Load images
    Mat img1 = left_img;
    Mat img2 = right_img;

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Could not load one or both images." << std::endl;
        return -1;
    }

    // Step 2: Detect keypoints and compute descriptors
    Ptr<SIFT> sift = SIFT::create();
    std::vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors1, descriptors2;

    sift->detectAndCompute(img1, noArray(), keypoints1, descriptors1);
    sift->detectAndCompute(img2, noArray(), keypoints2, descriptors2);

    // Step 3: Match features
    BFMatcher matcher(NORM_L2);
    std::vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // Sort matches by score
    std::sort(matches.begin(), matches.end());
    // Remove not so good matches
    const float ratio = 0.75f; // Keep top 75% matches
    const int numGoodMatches = matches.size() * ratio;
    matches.erase(matches.begin() + numGoodMatches, matches.end());

    // Draw the best matches
    Mat img_matches;
    drawMatches(img1, keypoints1, img2, keypoints2, matches, img_matches);
    imwrite("BestMatches.jpg", img_matches);
    waitKey(0);

    // Step 4: Estimate homography matrix
    std::vector<Point2f> points1, points2;

    for (size_t i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    Mat H = findHomography(points1, points2, RANSAC);

    // Step 5: Warp and stitch images
    Mat result;
    warpPerspective(img2, result, H, Size(img1.cols + img2.cols, max(img1.rows, img2.rows)));
    imwrite("wrap.jpg", result);


    Mat half(result, Rect(0, 0, img1.cols, img1.rows));
    img1.copyTo(half);

    // Display the stitched image
    imwrite("Stitched_Image.jpg", result);
    waitKey(0);

    return 0;
}


int main(int argc, char *argv[])
{   

    cv::Mat img_10 = cv::imread("./images/resize_10.jpg");
    cv::Mat img_11 = cv::imread("./images/resize_11.jpg");
    cv::Mat img_12 = cv::imread("./images/resize_12.jpg");
    cv::Mat img_13 = cv::imread("./images/resize_13.jpg");
    cv::Mat img_14 = cv::imread("./images/resize_14.jpg");
    cv::Mat img_15 = cv::imread("./images/resize_15.jpg");
    cv::Mat img_16 = cv::imread("./images/resize_16.jpg");
    cv::Mat img_17 = cv::imread("./images/resize_17.jpg");
    cv::Mat img_18 = cv::imread("./images/resize_18.jpg");

    // const std::string    img_path_1{argv[1]};
    // const std::string    img_path_2{argv[2]};  // 可以使图片/图片文件夹/视频文件

    // main_3();
    // return 0;

    Mat left_img = imread("/home/ytusdc/codes_zkyc/img_fusion/images/left.jpg", 1);    //左图
    Mat right_img = imread("/home/ytusdc/codes_zkyc/img_fusion/images/right.jpg", 1);    //右图

    Mat mat_10_11 = imread("./10-11.jpg", 1); 


    vector<Mat> images;
    images.push_back(img_10);
    images.push_back(img_11);
    images.push_back(img_12);
    images.push_back(img_13);
    images.push_back(img_14);
    images.push_back(img_15);
    images.push_back(img_16);
    images.push_back(img_17);
    images.push_back(img_18);

    Mat result;

    Mat result1, result2, result3;

    auto start = std::chrono::steady_clock::now();

    Stitcher::Mode mode = Stitcher::PANORAMA;
    // Stitcher::Mode mode = Stitcher::SCANS;

    Ptr<Stitcher> stitcher = Stitcher::create(mode);
    auto status = stitcher->stitch(images, result);

    
    // 记录结束时间
    auto end = std::chrono::steady_clock::now();
    // 计算耗时
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    // 输出结果
    std::cout << "Time taken by function: " << duration.count() << " milliseconds" << std::endl;

    //   // 拼接方式-多通道融合
    // auto blender = detail::Blender::createDefault(detail::Blender::MULTI_BAND, true);
    // stitcher->setBlender(blender);

    // // 平面曲翘拼接
    // auto plane_warper = makePtr<cv::PlaneWarper>();
    // stitcher->setWarper(plane_warper);
    // status = stitcher->stitch(images, result2);
 
    // // 鱼眼拼接
    // auto fisheye_warper = makePtr<cv::FisheyeWarper>();
    // stitcher->setWarper(fisheye_warper);
    // status = stitcher->stitch(images, result3);


    std::cout << "status = " << status << std::endl;

    if (status != Stitcher::OK) {
        cerr << "Error: Can't stitch images, error code = " << int(status) << endl;
        return -1;
    }

    imwrite("dst_stitch.jpg", result);

    // imwrite("result1.png", result);
    // imwrite("result2.png", result2);
    // imwrite("result3.png", result3);


    // cv::Mat mat_surf = stitching_surf(img_18, mat_10_11);

    // // cv::Mat mat_dst = stitching_orb(left_img, right_img);

    // // cv::Mat mat_surf = stitching_orb(left_img, right_img);

    // //   cv::Mat mat_surf = stitching_sift(left_img, right_img);



    // if (mat_surf.empty()) {
    //     std::cout << "Mat is empty." << std::endl;
    //     return 0;
    // }

    // imwrite("dst_surf.jpg", mat_surf);
    return 0;

}



