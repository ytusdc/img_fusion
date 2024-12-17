#include <chrono>
#include <iostream>

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
 
using namespace cv;
using namespace std;
using namespace cv::detail;

/*
https://blog.csdn.net/GIS_feifei/article/details/102875389
没有使用 曝光补偿器 和 接缝拼接器。

基础版本，

*/

namespace stitch_temp {

int stitch_v1(string filepath)
{
	//获取图片路径
	vector<String>image_names;  //所有图片名字
	// String filepath = "/home/ytusdc/codes_zkyc/img_fusion/img_test";   //图片存储路径
	glob(filepath, image_names, false);
	size_t num_images = image_names.size(); //图片数量
	cout << "检索到的图片为：" << endl;
	for (int i = 0; i < num_images; ++i)
	{
		cout << "Image #" <<i+1<<": "<< image_names[i]<<endl;
	}
	cout << endl;
 
	//存储图像、尺寸、特征点
	vector<ImageFeatures> features(num_images);  //存储图像特征点
	vector<Mat> images(num_images); //存储所有图像
	vector<Size> images_sizes(num_images); //存储图像的尺寸
	Ptr<Feature2D> featurefinder = cv::SIFT::create();//特征点检测方法
	for (int i = 0; i < num_images; ++i)
	{
		images[i] = cv::imread(samples::findFile(image_names[i]));//读取每一张图片
		images_sizes[i] = images[i].size();
		computeImageFeatures(featurefinder, images[i], features[i]);    //计算图像特征
		//features[i].img_idx = i;
		cout << "image #" << i + 1 << "特征点为: " << features[i].keypoints.size() << " 个"<<"  "<< "尺寸为: " << images_sizes[i] << endl;
	}
	cout << endl;
 
	//图像特征点匹配
	vector<MatchesInfo> pairwise_matches; //表示特征匹配信息变量
	Ptr<FeaturesMatcher> matcher = makePtr<BestOf2NearestMatcher>(false, 0.3f, 6, 6); //定义特征匹配器，2NN方法
	(*matcher)(features, pairwise_matches);  //进行特征匹配
 
	//预估相机参数
	Ptr<Estimator> estimator;
	estimator = makePtr<HomographyBasedEstimator>();  //水平估计
	vector<CameraParams> cameras;  //相机参数素组
	(*estimator)(features, pairwise_matches, cameras);  //得到相机参数
	cout << "预估相机参数:" << endl;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		cout << "camera #" << i + 1 << ":\n内参数矩阵K:\n" << cameras[i].K() << "\n旋转矩阵R:\n" << cameras[i].R << "\n焦距focal: " << cameras[i].focal << endl;
	}
	cout << endl;
 
	//光束平差，精确相机参数
	Ptr<detail::BundleAdjusterBase> adjuster;
	adjuster = makePtr<detail::BundleAdjusterRay>();
	(*adjuster)(features, pairwise_matches, cameras);
	cout << "精确相机参数" << endl;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		cout << "camera #" << i + 1 << ":\n内参数矩阵K:\n" << cameras[i].K() << "\n旋转矩阵R:\n" << cameras[i].R << "\n焦距focal: " << cameras[i].focal << endl;
	}
	cout << endl;
 
	//波形矫正
	vector<Mat> mat;
	for (size_t i = 0; i < cameras.size(); ++i)
		mat.push_back(cameras[i].R);
	waveCorrect(mat, WAVE_CORRECT_HORIZ); //水平校正
	for (size_t i = 0; i < cameras.size(); ++i)
		cameras[i].R = mat[i];
	cout << endl;
 
	//创建mask图像
	vector<Mat> masks(num_images);
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}
 
	//图像、掩码变换
	vector<Mat> masks_warp(num_images);  //mask扭曲
	vector<Mat> images_warp(num_images); //图像扭曲
	vector<Point> corners(num_images);   //图像左角点
	vector<Size> sizes(num_images);		 //图像尺寸
	Ptr<WarperCreator> warper_creator=makePtr<cv::CylindricalWarper>();  //柱面投影
	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(cameras[0].focal));  //因为图像焦距都一样
	for (int i = 0; i < num_images; ++i)
	{
		Mat K;
		cameras[i].K().convertTo(K, CV_32F);
		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warp[i]);  //扭曲图像images->images_warp
		sizes[i] = images_warp[i].size();
		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warp[i]);  //扭曲masks->masks_warped
	}
	for (int i = 0; i < num_images; ++i)
	{
		cout << "Image #" << i + 1 << "  corner: " << corners[i] << "  " << "size: " << sizes[i] << endl;
	}
	cout << endl;
 
	//图像融合
	Ptr<Blender> blender; //定义图像融合器
	blender = Blender::createDefault(Blender::NO, false); //简单融合方法
	blender->prepare(corners, sizes);  //生成全景图像区域
	for (int i = 0; i < num_images; ++i)
	{
		images_warp[i].convertTo(images_warp[i], CV_16S);
		blender->feed(images_warp[i], masks_warp[i], corners[i]);  //处理图像 初始化数据
	}
	Mat result, result_mask;
	blender->blend(result, result_mask);  //blend( InputOutputArray dst, InputOutputArray dst_mask  )混合并返回最后的pano。
	imwrite("result.jpg", result);
	imwrite("result_mask.jpg", result_mask);

    // Stitcher::Mode mode = Stitcher::PANORAMA;
    // Ptr<Stitcher> stitcher = Stitcher::create(mode);
    // auto status = stitcher->composePanorama(images, result);
 
	return 0;
}

}