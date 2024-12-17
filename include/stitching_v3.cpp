#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

using namespace cv;
using namespace std;
using namespace cv::detail;

#define ENABLE_LOG 1

// Default command line args
vector<string> img_names;
bool preview = false;
bool try_gpu = true;
double work_megapix = 0.6;
double seam_megapix = 0.1;
double compose_megapix = 1;
float conf_thresh = 1.f;
string features_type = "surf";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = false;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
//string warp_type = "spherical";
string warp_type = "cylindrical";
//int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
int expos_comp_type = ExposureCompensator::NO;
float match_conf = 0.3f;
//string seam_find_type = "gc_color";
string seam_find_type = "no";
int blend_type = Blender::MULTI_BAND;
float blend_strength = 3;
string result_name = "result.jpg";

/*
原文链接：https://blog.csdn.net/czl389/article/details/60769026

OpenCV自带的stitching模块在追求拼接质量方面已经做得很好了，但是实时性不够，即使是拼接两幅图像。比如源程序拼接两幅640*480分辨率的图像，拼接时间为4.78″。
对stitching_detailed.cpp源码进行改造，有利于提高实时性的举措有：
调节初始化参数。比如使用GPU，这个需要重新编译OPenCV库。
减少视频帧的分辨率。减少为320*240。
将特征提取、匹配、变换矩阵计算等步骤归为初始化部分。变换矩阵不再逐帧计算。                
*/
int stitch_v3()
{
	//打开摄像头
	VideoCapture cap1(0);
	VideoCapture cap2(1);

	double rate = 60;
	int delay = 1000 / rate;
	bool stop(false);
	Mat frame1;
	Mat frame2;
	Mat frame;
	int k = 100;

	namedWindow("cam1", CV_WINDOW_AUTOSIZE);
	namedWindow("cam2", CV_WINDOW_AUTOSIZE);
	namedWindow("stitch", CV_WINDOW_AUTOSIZE);

	if (cap1.isOpened() && cap2.isOpened())
	{
		cout << "*** ***" << endl;
		cout << "摄像头已启动！" << endl;
	}
	else
	{
		cout << "*** ***" << endl;
		cout << "警告：请检查摄像头是否安装好!" << endl;
		cout << "程序结束！" << endl << "*** ***" << endl;
		return -1;
	}

	cap1.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap1.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	cap2.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cap2.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	cap1.set(CV_CAP_PROP_FOCUS, 0);
	cap2.set(CV_CAP_PROP_FOCUS, 0);

	//获取两幅图像，通过这两幅图像来估计摄像机参数
	while (k--)
	{
		if (cap1.read(frame1) && cap2.read(frame2))
		{
			imshow("cam1", frame1);
			imshow("cam2", frame2);
			imwrite("frame1.bmp", frame1);
			imwrite("frame2.bmp", frame2);
		}
	}

	//计算相机内参数及旋转矩阵等参数
#if ENABLE_LOG
	int64 app_start_time = getTickCount();
#endif

	cv::setBreakOnError(true);

	//读入图片
	img_names.push_back("frame1.bmp");
	img_names.push_back("frame2.bmp");
	// Check if have enough images
	int num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return -1;
	}

	double work_scale = 1, seam_scale = 1, compose_scale = 0.5;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

	cout << "Finding features..." << endl;
#if ENABLE_LOG
	int64 t = getTickCount();
#endif

	Ptr<FeaturesFinder> finder;
	if (features_type == "surf")
	{
#if defined(HAVE_OPENCV_NONFREE) && defined(HAVE_OPENCV_GPU)
		if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
			finder = new SurfFeaturesFinderGpu();
		else
#endif
			finder = new SurfFeaturesFinder();
	}
	else if (features_type == "orb")
	{
		finder = new OrbFeaturesFinder();
	}
	else
	{
		cout << "Unknown 2D features type: '" << features_type << "'.\n";
		return -1;
	}

	Mat full_img, img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;

	for (int i = 0; i < num_images; ++i)
	{
		full_img = imread(img_names[i]);
		full_img_sizes[i] = full_img.size();

		if (full_img.empty())
		{
			LOGLN("Can't open image " << img_names[i]);
			return -1;
		}
		if (work_megapix < 0)
		{
			img = full_img;
			work_scale = 1;
			is_work_scale_set = true;
		}
		else
		{
			if (!is_work_scale_set)
			{
				work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
				is_work_scale_set = true;
			}
			resize(full_img, img, Size(), work_scale, work_scale);
		}
		if (!is_seam_scale_set)
		{
			seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
			seam_work_aspect = seam_scale / work_scale;
			is_seam_scale_set = true;
		}

		(*finder)(img, features[i]);
		features[i].img_idx = i;
		LOGLN("Features in image #" << i + 1 << ": " << features[i].keypoints.size());

		resize(full_img, img, Size(), seam_scale, seam_scale);
		images[i] = img.clone();
	}

	finder->collectGarbage();
	full_img.release();
	img.release();

	cout << "Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	cout << ("Pairwise matching") << endl;
#if ENABLE_LOG
	t = getTickCount();
#endif
	vector<MatchesInfo> pairwise_matches;
	BestOf2NearestMatcher matcher(try_gpu, match_conf);
	matcher(features, pairwise_matches);
	matcher.collectGarbage();
	cout << ("Pairwise matching, time: ") << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

	// Check if we should save matches graph
	if (save_graph)
	{
		LOGLN("Saving matches graph...");
		ofstream f(save_graph_to.c_str());
		f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
	}

	// Leave only images we are sure are from the same panorama
	vector<int> indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
	vector<Mat> img_subset;
	vector<string> img_names_subset;
	vector<Size> full_img_sizes_subset;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		img_names_subset.push_back(img_names[indices[i]]);
		img_subset.push_back(images[indices[i]]);
		full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
	}

	images = img_subset;
	img_names = img_names_subset;
	full_img_sizes = full_img_sizes_subset;

	// Check if we still have enough images
	num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		LOGLN("Need more images");
		return -1;
	}

	HomographyBasedEstimator estimator;
	vector<CameraParams> cameras;
	estimator(features, pairwise_matches, cameras);

	for (size_t i = 0; i < cameras.size(); ++i)
	{
		Mat R;
		cameras[i].R.convertTo(R, CV_32F);
		cameras[i].R = R;
		cout << ("Initial intrinsics #") << indices[i] + 1 << ":\n" << cameras[i].K() << endl;
	}

	Ptr<detail::BundleAdjusterBase> adjuster;
	if (ba_cost_func == "reproj") adjuster = new detail::BundleAdjusterReproj();
	else if (ba_cost_func == "ray") adjuster = new detail::BundleAdjusterRay();
	else
	{
		cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
		return -1;
	}
	adjuster->setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	if (ba_refine_mask[0] == 'x') refine_mask(0, 0) = 1;
	if (ba_refine_mask[1] == 'x') refine_mask(0, 1) = 1;
	if (ba_refine_mask[2] == 'x') refine_mask(0, 2) = 1;
	if (ba_refine_mask[3] == 'x') refine_mask(1, 1) = 1;
	if (ba_refine_mask[4] == 'x') refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);
	(*adjuster)(features, pairwise_matches, cameras);

	// Find median focal length

	vector<double> focals;
	for (size_t i = 0; i < cameras.size(); ++i)
	{
		cout << ("Camera #") << indices[i] + 1 << ":\n" << cameras[i].K() << endl;
		focals.push_back(cameras[i].focal);
	}

	sort(focals.begin(), focals.end());
	float warped_image_scale;
	if (focals.size() % 2 == 1)
		warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
	else
		warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

	if (do_wave_correct)
	{
		vector<Mat> rmats;
		for (size_t i = 0; i < cameras.size(); ++i)
			rmats.push_back(cameras[i].R.clone());
		waveCorrect(rmats, wave_correct);
		for (size_t i = 0; i < cameras.size(); ++i)
			cameras[i].R = rmats[i];
	}
	///
	cout << ("Warping images (auxiliary)... ") << endl;
#if ENABLE_LOG
	t = getTickCount();
#endif

	vector<Point> corners(num_images);
	vector<Mat> masks_warped(num_images);
	vector<Mat> images_warped(num_images);
	vector<Size> sizes(num_images);
	vector<Mat> masks(num_images);

	// Preapre images masks
	for (int i = 0; i < num_images; ++i)
	{
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	// Warp images and their masks

	Ptr<WarperCreator> warper_creator;
#if defined(HAVE_OPENCV_GPU)
	if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
	{
		if (warp_type == "plane") warper_creator = new cv::PlaneWarperGpu();
		else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarperGpu();
		else if (warp_type == "spherical") warper_creator = new cv::SphericalWarperGpu();
	}
	else
#endif
	{
		if (warp_type == "plane") warper_creator = new cv::PlaneWarper();
		else if (warp_type == "cylindrical") warper_creator = new cv::CylindricalWarper();
		else if (warp_type == "spherical") warper_creator = new cv::SphericalWarper();
		else if (warp_type == "fisheye") warper_creator = new cv::FisheyeWarper();
		else if (warp_type == "stereographic") warper_creator = new cv::StereographicWarper();
		else if (warp_type == "compressedPlaneA2B1") warper_creator = new cv::CompressedRectilinearWarper(2, 1);
		else if (warp_type == "compressedPlaneA1.5B1") warper_creator = new cv::CompressedRectilinearWarper(1.5, 1);
		else if (warp_type == "compressedPlanePortraitA2B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(2, 1);
		else if (warp_type == "compressedPlanePortraitA1.5B1") warper_creator = new cv::CompressedRectilinearPortraitWarper(1.5, 1);
		else if (warp_type == "paniniA2B1") warper_creator = new cv::PaniniWarper(2, 1);
		else if (warp_type == "paniniA1.5B1") warper_creator = new cv::PaniniWarper(1.5, 1);
		else if (warp_type == "paniniPortraitA2B1") warper_creator = new cv::PaniniPortraitWarper(2, 1);
		else if (warp_type == "paniniPortraitA1.5B1") warper_creator = new cv::PaniniPortraitWarper(1.5, 1);
		else if (warp_type == "mercator") warper_creator = new cv::MercatorWarper();
		else if (warp_type == "transverseMercator") warper_creator = new cv::TransverseMercatorWarper();
	}

	if (warper_creator.empty())
	{
		cout << "Can't create the following warper '" << warp_type << "'\n";
		return 1;
	}

	Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

	for (int i = 0; i < num_images; ++i)
	{
		Mat_<float> K;
		cameras[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;
		K(0, 0) *= swa; K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;

		corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();

		warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
	}

	vector<Mat> images_warped_f(num_images);
	for (int i = 0; i < num_images; ++i)
		images_warped[i].convertTo(images_warped_f[i], CV_32F);

	cout << "Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;
	warp end/
	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, images_warped, masks_warped);

	Ptr<SeamFinder> seam_finder;
	if (seam_find_type == "no")
		seam_finder = new detail::NoSeamFinder();
	else if (seam_find_type == "voronoi")
		seam_finder = new detail::VoronoiSeamFinder();
	else if (seam_find_type == "gc_color")
	{
#if defined(HAVE_OPENCV_GPU)
		if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
			seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR);
		else
#endif
			seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR);
	}
	else if (seam_find_type == "gc_colorgrad")
	{
#if defined(HAVE_OPENCV_GPU)
		if (try_gpu && gpu::getCudaEnabledDeviceCount() > 0)
			seam_finder = new detail::GraphCutSeamFinderGpu(GraphCutSeamFinderBase::COST_COLOR_GRAD);
		else
#endif
			seam_finder = new detail::GraphCutSeamFinder(GraphCutSeamFinderBase::COST_COLOR_GRAD);
	}
	else if (seam_find_type == "dp_color")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR);
	else if (seam_find_type == "dp_colorgrad")
		seam_finder = new detail::DpSeamFinder(DpSeamFinder::COLOR_GRAD);
	if (seam_finder.empty())
	{
		cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
		return 1;
	}

	seam_finder->find(images_warped_f, corners, masks_warped);


	// Release unused memory
	images.clear();
	images_warped.clear();
	images_warped_f.clear();
	masks.clear();
	///exposure&seam end///

	//实时拼接
	while (!stop)
	{
		if (cap1.read(frame1) && cap2.read(frame2))
		{
			imshow("cam1", frame1);
			imshow("cam2", frame2);
			imwrite("frame1.bmp", frame1);
			imwrite("frame2.bmp", frame2);

			//彩色帧转灰度
			//cvtColor(frame1, frame1, CV_RGB2GRAY);
			//cvtColor(frame2, frame2, CV_RGB2GRAY);


			//拼接过程
			//读入图片
			cout << "Compositing..." << endl;
#if ENABLE_LOG
			t = getTickCount();
#endif

			Mat img_warped, img_warped_s;
			Mat dilated_mask, seam_mask, mask, mask_warped;
			Ptr<Blender> blender;
			//double compose_seam_aspect = 1;
			double compose_work_aspect = 1;

			img_names.pop_back();
			img_names.pop_back();
			img_names.push_back("frame1.bmp");
			img_names.push_back("frame2.bmp");

			for (int img_idx = 0; img_idx < num_images; ++img_idx)
			{
				LOGLN("Compositing image #" << indices[img_idx] + 1);

				// Read image and resize it if necessary
				full_img = imread(img_names[img_idx]);/!!!!!!!!!!!!!!!!!!!!!!!!!!参数固定，可以试着读取不同图像
				if (!is_compose_scale_set)
				{
					if (compose_megapix > 0)
						compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
					is_compose_scale_set = true;

					// Compute relative scales
					//compose_seam_aspect = compose_scale / seam_scale;
					compose_work_aspect = compose_scale / work_scale;

					// Update warped image scale
					warped_image_scale *= static_cast<float>(compose_work_aspect);
					warper = warper_creator->create(warped_image_scale);

					// Update corners and sizes
					for (int i = 0; i < num_images; ++i)
					{
						// Update intrinsics
						cameras[i].focal *= compose_work_aspect;
						cameras[i].ppx *= compose_work_aspect;
						cameras[i].ppy *= compose_work_aspect;

						// Update corner and size
						Size sz = full_img_sizes[i];
						if (std::abs(compose_scale - 1) > 1e-1)
						{
							sz.width = cvRound(full_img_sizes[i].width * compose_scale);
							sz.height = cvRound(full_img_sizes[i].height * compose_scale);
						}

						Mat K;
						cameras[i].K().convertTo(K, CV_32F);
						Rect roi = warper->warpRoi(sz, K, cameras[i].R);
						corners[i] = roi.tl();
						sizes[i] = roi.size();
					}
				}
				if (abs(compose_scale - 1) > 1e-1)
					resize(full_img, img, Size(), compose_scale, compose_scale);
				else
					img = full_img;
				full_img.release();
				Size img_size = img.size();

				Mat K;
				cameras[img_idx].K().convertTo(K, CV_32F);

				// Warp the current image
				warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

				// Warp the current image mask
				mask.create(img_size, CV_8U);
				mask.setTo(Scalar::all(255));
				warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

				// Compensate exposure
				compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

				img_warped.convertTo(img_warped_s, CV_16S);
				img_warped.release();
				img.release();
				mask.release();

				dilate(masks_warped[img_idx], dilated_mask, Mat());
				resize(dilated_mask, seam_mask, mask_warped.size());
				mask_warped = seam_mask & mask_warped;

				if (blender.empty())
				{
					blender = Blender::createDefault(blend_type, try_gpu);
					Size dst_sz = resultRoi(corners, sizes).size();
					float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
					if (blend_width < 1.f)
						blender = Blender::createDefault(Blender::NO, try_gpu);
					else if (blend_type == Blender::MULTI_BAND)
					{
						MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
						mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
						cout << "Multi-band blender, number of bands: " << mb->numBands() << endl;
					}
					else if (blend_type == Blender::FEATHER)
					{
						FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
						fb->setSharpness(1.f / blend_width);
						LOGLN("Feather blender, sharpness: " << fb->sharpness());
					}
					blender->prepare(corners, sizes);
				}

				// Blend the current image
				blender->feed(img_warped_s, mask_warped, corners[img_idx]);
			}
			Mat result, result_mask;
			blender->blend(result, result_mask);

			cout << "Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec" << endl;

			result.convertTo(frame, CV_8UC1);
			imshow("stitch", frame);
		}
		else
		{
			cout << "----------------------" << endl;
			cout << "waitting..." << endl;
		}

		if (waitKey(1) == 13)
		{
			stop = true;
			cout << "程序结束！" << endl;
			cout << "*** ***" << endl;
		}
	}
	return 0;
}