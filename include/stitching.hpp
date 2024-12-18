#ifndef STITCHING_HPP
#define STITCHING_HPP


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
	通过 initStitchParam 传入图片，先将特征提取、匹配、变换矩阵计算等步骤归为初始化部分，
	计算得到相关变换参数，6张 640*480 图片，计算用时1.5s-2s 左右

	然后 beginStitch 传入跟initStitchParam，相同相机获取的图片img，可以直接根据上一步计算得到的参数,进行拼接，
	缩短拼接的用时争取达到实时性， 6张 640*480 图片，拼接用时 60ms 左右
*/

class Stitch_Custom {

public:
	explicit Stitch_Custom(){};
    ~Stitch_Custom(){};

	// return 为 0 时，方可继续
	int initStitchParam(std::vector<cv::Mat> img_path_vec);
	int beginStitch(std::vector<cv::Mat> img_vec, cv::Mat& img_stitch);

	void get_vec(string file_path, std::vector<cv::Mat>& init_img_vec, std::vector<cv::Mat>& img_vec);


public:
	// Default command line args  默认命令行参数
	// vector<String> img_names;
	// bool preview = false;
	bool try_cuda = false;
	double work_megapix = 0.6;
	double seam_megapix = 0.1;
	double compose_megapix = -1;
	float conf_thresh = 1.f;

	// #ifdef HAVE_OPENCV_XFEATURES2D
	// string features_type = "surf";
	// #else
	// string features_type = "orb";
	// #endif
	// string matcher_type = "homography";
	
	string estimator_type = "homography";
	string ba_cost_func = "ray";
	string ba_refine_mask = "xxxxx";
	bool do_wave_correct = true;
	WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
	// bool save_graph = false;
	// std::string save_graph_to;

	string warp_type = "cylindrical";
	int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
	int expos_comp_nr_feeds = 1;
	int expos_comp_nr_filtering = 2;
	int expos_comp_block_size = 32;
	// float match_conf = 0.3f;
	string seam_find_type = "gc_color";
	int blend_type = Blender::MULTI_BAND;
	int timelapse_type = Timelapser::AS_IS;
	float blend_strength = 5;
	// string result_name = "result.jpg";
	bool timelapse = false;
	// int range_width = -1;
	double seam_work_aspect = 1;


	//提出参数
	size_t num_images;
	bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
	double work_scale = 1, seam_scale = 1, compose_scale = 1;
	float warped_image_scale;

	Ptr<RotationWarper> warper;
	Ptr<WarperCreator> warper_creator;
	vector<CameraParams> cameras;  //相机参数

	vector<Size> full_img_sizes;
	vector<Point> corners;
	vector<UMat> masks_warped;
	vector<Size> sizes;
	Ptr<ExposureCompensator> compensator;


	vector<int> indices; // 可用拼接图片子集id
	size_t num_initparam_imgvec;
};

#endif  // STITCHING_HPP


