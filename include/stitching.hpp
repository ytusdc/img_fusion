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


namespace stitch_custom {
// #define ENABLE_LOG 1
// // Default command line args  默认命令行参数
// vector<String> img_names;
// bool preview = false;
// bool try_cuda = false;
// double work_megapix = 0.6;
// double seam_megapix = 0.1;
// double compose_megapix = -1;
// float conf_thresh = 1.f;
// #ifdef HAVE_OPENCV_XFEATURES2D
// string features_type = "surf";
// #else
// string features_type = "orb";
// #endif
// string matcher_type = "homography";
// string estimator_type = "homography";
// string ba_cost_func = "ray";
// string ba_refine_mask = "xxxxx";
// bool do_wave_correct = true;
// WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
// bool save_graph = false;
// std::string save_graph_to;
// string warp_type = "cylindrical";
// int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
// int expos_comp_nr_feeds = 1;
// int expos_comp_nr_filtering = 2;
// int expos_comp_block_size = 32;
// float match_conf = 0.3f;
// string seam_find_type = "gc_color";
// int blend_type = Blender::MULTI_BAND;
// int timelapse_type = Timelapser::AS_IS;
// float blend_strength = 5;
// string result_name = "result.jpg";
// bool timelapse = false;
// int range_width = -1;
// double seam_work_aspect = 1;


class Stitch_Custom {

public:

	explicit Stitch_Custom(){};
    ~Stitch_Custom(){};

	bool initStitchParam(std::vector<String> img_path_vec);

	bool beginStitch(std::vector<cv::Mat> img_vec);

public:
	// Default command line args  默认命令行参数
	vector<String> img_names;
	bool preview = false;
	bool try_cuda = false;
	double work_megapix = 0.6;
	double seam_megapix = 0.1;
	double compose_megapix = -1;
	float conf_thresh = 1.f;
	#ifdef HAVE_OPENCV_XFEATURES2D
	string features_type = "surf";
	#else
	string features_type = "orb";
	#endif
	string matcher_type = "homography";
	string estimator_type = "homography";
	string ba_cost_func = "ray";
	string ba_refine_mask = "xxxxx";
	bool do_wave_correct = true;
	WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
	bool save_graph = false;
	std::string save_graph_to;
	string warp_type = "cylindrical";
	int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
	int expos_comp_nr_feeds = 1;
	int expos_comp_nr_filtering = 2;
	int expos_comp_block_size = 32;
	float match_conf = 0.3f;
	string seam_find_type = "gc_color";
	int blend_type = Blender::MULTI_BAND;
	int timelapse_type = Timelapser::AS_IS;
	float blend_strength = 5;
	string result_name = "result.jpg";
	bool timelapse = false;
	int range_width = -1;
	double seam_work_aspect = 1;


};
}

#endif  // STITCHING_HPP


