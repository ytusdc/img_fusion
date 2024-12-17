#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

// #include "include/common.hpp"
#include "include/stitching.hpp"

using namespace std;
using namespace cv;
// using namespace stitch_temp;
using namespace stitch_custom;

int main(int argc, char* argv[])
{	
	// string file_path = "./img_test/*";

	// stitch_v1(file_path);
	// stitch_temp::stitch_v2(file_path);

	String path_10 = "./img_test/resize_10.jpg";
	String path_11 = "./img_test/resize_11.jpg";
	String path_12 = "./img_test/resize_12.jpg";
	String path_13 = "./img_test/resize_13.jpg";
	String path_14 = "./img_test/resize_14.jpg";
	String path_15 = "./img_test/resize_15.jpg";
	String path_16 = "./img_test/resize_16.jpg";
	String path_17 = "./img_test/resize_17.jpg";
	String path_18 = "./img_test/resize_18.jpg";

	cv::Mat img_10 = cv::imread(path_10);
	cv::Mat img_11 = cv::imread(path_11);
	cv::Mat img_12 = cv::imread(path_12);
	cv::Mat img_13 = cv::imread(path_13);
	cv::Mat img_14 = cv::imread(path_14);
	cv::Mat img_15 = cv::imread(path_15);
	cv::Mat img_16 = cv::imread(path_16);
	cv::Mat img_17 = cv::imread(path_17);
	cv::Mat img_18 = cv::imread(path_18);


	std::vector<String> path_vec;
	std::vector<cv::Mat> img_vec;

	// path_vec.push_back(path_10);
	// img_vec.push_back(img_10);

	// path_vec.push_back(path_11);
	// img_vec.push_back(img_11);

	// path_vec.push_back(path_12);
	// img_vec.push_back(img_12);

	path_vec.push_back(path_13);
	img_vec.push_back(img_13);

	path_vec.push_back(path_14);
	img_vec.push_back(img_14);

	path_vec.push_back(path_15);
	// img_vec.push_back(img_15);

	path_vec.push_back(path_16);
	// img_vec.push_back(img_16);

	// path_vec.push_back(path_17);
	img_vec.push_back(img_17);

	// path_vec.push_back(path_18);
	img_vec.push_back(img_18);


	auto stitch_custom = new Stitch_Custom();

	auto start_init = std::chrono::high_resolution_clock::now();

	stitch_custom->initStitchParam(path_vec);


	auto end_init = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed_init = end_init - start_init;
	// 输出结果
	std::cout << "init程序耗时: " << elapsed_init.count() << " ms" << std::endl;



	std::cout<< "******************" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();

	stitch_custom->beginStitch(img_vec);

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed_stitch = end - start;
	// 输出结果
	std::cout << "stitch程序耗时: " << elapsed_stitch.count() << " ms" << std::endl;

	return 0;
}