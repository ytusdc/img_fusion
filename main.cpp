#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>

#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"

#include "include/common.hpp"
#include "include/stitching.hpp"

using namespace std;
using namespace cv;
using namespace stitch_temp;

int main(int argc, char* argv[])
{	
	// string file_path = "./images/3/*.jpg";


    // // string file_path = "./img_test";
	// stitch_temp::stitch_v2(file_path);

    // return 0;

	string path_10 = "./img_test/resize_10.jpg";
	string path_11 = "./img_test/resize_11.jpg";
	string path_12 = "./img_test/resize_12.jpg";
	string path_13 = "./img_test/resize_13.jpg";
	string path_14 = "./img_test/resize_14.jpg";
	string path_15 = "./img_test/resize_15.jpg";
	string path_16 = "./img_test/resize_16.jpg";
	string path_17 = "./img_test/resize_17.jpg";
	string path_18 = "./img_test/resize_18.jpg";

	cv::Mat img_10 = cv::imread(path_10);
	cv::Mat img_11 = cv::imread(path_11);
	cv::Mat img_12 = cv::imread(path_12);
	cv::Mat img_13 = cv::imread(path_13);
	cv::Mat img_14 = cv::imread(path_14);
	cv::Mat img_15 = cv::imread(path_15);
	cv::Mat img_16 = cv::imread(path_16);
	cv::Mat img_17 = cv::imread(path_17);
	cv::Mat img_18 = cv::imread(path_18);

	std::vector<cv::Mat> init_img_vec;
	std::vector<cv::Mat> img_vec;

	// img_vec.push_back(img_10);
	// init_img_vec.push_back(img_10);

	// img_vec.push_back(img_11);
	// init_img_vec.push_back(img_11);

	// img_vec.push_back(img_12);
	// init_img_vec.push_back(img_12);

	img_vec.push_back(img_13);
	init_img_vec.push_back(img_13);

    img_vec.push_back(img_14);
	init_img_vec.push_back(img_14);

    img_vec.push_back(img_15);
	init_img_vec.push_back(img_15);

    img_vec.push_back(img_16);
	init_img_vec.push_back(img_16);

    img_vec.push_back(img_17);
	init_img_vec.push_back(img_17);

    img_vec.push_back(img_18);
	init_img_vec.push_back(img_18);


    string path = "./img_test/*.jpg";


    auto stitch_custom = new Stitch_Custom();


    // stitch_custom->get_vec(path, init_img_vec, img_vec);


	auto start_init = std::chrono::high_resolution_clock::now();

	int ret;
	ret = stitch_custom->initStitchParam(img_vec);

	if(ret != 0) {
		std::cout<< "无法拼接, 出现错误, 请根据log检查" << std::endl;
		return -1;
	}


	auto end_init = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed_init = end_init - start_init;
	// 输出结果
	std::cout << "init程序耗时: " << elapsed_init.count() << " ms" << std::endl;

	std::cout<< "******************" << std::endl;

	int count = 1000;

	cv::Mat result_stitch;

	for (int i=0; i <= count; i++) {
		auto start = std::chrono::high_resolution_clock::now();

		ret = stitch_custom->beginStitch(init_img_vec, result_stitch);
		
		if(ret != 0) {
			std::cout<< "拼接报错, 请根据log检查" << std::endl;
			return 0;
		}

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_stitch = end - start;
		// 输出结果
		std::cout << "stitch程序耗时: " << elapsed_stitch.count() << " ms" << std::endl;

		char text_name[256];  
        // sprintf(text_name, "result_%d.jpg", i);
           sprintf(text_name, "result_0.jpg", i);
		cv::imwrite(text_name, result_stitch);

	}

	return 0;
}