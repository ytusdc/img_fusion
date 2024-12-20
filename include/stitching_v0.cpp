#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <opencv2/stitching.hpp>
#include "common.hpp"

using namespace std;
using namespace cv;
namespace stitch_temp {
int stitch_offical(string file_path){

	std::vector<cv::Mat> img_vec;
	vector<string> img_names;
	glob(file_path, img_names, false);

	
	size_t num_images = img_names.size();
	img_vec.resize(num_images);

	cv::Mat img;
	for (int i = 0; i < num_images; ++i)
	{	
		img = imread(samples::findFile(img_names[i]));
		img_vec[i] = img.clone();
	}

	cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create();
    // 使用stitch函数进行拼接
    Mat pano;
    Stitcher::Status status = stitcher->stitch(img_vec, pano);
    if (status != Stitcher::OK)
    {
        cout << "Can't stitch images, error code = " << int(status) << endl;
        return -1;
    }
    imwrite("result_offical.jpg", pano);
	return 0;
}

}