#include "stitch.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>


#include <numeric>
using namespace cv;

cv::Mat Stitch_Custom::stitch(
            cv::Mat front, 
            cv::Mat right_front,
            cv::Mat right_back,
            cv::Mat back,
            cv::Mat left_back,
            cv::Mat left_front) {

    std::cout<< "test" << std::endl;
     return Mat();
}


cv::Mat Stitch_Custom::hconcat(const vector<Mat>& images) {
    // 确保所有图像的高度相同
    int height = -1;
    for (const auto& img : images) {
        if (height == -1) height = img.rows;
        else if (img.rows != height) {
            cerr << "Error: All images must have the same height." << endl;
            return Mat();
        }
    }

    // 计算总宽度
    int total_width = 0;
    for (const auto& img : images) {
        total_width += img.cols;
    }

    // 创建结果图像
    Mat result(height, total_width, images[0].type());

    // 将所有图像复制到结果图像中
    int current_x = 0;
    for (const auto& img : images) {
        img.copyTo(result(Rect(current_x, 0, img.cols, img.rows)));
        current_x += img.cols;
    }

    return result;
}

cv::Mat Stitch_Custom::stitch_hard(std::vector<ImgInfo> imginfo_vec) {
    int out_width = 0;   // 输出图片宽度
    int sum_offset = 0;  // 累计偏移量
    int max_up_offset_sum = 0;   // 最大向上累计偏移量

    int max_height = 0;  // 以第一张图片左上角x为标准，加入偏移后的图片高度（不计算高出x的位置）
    for (const auto& imginfo : imginfo_vec) {
        int crop_height = imginfo.crop_height;
        int crop_width = imginfo.crop_width;
        int up_offset = imginfo.up_offset;
        sum_offset += up_offset;
        max_up_offset_sum =  max_up_offset_sum > sum_offset ? max_up_offset_sum : sum_offset;
        int offset_height = crop_height - sum_offset;    // 减去偏移后，剩下的图片高度
        max_height = max_height > offset_height ?  max_height : offset_height;
        out_width += crop_width;
    }

    int out_height = max_up_offset_sum + max_height;  // 输出图片高度
    int current_width = 0;
    cv::Mat img_out = cv::Mat::zeros(out_height, out_width, CV_8UC3);

    std::cout<< "out_height = " << out_height << std::endl;
    std::cout<< "out_width = " << out_width << std::endl;

    int sum_offset_2 = 0;  // 累计偏移量，图片相对于第一张图的偏移量
    for (const auto& imginfo : imginfo_vec) {
        int rect_top_left_x = current_width;

        int up_offset = imginfo.up_offset;
        sum_offset_2 += up_offset;
        int rect_top_left_y = max_up_offset_sum - sum_offset_2;

        // std::cout<< "rect_top_left_x = " << rect_top_left_x << std::endl;
        // std::cout<< "rect_top_left_y = " << rect_top_left_y << std::endl;
        // std::cout<< " ************ " << std::endl;

        int w = imginfo.img_crop.cols;
        int h = imginfo.img_crop.rows;
        
        // std::cout<< "w = " << w << std::endl;
        // std::cout<< "h = " << h << std::endl;


        cv::Rect roi_rect = cv::Rect(rect_top_left_x, rect_top_left_y, imginfo.img_crop.cols, imginfo.img_crop.rows);
        imginfo.img_crop.copyTo(img_out(roi_rect));
        current_width += imginfo.crop_width;
    }

    return img_out;
    //  cv::imshow("img_out", img_out);
}

cv::Mat Stitch_Custom::rotate(Mat& image, int rotate_angle) {
	Mat dst, M;
	int w = image.cols;
	int h = image.rows;
    // 获取旋转矩阵        旋转中心   角度   缩放比例 1 
	M = getRotationMatrix2D(Point(w / 2, h / 2), rotate_angle, 1.0);
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));
	int nw = cos * w + sin * h;
	int nh = sin * w + cos * h;
	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	//warpAffine(image, dst, M, Size(w, h), INTER_LINEAR, 0, Scalar(255, 0, 0));
	warpAffine(image, dst, M, Size(nw, nh));
	// namedWindow("旋转演示", WINDOW_AUTOSIZE);
	// imshow("旋转演示", dst);
    // waitKey(0);
    imwrite("rotat.jpg", dst);
    return dst;
}

/*
图片旋转 + 旋转后裁剪
*/
bool Stitch_Custom::rotate(ImgInfo& image_info) {
	Mat dst, M;
	int w = image_info.img.cols;
	int h = image_info.img.rows;
    // 获取旋转矩阵        旋转中心   角度   缩放比例 1 
	M = getRotationMatrix2D(Point(w / 2, h / 2), image_info.rotate_angle, 1.0);
	double cos = abs(M.at<double>(0, 0));
	double sin = abs(M.at<double>(0, 1));
	int nw = cos * w + sin * h;
	int nh = sin * w + cos * h;

	M.at<double>(0, 2) += (nw / 2 - w / 2);
	M.at<double>(1, 2) += (nh / 2 - h / 2);
	//warpAffine(image, dst, M, Size(w, h), INTER_LINEAR, 0, Scalar(255, 0, 0));
	warpAffine(image_info.img, dst, M, Size(nw, nh));
	// namedWindow("旋转演示", WINDOW_AUTOSIZE);
	// imshow("旋转演示", dst);
    // waitKey(0);

    // 图片旋转后裁剪
    Rect roi(image_info.top_left.x, image_info.top_left.y, image_info.crop_width, image_info.crop_height);
    // 检查矩形是否超出图像边界
    if (roi.x + roi.width > dst.cols || roi.y + roi.height > dst.rows) {
        // throw std::invalid_argument("The ROI is out of the rotate image bounds!");
        return false;
    }

    image_info.img_crop = dst(roi).clone();
    return true;
    // imwrite("rotat.jpg", dst);
    // return dst;
}

bool Stitch_Custom::stitch_rotate(std::vector<ImgInfo> imginfo_vec, cv::Mat& img_result) {

    int count = 0;
    bool ret;
    for (auto& imginfo : imginfo_vec) {
        ret = this->rotate(imginfo);
        if (!ret) {
            std::cout<< "图片裁剪区域超出图片边界，定位到图片：" << count+1 << std::endl;
            return false;
        } else {

            // char text_name[256];  
            // sprintf(text_name, "rotate_%d.jpg", count);
            // cv::imwrite(text_name, imginfo.img_crop);
        }
        count++;
    }

    matchBrightness(imginfo_vec);

    img_result = this->stitch_hard(imginfo_vec);
    return true;
}

void Stitch_Custom::applyCLAHE(Mat& img) {
    // 将图像转换为 Lab 颜色空间
    Mat lab;
    cvtColor(img, lab, COLOR_BGR2Lab);

    // 分离通道
    vector<Mat> lab_planes(3);
    split(lab, lab_planes);

    // 创建并应用 CLAHE
    Ptr<CLAHE> clahe = createCLAHE();
    clahe->setClipLimit(4);
    clahe->apply(lab_planes[0], lab_planes[0]);

    // 合并通道并转换回 BGR 颜色空间
    merge(lab_planes, lab);
    cvtColor(lab, img, COLOR_Lab2BGR);
}


float Stitch_Custom::calculateAverageBrightness(const Mat& img) {
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Scalar mean = cv::mean(gray);
    std::cout << mean[0] << std::endl;
    return mean[0];
}

// 平均亮度调整
void Stitch_Custom::matchBrightness(std::vector<ImgInfo>& imginfo_vec) {
    // 计算每张图像的平均亮度
    vector<float> avgBrightnesses;
    for (const auto& imginfo : imginfo_vec) {
        avgBrightnesses.push_back(calculateAverageBrightness(imginfo.img_crop));
    }

    // avgBrightnesses.push_back(-1000);

    // 计算所有图像的平均亮度
    float totalAvgBrightness = accumulate(avgBrightnesses.begin(), avgBrightnesses.end(), 0.0f) / avgBrightnesses.size();

    // 根据差异调整每张图像的亮度
    for (size_t i = 0; i < imginfo_vec.size(); ++i) {
        float delta = totalAvgBrightness - avgBrightnesses[i];
        Mat adjusted;
        imginfo_vec[i].img_crop.convertTo(adjusted, -1, 1, delta);
        imginfo_vec[i].img_crop = adjusted;
    }
}
