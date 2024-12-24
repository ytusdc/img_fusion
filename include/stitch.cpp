#include "stitch.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>


#include <numeric>
using namespace cv;

/*
    blur_width: 拼缝处模糊宽度（像素），其中一边的宽度
    weight_edge: 模糊宽度的边缘（远离拼缝处）的黑色像素权重，取值（0-1），对应百分比0-100%
    weight_middle: 模糊宽度的中间（拼缝处）的黑色像素权重，取值（0-1）， 对应百分比0-100%
*/
Stitch_Custom::Stitch_Custom(int blur_width, double weight_edge, double weight_middle) {
    if (blur_width == 0) {
        m_is_blur = false;
    } else {
        if (weight_edge < 0 | weight_middle < 0) {
            std::cout<< "概率值不能小于0" << std::endl;
            m_is_blur = false;
        }
        else if (weight_edge < m_epsilon && weight_middle < m_epsilon ) {
            //概率值都为0则不进行模糊化
            m_is_blur = false;
        } 
        else {
            m_is_blur = true;
            m_begin_weight = weight_edge;
            m_end_weight = weight_middle;
        }
    }

    m_blur_width = blur_width;
    
}

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

    // vector<int> join_line_vec;
    m_join_line_vec.clear();
    int count = 0;
    bool ret;

    int join_line_accum = 0;
    for (int i=0; i<imginfo_vec.size(); i++) {
        if (i != (imginfo_vec.size()-1)) {
            int value = imginfo_vec[i].bottom_right.x - imginfo_vec[i].top_left.x;
            join_line_accum += value;
            // std::cout<< join_line_accum << std::endl;
            m_join_line_vec.push_back(join_line_accum);
        }
    }
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
    cv::Mat img_src = this->stitch_hard(imginfo_vec);
    img_result = img_src.clone();
    this->OptimizeSeam(img_src, img_result);
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



//优化两图的连接处，使得拼接自然
void Stitch_Custom::OptimizeSeam(Mat& img1, Mat& dst)
{   
    if (!m_is_blur) {
        return;
    }

    int rows = dst.rows;
    int cols = img1.cols; //注意，是列数*通道数
    // double alpha = 1;//img1中像素的权重

    float interval_value = (this->m_end_weight - this->m_begin_weight) / this->m_blur_width; 

    for (auto join_line: m_join_line_vec) 
    {
        for (int i = 0; i < rows; i++)
        {
            uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
            // uchar* t = trans.ptr<uchar>(i);
            uchar* d = dst.ptr<uchar>(i);
            for (int j = 0; j <= this->m_blur_width; j++)
            {   
                double weight = m_begin_weight + interval_value * (this->m_blur_width - j);  // 黑色像素的权重

                // weight = 0.3;
                int col_left = join_line - j;
                int col_right = join_line + j;

                if (j == 0) {
                    d[col_left * 3] = p[col_left * 3] * (1 - weight) + 0 * weight;
                    d[col_left * 3 + 1] = p[col_left * 3 + 1] * (1 - weight) + 0 * weight;
                    d[col_left * 3 + 2] = p[col_left * 3 + 2] * (1 - weight) + 0 * weight;
                } else {
                    d[col_left * 3] = p[col_left * 3] * (1 - weight) + 0 * weight;
                    d[col_left * 3 + 1] = p[col_left * 3 + 1] * (1 - weight) + 0 * weight;
                    d[col_left * 3 + 2] = p[col_left * 3 + 2] * (1 - weight) + 0 * weight;

                    d[col_right * 3] = p[col_right * 3] * (1 - weight) + 0 * weight;
                    d[col_right * 3 + 1] = p[col_right * 3 + 1] * (1 - weight) + 0 * weight;
                    d[col_right * 3 + 2] = p[col_right * 3 + 2] * (1 - weight) + 0 * weight;
                }
            }
        }
    }
}


bool Stitch_Custom::crop_image(cv::Point top_left, cv::Point bottom_right, cv::Mat& src_img, cv::Mat& result_img) {

    // int crop_width = bottom_right.x - top_left.x;
    // int crop_height = bottom_right.y - top_left.y;

    cv::Rect roi_rect(top_left, bottom_right);
        // 检查矩形是否超出图像边界
    if (roi_rect.x + roi_rect.width > src_img.cols || roi_rect.y + roi_rect.height > src_img.rows) {
        // throw std::invalid_argument("The ROI is out of the image bounds!");
        std::cout<< "The ROI is out of the image bounds!" << std::endl;
        return false;
    }

    // img = image;
    // croppedImage, 根据矩形区域创建ROI
    result_img = src_img(roi_rect).clone(); // 使用.clone()创建独立副本
    return true;
}