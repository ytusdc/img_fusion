#include "stitch.hpp"

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

    // std::cout<< "out_height = " << out_height << std::endl;
    // std::cout<< "out_width = " << out_width << std::endl;

    int sum_offset_2 = 0;  // 累计偏移量，图片相对于第一张图的偏移量
    for (const auto& imginfo : imginfo_vec) {
        int rect_top_left_x = current_width;

        int up_offset = imginfo.up_offset;
        sum_offset_2 += up_offset;
        int rect_top_left_y = max_up_offset_sum - sum_offset_2;

        // std::cout<< "rect_top_left_x = " << rect_top_left_x << std::endl;
        // std::cout<< "rect_top_left_y = " << rect_top_left_y << std::endl;
        // std::cout<< " ************ " << std::endl;

        cv::Rect roi_rect = cv::Rect(rect_top_left_x, rect_top_left_y, imginfo.img.cols, imginfo.img.rows);
        imginfo.img.copyTo(img_out(roi_rect));
        current_width += imginfo.crop_width;
    }

    return img_out;
    //  cv::imshow("img_out", img_out);
}
