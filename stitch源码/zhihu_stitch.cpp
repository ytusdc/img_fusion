
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

// 2.1.1 特征检测和图像匹配，得到可用于拼接的图像
Stitcher::Status Stitcher::matchImages()
{
    // 若输入图像小于两张，则无法拼接
    if ((int)imgs_.size() < 2)
    {
        return ERR_NEED_MORE_IMGS;
    }
​
    // 初始化参数
    work_scale_ = 1;                        /* 控制特征检测时图像的缩放比例（0~1）*/
    seam_scale_ = 1;                        /* 控制图像拼接时图像的缩放比例（0~1） */
    seam_work_aspect_ = 1;                  /* 图像变形参数 */
    bool is_work_scale_set = false;         /* 标记：work_scale_是否更新 */
    bool is_seam_scale_set = false;         /* 标记：seam_scale_是否更新 */
    features_.resize(imgs_.size());         /* 图像特征点信息（图像序号，图像大小，关键点等） */
    seam_est_imgs_.resize(imgs_.size());    /* 指定图像拼接时，用于检测拼接边界的图像 */
    full_img_sizes_.resize(imgs_.size());   /* 原图像尺寸*/
​
    // 特征点检测图像
    std::vector<UMat> feature_find_imgs(imgs_.size());
    std::vector<UMat> feature_find_masks(masks_.size());
​
    /* 
        (1)通过registr_resol_参数改变特征检测图像feature_find_imgs 和 mask图像feature_find_masks
        (2)通过seam_est_resol_参数改变接缝检测图像seam_est_imgs_
        (2)确定seam_work_aspect_的值
    */
    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        full_img_sizes_[i] = imgs_[i].size();
        if (registr_resol_ < 0)
        {
            feature_find_imgs[i] = imgs_[i];
            work_scale_ = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale_ = std::min(1.0, std::sqrt(registr_resol_ * 1e6 / full_img_sizes_[i].area()));
                is_work_scale_set = true;
            }
            resize(imgs_[i], feature_find_imgs[i], Size(), work_scale_, work_scale_, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale_ = std::min(1.0, std::sqrt(seam_est_resol_ * 1e6 / full_img_sizes_[i].area()));
            seam_work_aspect_ = seam_scale_ / work_scale_;
            is_seam_scale_set = true;
        }
​
        if (!masks_.empty())
        {
            resize(masks_[i], feature_find_masks[i], Size(), work_scale_, work_scale_, INTER_NEAREST);
        }
        features_[i].img_idx = (int)i;
        LOGLN("Features in image #" << i+1 << ": " << features_[i].keypoints.size());
​
        resize(imgs_[i], seam_est_imgs_[i], Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
    }
​
​
    /*
        2.1.1.1 特征点检测与描述
        输入：特征点检测模型features_finder_，默认为 ORB 特征点检测方法与描述
        输入：特征点检测图像feature_find_imgs，mask图像feature_find_masks
        输出：每幅图的特征点信息features_（:vector<ImageFeatures>）
    */
    detail::computeImageFeatures(features_finder_, feature_find_imgs, features_, feature_find_masks);
​
    // 清空feature_find_imgs和feature_find_masks以节约内存，不再需要
    feature_find_imgs.clear();
    feature_find_masks.clear();
​
    /*
        2.1.1.2 通过特征点信息得到图像之间的匹配关系 pairwise_matches_ 
        输入：特征点features_， 匹配mask图像matching_mask_
        输出：匹配关系pairwise_matches_
        当拼接模型为：PANORAMA 时，匹配方法为：BestOf2NearestMatcher 一种基于特征点的暴力匹配策略
        当拼接模型为：SCANS 时，匹配方法为：AffineBestOf2NearestMatcher 基于最近邻的仿射变换算法，一种改进的RANSAC方法
    */
    (*features_matcher_)(features_, pairwise_matches_, matching_mask_);
    // 释放垃圾，从而提高程序性能。
    features_matcher_->collectGarbage();
​
    /*
        2.1.1.3 筛选用于图像拼接的图像，保留确定来自同一全景图的图像
        输入：特征点features_， 匹配关系 pairwise_matches_ ，阈值conf_thresh_
        输出：需要保留的图像序号indices_
    */
    indices_ = detail::leaveBiggestComponent(features_, pairwise_matches_, (float)conf_thresh_);
    // 根据筛选结果保留图像
    std::vector<UMat> seam_est_imgs_subset;
    std::vector<UMat> imgs_subset;
    std::vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices_.size(); ++i)
    {
        imgs_subset.push_back(imgs_[indices_[i]]);
        seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
        full_img_sizes_subset.push_back(full_img_sizes_[indices_[i]]);
    }
    seam_est_imgs_ = seam_est_imgs_subset;
    imgs_ = imgs_subset;
    full_img_sizes_ = full_img_sizes_subset;
​
    // 若保留的图像小于2张，则无法拼接
    if ((int)imgs_.size() < 2)
    {
        return ERR_NEED_MORE_IMGS;
    }
​
    return OK;
}




// 2.1.2 估计图像拼接中相机参数, 并进行波形校正
Stitcher::Status Stitcher::estimateCameraParams()
{
    /*
        2.1.2.1 估计全局框架中的变换参数 cameras_
        输入：特征点信息features_，特征点匹配关系pairwise_matches_
        输出：相机参数cameras_
        当拼接模型为：PANORAMA 时，方法为：HomographyBasedEstimator ，估计相机位姿
        当拼接模型为：SCANS 时，方法为：AffineBasedEstimator，使用仿射变换来估计相机位姿
    */
    if (!(*estimator_)(features_, pairwise_matches_, cameras_))
        return ERR_HOMOGRAPHY_EST_FAIL;
    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        // R：存储相机外参，即变换矩阵
        Mat R;
        cameras_[i].R.convertTo(R, CV_32F);
        cameras_[i].R = R;
    }
​
    /*
        2.1.2.2 优化相机参数
        输入：特征点信息features_，特征点匹配关系pairwise_matches_
        输出：相机参数cameras_
        当拼接模型为：PANORAMA 时，方法为：BundleAdjusterRay ，使用基于空间射线（ray）的Bundle Adjustment（BA）算法
        当拼接模型为：SCANS 时，方法为：BundleAdjusterAffinePartial，估计部分仿射变换参数，并优化3D点的位置使其最终投影结果与观测结果最为接近
    */
    bundle_adjuster_->setConfThresh(conf_thresh_);
    if (!(*bundle_adjuster_)(features_, pairwise_matches_, cameras_))
        return ERR_CAMERA_PARAMS_ADJUST_FAIL;
​
    // 找到中值焦距并将其用作最终图像比例
    std::vector<double> focals;
    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        focals.push_back(cameras_[i].focal);
    }
    std::sort(focals.begin(), focals.end());
    if (focals.size() % 2 == 1)
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
​
    /*
        2.1.2.3 波形校正，使组成的图像更加平滑
        输入：相机外参R，变换矩阵，优化策略wave_correct_kind_
        输出：局部畸变的校正后的变换矩阵R
        当拼接模型为：PANORAMA 时，优化策略为：WAVE_CORRECT_HORIZ
        当拼接模型为：SCANS 时，do_wave_correct_为false，不进行优化
    */
    if (do_wave_correct_)
    {
        std::vector<Mat> rmats;
        for (size_t i = 0; i < cameras_.size(); ++i)
            rmats.push_back(cameras_[i].R.clone());
        detail::waveCorrect(rmats, wave_correct_kind_);
        for (size_t i = 0; i < cameras_.size(); ++i)
            cameras_[i].R = rmats[i];
    }
​
    return OK;
}



// 2.2 得到拼接结果
Stitcher::Status Stitcher::composePanorama(OutputArray pano)
{
    return composePanorama(std::vector<UMat>(), pano);
}
Stitcher::Status Stitcher::composePanorama(InputArrayOfArrays images, OutputArray pano)
{
    // 这里images 为空，if不成立
    std::vector<UMat> imgs;
    images.getUMatVector(imgs);
    if (!imgs.empty())
    {
        CV_Assert(imgs.size() == imgs_.size());
​
        UMat img;
        seam_est_imgs_.resize(imgs.size());
​
        for (size_t i = 0; i < imgs.size(); ++i)
        {
            imgs_[i] = imgs[i];
            resize(imgs[i], img, Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
            seam_est_imgs_[i] = img.clone();
        }
​
        std::vector<UMat> seam_est_imgs_subset;
        std::vector<UMat> imgs_subset;
​
        for (size_t i = 0; i < indices_.size(); ++i)
        {
            imgs_subset.push_back(imgs_[indices_[i]]);
            seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
        }
​
        seam_est_imgs_ = seam_est_imgs_subset;
        imgs_ = imgs_subset;
    }
​
    // 定义结果图像
    UMat pano_;
​
    std::vector<Point> corners(imgs_.size());
    std::vector<UMat> masks_warped(imgs_.size());
    std::vector<UMat> images_warped(imgs_.size());
    std::vector<Size> sizes(imgs_.size());
    std::vector<UMat> masks(imgs_.size());
​
    // 准备蒙版图像
    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        masks[i].create(seam_est_imgs_[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }
​

    /*
        2.2.1 旋转模型图像变形器
        warper_：用于图像拼接的投影变换
        当拼接模型为：PANORAMA 时，方法为：SphericalWarper ，一种曲线投影算法，将球面图像转换为平面图像。通常用于创建全景图像
        当拼接模型为：SCANS 时，方法为：AffineBasedEstimator，实现仿射变换，对图像进行平面仿射变换
        输入：用于变形的参数(warped_image_scale_ * seam_work_aspect_)
        输出：图像映射算法类RotationWarper的指针
    */
    Ptr<detail::RotationWarper> w = warper_->create(float(warped_image_scale_ * seam_work_aspect_));
    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        // K：存储相机内参，即相机矩阵
        Mat_<float> K;
        cameras_[i].K().convertTo(K, CV_32F);
        K(0,0) *= (float)seam_work_aspect_;
        K(0,2) *= (float)seam_work_aspect_;
        K(1,1) *= (float)seam_work_aspect_;
        K(1,2) *= (float)seam_work_aspect_;
​
        // 拼接图像变形 以及 masks图变形
        // corners:图像左上角
        corners[i] = w->warp(seam_est_imgs_[i], K, cameras_[i].R, interp_flags_, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();
        w->warp(masks[i], K, cameras_[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }
​
​
    /*
        2.2.2 补偿曝光，在检测接缝之前预补偿曝光
        exposure_comp_：用于补偿曝光
        当拼接模型为：PANORAMA 时，方法为：BlocksGainCompensator ，对图像中不同区域进行增益补偿
        当拼接模型为：SCANS 时，方法为：NoExposureCompensator，一种不使用曝光补偿的方法来校正拍摄照片中的一些失真
    */
    exposure_comp_->feed(corners, images_warped, masks_warped);
    for (size_t i = 0; i < imgs_.size(); ++i)
        exposure_comp_->apply(int(i), corners[i], images_warped[i], masks_warped[i]);
​
​
    /*
        2.2.3 检测图像之间的拼接接缝
        seam_finder_：检测图像之间的拼接接缝
        方法为：GraphCutSeamFinder ，使用图形切割算法，可以查找图像中最佳的拼接缝隙
    */
    std::vector<UMat> images_warped_f(imgs_.size());
    for (size_t i = 0; i < imgs_.size(); ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    seam_finder_->find(images_warped_f, corners, masks_warped);
​
    // 清理内存
    seam_est_imgs_.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();
​
​
    /* 将图像及其他信息更新为原始图像尺寸 */
    UMat img_warped, img_warped_s;
    UMat dilated_mask, seam_mask, mask, mask_warped;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;
    bool is_blender_prepared = false;
    double compose_scale = 1;   // 用于融合时图像的比例
    bool is_compose_scale_set = false;
    std::vector<detail::CameraParams> cameras_scaled(cameras_);
    UMat full_img, img;
    for (size_t img_idx = 0; img_idx < imgs_.size(); ++img_idx)
    {
        full_img = imgs_[img_idx];
        if (!is_compose_scale_set)
        {
            if (compose_resol_ > 0)
                compose_scale = std::min(1.0, std::sqrt(compose_resol_ * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;
​
            // 计算相对比例
            //compose_seam_aspect = compose_scale / seam_scale_;
            compose_work_aspect = compose_scale / work_scale_;
​
            // （4）更新变形的图像比例，对图像进行变换
            float warp_scale = static_cast<float>(warped_image_scale_ * compose_work_aspect);
            w = warper_->create(warp_scale);
​
            // 更新图像角点和尺寸
            for (size_t i = 0; i < imgs_.size(); ++i)
            {
                cameras_scaled[i].ppx *= compose_work_aspect;
                cameras_scaled[i].ppy *= compose_work_aspect;
                cameras_scaled[i].focal *= compose_work_aspect;
​
                Size sz = full_img_sizes_[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes_[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes_[i].height * compose_scale);
                }
​
                Mat K;
                cameras_scaled[i].K().convertTo(K, CV_32F);
                Rect roi = w->warpRoi(sz, K, cameras_scaled[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (std::abs(compose_scale - 1) > 1e-1)
        {
            resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
        }
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();
​
        // 变形当前图像 和 当前mask图像
        Mat K;
        cameras_scaled[img_idx].K().convertTo(K, CV_32F);
        w->warp(img, K, cameras_[img_idx].R, interp_flags_, BORDER_REFLECT, img_warped);
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        w->warp(mask, K, cameras_[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
​
​
        //  再次补偿曝光
        exposure_comp_->apply((int)img_idx, corners[img_idx], img_warped, mask_warped);
​
​
        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();
​
        // 确保接缝面罩的尺寸合适，使用形态学膨胀
        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        /*bitwise_and : RGB图像选取掩膜选定的区域 */
        bitwise_and(seam_mask, mask_warped, mask_warped);
​
​
        /*
            2.2.4 图像融合
            blender_：用于融合两个图像的对象
            融合方法为：MultiBandBlender，用于图像拼接的多频带混合算法
        */
        if (!is_blender_prepared)
        {
            blender_->prepare(corners, sizes);
            is_blender_prepared = true;
        }
        blender_->feed(img_warped_s, mask_warped, corners[img_idx]);
    }
    UMat result;
    blender_->blend(result, result_mask_);
​
    // 得到拼接结果
    result.convertTo(pano, CV_8U);
    return OK;
}