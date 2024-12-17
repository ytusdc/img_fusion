#ifndef OPENCV_STITCHING_STITCHER_HPP
#define OPENCV_STITCHING_STITCHER_HPP
​
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
​
​
#if defined(Status)
#  warning Detected X11 'Status' macro definition, it can cause build conflicts. Please, include this header before any X11 headers.
#endif
​
namespace cv {
​
class CV_EXPORTS_W Stitcher
{
public:
    /* 定义了一个静态常量 ORIG_RESOL，用于表示原始分辨率，其值设为-1.0。该常量可以用于缩放图像，
    以改变图像的分辨率，使得拼接更加准确。 */
#if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900/*MSVS 2015*/)
    static constexpr double ORIG_RESOL = -1.0;
#else
    // support MSVS 2013
    static const double ORIG_RESOL; // Initialized in stitcher.cpp
#endif
​
    /* 返回函数运行的状态 */
    enum Status
    {
        OK = 0,
        ERR_NEED_MORE_IMGS = 1,
        ERR_HOMOGRAPHY_EST_FAIL = 2,
        ERR_CAMERA_PARAMS_ADJUST_FAIL = 3
    };
​
    /* 图像拼接模式 */
    enum Mode
    {
        /* 全景模式，即从不同视角拍摄的图像进行拼接, 透视变换*/
        PANORAMA = 0,
        /* 扫描模式，即同一个平面上的图像进行拼接, 仿射变换,默认不补偿曝光 */
        SCANS = 1,
    };
​
    /* 1 拼接模型选择以及一些参数初始化，返回CoVision_stitcher类的指针 */
    CV_WRAP static Ptr<Stitcher> create(Mode mode = Stitcher::PANORAMA);
    
    /* 2 图像拼接获得最终结果，输入：图像组，输出：拼接结果图像 */
    CV_WRAP Status stitch(InputArrayOfArrays images, OutputArray pano);
    CV_WRAP Status stitch(InputArrayOfArrays images, InputArrayOfArrays masks, OutputArray pano);
​
​
    /* 一些设置参数的接口函数， 在 create 函数中初始化*/
    // 1.1 registr_resol_ ：影响work_scale_的大小 
    CV_WRAP double registrationResol() const { return registr_resol_; }
    CV_WRAP void setRegistrationResol(double resol_mpx) { registr_resol_ = resol_mpx; }
    // 1.2 seam_est_resol_ ：影响seam_scale_的大小
    CV_WRAP double seamEstimationResol() const { return seam_est_resol_; }
    CV_WRAP void setSeamEstimationResol(double resol_mpx) { seam_est_resol_ = resol_mpx; }
    // 1.3
    CV_WRAP double compositingResol() const { return compose_resol_; }
    CV_WRAP void setCompositingResol(double resol_mpx) { compose_resol_ = resol_mpx; }
    // 1.4
    CV_WRAP double panoConfidenceThresh() const { return conf_thresh_; }
    CV_WRAP void setPanoConfidenceThresh(double conf_thresh) { conf_thresh_ = conf_thresh; }
    // 1.5
    Ptr<detail::SeamFinder> seamFinder() { return seam_finder_; }
    const Ptr<detail::SeamFinder> seamFinder() const { return seam_finder_; }
    void setSeamFinder(Ptr<detail::SeamFinder> seam_finder) { seam_finder_ = seam_finder; }
    // 1.6
    Ptr<detail::Blender> blender() { return blender_; }
    const Ptr<detail::Blender> blender() const { return blender_; }
    void setBlender(Ptr<detail::Blender> b) { blender_ = b; }
    // 1.7 features_finder_ ： 特征检测和描述方法模型
    Ptr<Feature2D> featuresFinder() { return features_finder_; }
    Ptr<Feature2D> featuresFinder() const { return features_finder_; }
    void setFeaturesFinder(Ptr<Feature2D> features_finder)
    {
        features_finder_ = features_finder;
    }
    // 1.8
    CV_WRAP InterpolationFlags interpolationFlags() const { return interp_flags_; }
    CV_WRAP void setInterpolationFlags(InterpolationFlags interp_flags) { interp_flags_ = interp_flags; }
​
    /* 当mode为 PANORAMA 和 SCANS 时都需要设置的参数 */
    // 1.9
    Ptr<detail::Estimator> estimator() { return estimator_; }
    const Ptr<detail::Estimator> estimator() const { return estimator_; }
    void setEstimator(Ptr<detail::Estimator> estimator)
    {
        estimator_ = estimator;
    }
    // 1.10
    CV_WRAP bool waveCorrection() const { return do_wave_correct_; }
    CV_WRAP void setWaveCorrection(bool flag) { do_wave_correct_ = flag; }
    // 1.11
    Ptr<detail::FeaturesMatcher> featuresMatcher() { return features_matcher_; }
    Ptr<detail::FeaturesMatcher> featuresMatcher() const { return features_matcher_; }
    void setFeaturesMatcher(Ptr<detail::FeaturesMatcher> features_matcher)
    {
        features_matcher_ = features_matcher;
    }
    // 1.12
    Ptr<detail::BundleAdjusterBase> bundleAdjuster() { return bundle_adjuster_; }
    const Ptr<detail::BundleAdjusterBase> bundleAdjuster() const { return bundle_adjuster_; }
    void setBundleAdjuster(Ptr<detail::BundleAdjusterBase> bundle_adjuster)
    {
        bundle_adjuster_ = bundle_adjuster;
    }
    // 1.13
    Ptr<WarperCreator> warper() { return warper_; }
    const Ptr<WarperCreator> warper() const { return warper_; }
    void setWarper(Ptr<WarperCreator> creator) { warper_ = creator; }
    // 1.14
    Ptr<detail::ExposureCompensator> exposureCompensator() { return exposure_comp_; }
    const Ptr<detail::ExposureCompensator> exposureCompensator() const { return exposure_comp_; }
    void setExposureCompensator(Ptr<detail::ExposureCompensator> exposure_comp)
    {
        exposure_comp_ = exposure_comp;
    }
  
    /* 当mode为 PANORAMA 需要设置的参数 */
    // 1.15
    detail::WaveCorrectKind waveCorrectKind() const { return wave_correct_kind_; }
    void setWaveCorrectKind(detail::WaveCorrectKind kind) { wave_correct_kind_ = kind; }
    
​
    /* 2 图像拼接获得最终结果，输入：图像组，输出：拼接结果图像 */
    // 2.1 特征检测和图像匹配 与 估计图像拼接中相机参数
    CV_WRAP Status estimateTransform(InputArrayOfArrays images, InputArrayOfArrays masks = noArray());
    // 2.1.1 特征检测和图像匹配
    Status matchImages();
    // 2.1.2 估计图像拼接中相机参数, 包括旋转矩阵和平移矩阵
    Status estimateCameraParams();
​
    // 2.2 得到拼接结果
    CV_WRAP Status composePanorama(OutputArray pano);
    CV_WRAP Status composePanorama(InputArrayOfArrays images, OutputArray pano);
​
​
​
    // 设置特征点mask图像
    const cv::UMat& matchingMask() const { return matching_mask_; }
    void setMatchingMask(const cv::UMat &mask)
    {
        CV_Assert(mask.type() == CV_8U && mask.cols == mask.rows);
        matching_mask_ = mask.clone();
    }
​
    // 自定义设置一组变换，以便在拼接图像之前，可以先将它们进行变换。例如，可以将拼接的图像平移，旋转或缩放
    Status setTransform(InputArrayOfArrays images,
                        const std::vector<detail::CameraParams> &cameras,
                        const std::vector<int> &component);
    Status setTransform(InputArrayOfArrays images, const std::vector<detail::CameraParams> &cameras);
​
    // 获得一些参数
    std::vector<int> component() const { return indices_; }
    std::vector<detail::CameraParams> cameras() const { return cameras_; }
    CV_WRAP double workScale() const { return work_scale_; }
    UMat resultMask() const { return result_mask_; }
​
private:
​
    double registr_resol_;      /* 影响work_scale_的大小 */
    double work_scale_;         /* 控制特征检测时图像的缩放比例（0~1）*/
    double seam_est_resol_;     /* 影响seam_scale_的大小 */
    double seam_scale_;         /* 控制图像拼接时图像的缩放比例（0~1） */
    double compose_resol_;      /* 影响compose_scale的大小 */ 
    double conf_thresh_;        /* 置信度阈值，确定图像是否参与拼接 */ 
    InterpolationFlags interp_flags_;                    /* 指定插值方式：双线性插值（默认） */
    Ptr<Feature2D> features_finder_;                     /* 特征检测和描述方法模型 */
    Ptr<detail::FeaturesMatcher> features_matcher_;      /* 特征点匹配 */
    cv::UMat matching_mask_;                             /* 特征点蒙版 */
    Ptr<detail::BundleAdjusterBase> bundle_adjuster_;    /* 相机参数优化调整器 */   
    Ptr<detail::Estimator> estimator_;                   /* 用于图像拼接的估计器 */
    bool do_wave_correct_;                               /* 是否波形优化 */
    detail::WaveCorrectKind wave_correct_kind_;          /* 波形校正策略 */  
    Ptr<WarperCreator> warper_;                          /* 创建适用于图像拼接的投影变换 */
    Ptr<detail::ExposureCompensator> exposure_comp_;     /* 补偿曝光类型 */
    Ptr<detail::SeamFinder> seam_finder_;                /* 计算拼接边缘的对象 */
    Ptr<detail::Blender> blender_;                       /* 融合两个图像的对象 */  
​
    /* 图像与mask图的数组 */
    std::vector<cv::UMat> imgs_;
    std::vector<cv::UMat> masks_;
​
    std::vector<cv::Size> full_img_sizes_;              /*完整的图像尺寸*/
    std::vector<detail::ImageFeatures> features_;       /* 图像特征点信息（图像序号，图像大小，关键点等） */
    std::vector<detail::MatchesInfo> pairwise_matches_; /* 包含有关两个图像之间匹配的信息 */
    std::vector<cv::UMat> seam_est_imgs_;               /* 指定图像拼接时，用于检测拼接边界的图像 */
    std::vector<int> indices_;                          
    std::vector<detail::CameraParams> cameras_;         /* 相机参数：内参外参等 */
    UMat result_mask_;
    
    double seam_work_aspect_;
    double warped_image_scale_;
};
​
} // namespace cv
​
#endif // OPENCV_STITCHING_STITCHER_HPP