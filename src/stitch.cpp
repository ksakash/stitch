#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

#define PI 3.14285714286

using namespace Eigen;
using namespace std;
using namespace std::chrono;

float yaw = 0, pitch = 0, roll = 0;
cv::Mat result;
cv::cuda::GpuMat prev_mask;

struct imageData {
    std::string imageName = "";
    double latitude = 0;
    double longitude = 0;
    double altitudeFeet = 0;
    double altitudeMeter = 0;
    double roll = 0;
    double pitch = 0;
    double yaw = 0;
};

cv::Mat computeUnRotMatrix (imageData& pose) {
    double a = (pose.yaw * PI) / 180;
    double b = (pose.pitch * PI) / 180;
    double g = (pose.roll * PI) / 180;
    Matrix3d Rz;
    Rz << cos (a), -sin (a), 0,
          sin (a), cos (a), 0,
          0, 0, 1;
    Matrix3d Ry;
    Ry << cos (b), 0, sin (b),
          0, 1, 0,
          -sin (b), 0, cos (b);
    Matrix3d Rx;
    Rx << 1, 0, 0,
          0, cos (g), -sin (g),
          0, sin (g), cos (g);
    Matrix3d R = Rz * (Rx * Ry);
    R(0,2) = 0;
    R(1,2) = 0;
    R(2,2) = 1;
    Matrix3d Rtrans = R.transpose ();
    Matrix3d InvR = Rtrans.inverse ();
    cv::Mat transformation = (cv::Mat_<double>(3,3) << InvR(0,0), InvR(0,1), InvR(0,2),
                                                       InvR(1,0), InvR(1,1), InvR(1,2),
                                                       InvR(2,0), InvR(2,1), InvR(2,2));
    return transformation;
}

cv::Mat warpPerspectiveWithPadding (const cv::Mat& image, cv::Mat& transformation) {
    int height = image.rows;
    int width = image.cols;
    cv::Mat small_img;
    cv::resize (image, small_img, cv::Size (width/2, height/2));
    std::vector<cv::Point2f> corners = {cv::Point2f (0,0), cv::Point2f (0,height/2),
                                   cv::Point2f (width/2,height/2), cv::Point2f (width/2,0)};
    std::vector<cv::Point2f> warpedCorners;
    cv::perspectiveTransform (corners, warpedCorners, transformation);
    float xMin = 1e9, xMax = -1e9;
    float yMin = 1e9, yMax = -1e9;
    for (int i = 0; i < 4; i++) {
        xMin = (xMin > warpedCorners[i].x)? warpedCorners[i].x : xMin;
        xMax = (xMax < warpedCorners[i].x)? warpedCorners[i].x : xMax;
        yMin = (yMin > warpedCorners[i].y)? warpedCorners[i].y : yMin;
        yMax = (yMax < warpedCorners[i].y)? warpedCorners[i].y : yMax;
    }
    int xMin_ = (xMin - 0.5);
    int xMax_ = (xMax + 0.5);
    int yMin_ = (yMin - 0.5);
    int yMax_ = (yMax + 0.5);
    cv::Mat translation = (cv::Mat_<double>(3,3) << 1, 0, -xMin_, 0, 1, -yMin_, 0, 0, 1);
    cv::Mat fullTransformation = translation * transformation;
    cv::cuda::GpuMat result;
    cv::cuda::GpuMat gpu_img (small_img);
    cv::cuda::GpuMat gpu_ft (fullTransformation);
    cv::cuda::warpPerspective (gpu_img, result, fullTransformation,
                                cv::Size (xMax_-xMin_, yMax_-yMin_));
    cv::Mat result_ (result.size(), result.type());
    result.download (result_);
    return result_;
}

cv::Mat combinePair (cv::Mat& img1, cv::Mat& img2) {

    cv::cuda::GpuMat img1_gpu (img1), img2_gpu (img2);
    cv::cuda::GpuMat img1_gray_gpu, img2_gray_gpu;

    cv::cuda::cvtColor (img1_gpu, img1_gray_gpu, cv::COLOR_BGR2GRAY);
    cv::cuda::cvtColor (img2_gpu, img2_gray_gpu, cv::COLOR_BGR2GRAY);

    cv::cuda::GpuMat mask1;
    cv::cuda::GpuMat mask2;

    if (prev_mask.empty()) {
        cv::cuda::threshold (img1_gray_gpu, mask1, 1, 255, cv::THRESH_BINARY);
    }
    else {
        mask1 = prev_mask;
    }
    cv::cuda::threshold (img2_gray_gpu, mask2, 1, 255, cv::THRESH_BINARY);

    cv::Ptr<cv::cuda::SURF_CUDA> detector = cv::cuda::SURF_CUDA::create (1000);

    cv::cuda::GpuMat keypoints1_gpu, descriptors1_gpu;
    detector->detectWithDescriptors (img1_gray_gpu, mask1,
                                    keypoints1_gpu, descriptors1_gpu);
    std::vector<cv::KeyPoint> keypoints1;
    detector->downloadKeypoints (keypoints1_gpu, keypoints1);

    cv::cuda::GpuMat keypoints2_gpu, descriptors2_gpu;
    detector->detectWithDescriptors (img2_gray_gpu, mask2,
                                    keypoints2_gpu, descriptors2_gpu);
    std::vector<cv::KeyPoint> keypoints2;
    detector->downloadKeypoints (keypoints2_gpu, keypoints2);

    cv::Ptr<cv::cuda::DescriptorMatcher> matcher =
        cv::cuda::DescriptorMatcher::createBFMatcher ();

    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher->knnMatch (descriptors2_gpu, descriptors1_gpu, knn_matches, 2);

    std::vector<cv::DMatch> matches;
    std::vector<std::vector<cv::DMatch>>::const_iterator it;
    for (it = knn_matches.begin(); it != knn_matches.end(); ++it) {
        if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.55) {
            matches.push_back((*it)[0]);
        }
    }

    std::vector<cv::Point2f> src_pts;
    std::vector<cv::Point2f> dst_pts;
    for (auto m : matches) {
        src_pts.push_back (keypoints2[m.queryIdx].pt);
        dst_pts.push_back (keypoints1[m.trainIdx].pt);
    }

    cv::Mat A = cv::estimateRigidTransform(src_pts, dst_pts, false);
    int height1 = img1.rows, width1 = img1.cols;
    int height2 = img2.rows, width2 = img2.cols;

    std::vector<std::vector<float>> corners1 {{0,0},{0,height1},{width1,height1},{width1,0}};
    std::vector<std::vector<float>> corners2 {{0,0},{0,height2},{width2,height2},{width2,0}};

    std::vector<std::vector<float>> warpedCorners2 (4, std::vector<float>(2));
    std::vector<std::vector<float>> allCorners = corners1;

    for (int i = 0; i < 4; i++) {
        float cornerX = corners2[i][0];
        float cornerY = corners2[i][1];
        warpedCorners2[i][0] = A.at<double> (0,0) * cornerX +
                            A.at<double> (0,1) * cornerY + A.at<double> (0,2);
        warpedCorners2[i][1] = A.at<double> (1,0) * cornerX +
                            A.at<double> (1,1) * cornerY + A.at<double> (1,2);
        allCorners.push_back (warpedCorners2[i]);
    }

    float xMin = 1e9, xMax = -1e9;
    float yMin = 1e9, yMax = -1e9;
    for (int i = 0; i < 7; i++) {
        xMin = (xMin > allCorners[i][0])? allCorners[i][0] : xMin;
        xMax = (xMax < allCorners[i][0])? allCorners[i][0] : xMax;
        yMin = (yMin > allCorners[i][1])? allCorners[i][1] : yMin;
        yMax = (yMax < allCorners[i][1])? allCorners[i][1] : yMax;
    }
    int xMin_ = (xMin - 0.5);
    int xMax_ = (xMax + 0.5);
    int yMin_ = (yMin - 0.5);
    int yMax_ = (yMax + 0.5);

    cv::Mat translation = (cv::Mat_<double>(3,3) << 1, 0, -xMin_, 0, 1, -yMin_, 0, 0, 1);

    cv::cuda::GpuMat warpedResImg;
    cv::cuda::warpPerspective (img1_gpu, warpedResImg, translation,
                               cv::Size (xMax_-xMin_, yMax_-yMin_));

    cv::cuda::GpuMat warpedImageTemp;
    cv::cuda::warpPerspective (img2_gpu, warpedImageTemp, translation,
                               cv::Size (xMax_ - xMin_, yMax_ - yMin_));
    cv::cuda::GpuMat warpedImage2;
    cv::cuda::warpAffine (warpedImageTemp, warpedImage2, A,
                          cv::Size (xMax_ - xMin_, yMax_ - yMin_));

    cv::cuda::GpuMat mask;
    cv::cuda::threshold (warpedImage2, mask, 1, 255, cv::THRESH_BINARY);
    int type = warpedResImg.type();

    warpedResImg.convertTo (warpedResImg, CV_32FC3);
    warpedImage2.convertTo (warpedImage2, CV_32FC3);
    mask.convertTo (mask, CV_32FC3, 1.0/255);
    cv::Mat mask_;
    mask.download (mask_);

    cv::cuda::GpuMat dst (warpedImage2.size(), warpedImage2.type());
    cv::cuda::multiply (mask, warpedImage2, warpedImage2);

    cv::Mat diff_ = cv::Scalar::all (1.0) - mask_;
    cv::cuda::GpuMat diff (diff_);
    cv::cuda::multiply(diff, warpedResImg, warpedResImg);
    cv::cuda::add (warpedResImg, warpedImage2, dst);
    dst.convertTo (dst, type);

    cv::cuda::GpuMat dst_gray;
    cv::cuda::cvtColor (dst, dst_gray, cv::COLOR_BGR2GRAY);
    cv::cuda::threshold (dst_gray, prev_mask, 1, 255, cv::THRESH_BINARY);

    float h = prev_mask.rows;
    float w = prev_mask.cols;
    if (h > 4000 || w > 4000) {
        if (h > 4000) {
            float hx = 4000.0/h;
            h = h * hx;
            w = w * hx;
        }
        else if (w > 4000) {
            float wx = 4000.0/w;
            w = w * wx;
            h = h * wx;
        }
        cv::cuda::resize (prev_mask, prev_mask, cv::Size (w, h));
    }

    cv::Mat ret;
    dst.download (ret);
    return ret;
}

cv::Mat combine (std::vector<cv::Mat>& imageList) {
    cv::Mat result = imageList[0];
    for (int i = 1; i < imageList.size(); i++) {
        cv::Mat image = imageList[i];
        cout << i << endl;
        auto start = high_resolution_clock::now();
        result = combinePair (result, image);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds> (end-start);
        cout << "time taken by the functions: " << duration.count() << endl;
        float h = result.rows;
        float w = result.cols;
        if (h > 4000 || w > 4000) {
            if (h > 4000) {
                float hx = 4000.0/h;
                h = h * hx;
                w = w * hx;
            }
            else if (w > 4000) {
                float wx = 4000.0/w;
                w = w * wx;
                h = h * wx;
            }
        }
        cout << h << " " << w << endl;
        cv::resize (result, result, cv::Size (w, h));
    }
    return result;
}

imageData readData () {
    imageData id;
    id.yaw = yaw;
    id.pitch = pitch;
    id.roll = roll;
    return id;
}

cv::Mat changePerspective (cv::Mat& image, imageData& data) {
    std::cout << "Warping Images Now" << std::endl;
    cv::Mat M = computeUnRotMatrix (data);
    cv::Mat correctedImage = warpPerspectiveWithPadding (image, M);
    std::cout << "Image Warping Done" << std::endl;
    return correctedImage;
}

int count_ = 0;
void image_cb (const sensor_msgs::ImageConstPtr& msg) {
    count_++;
    if (count_ % 40 != 0)
        return;
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat image = cv_ptr->image;
    if (result.empty()) {
        result = image;
    }
    imageData data = readData ();
    image = changePerspective (image, data);
    result = combinePair (result, image);
    float h = result.rows;
    float w = result.cols;
    if (h > 4000 || w > 4000) {
        if (h > 4000) {
            float hx = 4000.0/h;
            h = h * hx;
            w = w * hx;
        }
        else if (w > 4000) {
            float wx = 4000.0/w;
            w = w * wx;
            h = h * wx;
        }
        cv::resize (result, result, cv::Size (w, h));
    }
}

void pose_cb (const nav_msgs::Odometry& msg) {
    float w = msg.pose.pose.orientation.w;
    float x = msg.pose.pose.orientation.x;
    float y = msg.pose.pose.orientation.y;
    float z = msg.pose.pose.orientation.z;
    Eigen::Quaternionf q (w, x, y, z);
    auto euler = q.toRotationMatrix ().eulerAngles (2, 1, 0);
    yaw = euler[0];
    pitch = euler[1];
    roll = euler[2];
}

int main (int argc, char** argv) {
    ros::init (argc, argv, "stitch");
    ros::NodeHandle nh;
    image_transport::ImageTransport it (nh);
    image_transport::Subscriber img_sub =
                it.subscribe ("/iris/usb_cam/image_raw", 1, image_cb);
    ros::Subscriber pose_sub =
                    nh.subscribe ("/mavros/odometry/in", 10, pose_cb);
    ros::spin ();
    return 0;
}
