#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>

#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>

#include <Eigen/Geometry>
#include <Eigen/Dense>

int count = 0;
std::string dir = "/home/ksakash/Downloads/gazebo_images/images";
float yaw = 0, pitch = 0, roll = 0;

void image_cb (const sensor_msgs::ImageConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }
    cv::Mat image = cv_ptr->image;
    std::string image_name = dir + "/" + std::to_string (count) + ".jpg";
    cv::imwrite (image_name, image);
    std::ofstream f;
    std::string filename = "/home/ksakash/Downloads/gazebo_images/imageData.txt";
    f.open (filename);

    if (!f.is_open ()) {
        std::cout << "error in reading file" << std::endl;
        return;
    }

    f << std::to_string (count) + ".jpg" << ",0,0,0,"
         << std::to_string (yaw) << "," << std::to_string (pitch)
         << "," << std::to_string (roll) << "\n";
    count++;
    f.close ();
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
    ros::init (argc, argv, "capture");
    ros::NodeHandle nh;
    image_transport::ImageTransport it (nh);
    image_transport::Subscriber img_sub = it.subscribe ("/iris/usb_cam/image_raw", 1, image_cb);
    ros::Subscriber pose_sub = nh.subscribe ("/mavros/odometry/in", 10, pose_cb);
    ros::spin ();
    return 0;
}
