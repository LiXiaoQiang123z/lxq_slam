/*
 * @Author: your name
 * @Date: 2022-03-17 19:04:36
 * @LastEditTime: 2022-03-17 20:00:48
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /v-slam-ch9/include/bundleAdjustment.h
 */
//
// Created by lxq on 2021/12/10.
//

#ifndef CH9_BUNDLEADJUSTMENT_H
#define CH9_BUNDLEADJUSTMENT_H

// #include <iostream>
// #include <vector>
// //#include <stdio.h>
// #include <Eigen/Core>
// #include <Eigen/Dense>
// #include "sophus/se3.hpp"
// #include <opencv2/core/core.hpp>
// #include <opencv2/opencv.hpp>


// using namespace std;
// using namespace cv;
#include "myslam/common.h"


Eigen::Matrix<double ,2,6> findPoseJacobian(
        Eigen::Matrix<double,4,4> &T,
        Eigen::Vector3d p);

Eigen::MatrixXd findJacobian(Eigen::MatrixXd &x);

void bundleAdjustment(
        std::vector<cv::Point3d> v_p3d,
        std::vector<cv::Point2d> v_p2d,
        Eigen::Matrix<double,4,4> &T,
        Mat &img,
        const int iterations);

extern  double fx ; //这一组是 十四讲ch5 照片对应相机内参
extern  double fy;
extern  double cx;
extern  double cy;


#endif //CH9_BUNDLEADJUSTMENT_H
