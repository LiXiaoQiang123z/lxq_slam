/*
 * @Author: your name
 * @Date: 2022-03-17 19:46:19
 * @LastEditTime: 2022-03-23 16:45:55
 * @LastEditors: Please set LastEditors
 * @Description: 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 * @FilePath: /v-slam-ch9/include/common.h
 */
#pragma once
#ifndef _COMMON_H
#define _COMMON_H


/************** 共用的第三方头文件 ***************/
//std
#include <vector>
#include <string>
//#include <stdio.h>
//#include <math.h>
#include <cmath>
#include <algorithm>
#include <unistd.h>


#include <iostream>
#include <fstream>

// Eigen
#include <Eigen/Core> // eigen核心部分
#include <Eigen/Dense> // 稠密矩阵的代数运算（逆，特征值等）
#include <Eigen/Geometry> // eigen的几何模块 R q T等
// typedefs for eigen
// double matricies
//typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
//typedef Eigen::Matrix<double, 10, 10> Mat1010;
//typedef Eigen::Matrix<double, 13, 13> Mat1313;
//typedef Eigen::Matrix<double, 8, 10> Mat810;
//typedef Eigen::Matrix<double, 8, 3> Mat83;
//typedef Eigen::Matrix<double, 6, 6> Mat66;
//typedef Eigen::Matrix<double, 5, 3> Mat53;
//typedef Eigen::Matrix<double, 4, 3> Mat43;
//typedef Eigen::Matrix<double, 4, 2> Mat42;
typedef Eigen::Matrix<double, 3, 3> Mat33;
//typedef Eigen::Matrix<double, 2, 2> Mat22;
//typedef Eigen::Matrix<double, 8, 8> Mat88;
//typedef Eigen::Matrix<double, 7, 7> Mat77;
//typedef Eigen::Matrix<double, 4, 9> Mat49;
//typedef Eigen::Matrix<double, 8, 9> Mat89;
//typedef Eigen::Matrix<double, 9, 4> Mat94;
//typedef Eigen::Matrix<double, 9, 8> Mat98;
//typedef Eigen::Matrix<double, 8, 1> Mat81;
//typedef Eigen::Matrix<double, 1, 8> Mat18;
//typedef Eigen::Matrix<double, 9, 1> Mat91;
//typedef Eigen::Matrix<double, 1, 9> Mat19;
//typedef Eigen::Matrix<double, 8, 4> Mat84;
//typedef Eigen::Matrix<double, 4, 8> Mat48;
//typedef Eigen::Matrix<double, 4, 4> Mat44;
//typedef Eigen::Matrix<double, 3, 4> Mat34;
//typedef Eigen::Matrix<double, 14, 14> Mat1414;
//
//// float matricies
//typedef Eigen::Matrix<float, 3, 3> Mat33f;
//typedef Eigen::Matrix<float, 10, 3> Mat103f;
//typedef Eigen::Matrix<float, 2, 2> Mat22f;
//typedef Eigen::Matrix<float, 3, 1> Vec3f;
//typedef Eigen::Matrix<float, 2, 1> Vec2f;
//typedef Eigen::Matrix<float, 6, 1> Vec6f;
//typedef Eigen::Matrix<float, 1, 8> Mat18f;
//typedef Eigen::Matrix<float, 6, 6> Mat66f;
//typedef Eigen::Matrix<float, 8, 8> Mat88f;
//typedef Eigen::Matrix<float, 8, 4> Mat84f;
//typedef Eigen::Matrix<float, 6, 6> Mat66f;
//typedef Eigen::Matrix<float, 4, 4> Mat44f;
//typedef Eigen::Matrix<float, 12, 12> Mat1212f;
//typedef Eigen::Matrix<float, 13, 13> Mat1313f;
//typedef Eigen::Matrix<float, 10, 10> Mat1010f;
//typedef Eigen::Matrix<float, 9, 9> Mat99f;
//typedef Eigen::Matrix<float, 4, 2> Mat42f;
//typedef Eigen::Matrix<float, 6, 2> Mat62f;
//typedef Eigen::Matrix<float, 1, 2> Mat12f;
//typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatXXf;
//typedef Eigen::Matrix<float, 14, 14> Mat1414f;

// double vectors
//typedef Eigen::Matrix<double, 14, 1> Vec14;
//typedef Eigen::Matrix<double, 13, 1> Vec13;
//typedef Eigen::Matrix<double, 10, 1> Vec10;
//typedef Eigen::Matrix<double, 9, 1> Vec9;
//typedef Eigen::Matrix<double, 8, 1> Vec8;
//typedef Eigen::Matrix<double, 7, 1> Vec7;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Matrix<double, 4, 1> Vec4;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;

//// float vectors
//typedef Eigen::Matrix<float, 12, 1> Vec12f;
//typedef Eigen::Matrix<float, 8, 1> Vec8f;
//typedef Eigen::Matrix<float, 10, 1> Vec10f;
//typedef Eigen::Matrix<float, 4, 1> Vec4f;
//typedef Eigen::Matrix<float, 12, 1> Vec12f;
//typedef Eigen::Matrix<float, 13, 1> Vec13f;
//typedef Eigen::Matrix<float, 9, 1> Vec9f;
//typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VecXf;
//typedef Eigen::Matrix<float, 14, 1> Vec14f;

// sophus
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

// for cv
#include <opencv2/opencv.hpp> // StereoSGBM
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp> // drawMatches
#include <opencv2/highgui/highgui.hpp> // highGUI图形用户界面
#include <chrono>
//#include <opencv2/imgcodecs.hpp> // 图像编码信息

// G2o
//#include <g2o/core/g2o_core_api.h>
//#include <g2o/core/base_vertex.h>
//#include <g2o/core/base_binary_edge.h>
//#include <g2o/core/base_unary_edge.h>
//#include <g2o/core/block_solver.h>
//#include <g2o/core/solver.h>
//#include <g2o/core/sparse_optimizer.h>
//#include <g2o/core/optimization_algorithm_levenberg.h> //三种方法 LM GN Dogleg
//#include <g2o/core/optimization_algorithm_gauss_newton.h>
//#include <g2o/core/optimization_algorithm_dogleg.h>
//#include <g2o/solvers/dense/linear_solver_dense.h>
//#include <g2o/solvers/csparse/linear_solver_csparse.h>
//#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/block_solver.h>
//#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/types/slam3d/se3quat.h> // se3quat
//#include <g2o/solvers/csparse/linear_solver_csparse.h>
//#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
//#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <boost/format.hpp>  // for formating strings


//Ceres
#include <ceres/ceres.h>

// pangolin :绘图
#include <pangolin/pangolin.h>

using cv::Mat;
using namespace std;
/************** 命名空间声明 ***************/
typedef Sophus::SE3d SE3;
typedef Sophus::SO3d SO3;

/************** TODO：预编译-不同方法的切换 ***************/
/**
 * NO1-bit : Feature_Extract_Match
 * 0    纯手写的特征提取+匹配：时间和精度都差点意思【缺少优化吧】
 * 1    调用opencv函数进行的特征提取+匹配
 * NO2&3-bit : Bundle Adjustment
 * 0   手写-公式推导-残差、雅克比、ldlt分解求delta_x、优化变量更新
 * 1 3 调用g2o || 1-调包 ; 3-手写【失败-执行优化的时候 || 手写特征有问题？？？】
 * 2    【未实现】调用Ceres
 */
#define  Optional_Program 021 // 【8进制表示】
/************** 参数-调整【调参】 ***************/
#define FAST_detect_percent 0.3 // 亮度阈值//0-1.0 百分比
#define FAST_detect_edge_point 20 // 640 * 480 图片边界：只取【edge_point， 640/480-edge_point】部分
#define FAST_N 16     // 9 11 12 14（通常FAST_12 FAST_9 FAST_11 等）




inline Vec2 toVec2(const cv::Point2f p) { return Vec2(p.x, p.y); }
#endif 