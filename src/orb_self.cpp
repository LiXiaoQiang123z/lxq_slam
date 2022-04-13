//
// Created by lxq on 2021/12/10.
//
#include "myslam/orb_self.h"
#include <cmath>

/* FAST角点提取函数
 *
 * @param img 需要提取角点的图像
 * @param percent 亮度阈值//0-1.0 百分比
 * @return
 */
VecKeyPoint FAST_detect(const cv::Mat &img, double percent){
    VecKeyPoint keypoints_;
    int keypoint_num = 0;
    // KeyPoint keypoint_(1.2,2.0,123,5);
    // 检查半径为3 ，的16个点
    int mask[16*2] = {
            -3,0,
            -3,1,
            -2,2,
            -1,3,
            0,3,
            1,3,
            2,2,
            3,1,
            3,0,
            3,-1,
            2,-2,
            1,-3,
            0,-3,
            -1,-3,
            -2,-2,
            -3,-1
    };
    std::vector<int> Vecfeature;
    int m_4[4]={1,5,9,13};//1,5,9,13 // 0,4,8,12 || 预测试操作：排除绝大多数不是角点的
    uchar num_differ[3]={0};
    uchar filter_num_differ[3]={0}; //分别代表 大-小-相等的数量：
    // bool** filter_boundary = {false};// 标志位：标记像素点是否被筛选过
    
    /* #define - const 变量 */
    // const uchar FAST_detect_edge_point = 20; // 边缘的值如何得来 || 480 * 640
    // const uchar FAST_N = 12;// 9 11 12 14

    for(int i = FAST_detect_edge_point; i< img.rows-FAST_detect_edge_point; ++i) {
        for (int j = FAST_detect_edge_point; j < img.cols - FAST_detect_edge_point; ++j) {
            num_differ[1] = num_differ[0] = num_differ[2] = 0; //标志位清零
            filter_num_differ[0] = filter_num_differ[1] = 0; // 数量清零
            // 1.预测试操作
            for (int m=0; m<4;++m){ 
                /**
                 * @brief 利用ptr遍历图像 ：圆上的点连续N个点 Im > Ip+T / Im < Ip-T
                 * 圆上的点Im = img.ptr<uchar>(i + mask[2 * m_4[m]])[j + mask[2 * m_4[m] + 1]]    || (i + mask[2 * m])  (j + mask(2*m+1)) ||
                 * Ip = img.ptr<uchar>(i)[j]
                 * T = percent * img.ptr<uchar>(i)[j]
                 */
                if (img.ptr<uchar>(i + mask[2 * m_4[m]])[j + mask[2 * m_4[m] + 1]] >
                    (1 + percent) * img.ptr<uchar>(i)[j])
                    ++filter_num_differ[0];
                else if (img.ptr<uchar>(i + mask[2 * m_4[m]])[j + mask[2 * m_4[m] + 1]] <
                         (1 - percent) * img.ptr<uchar>(i)[j])
                    ++filter_num_differ[1];
                else
                    ++filter_num_differ[2];
            }

            if(filter_num_differ[0] < 3 && filter_num_differ[1] < 3) //不成立
                continue;
            // 2. 检测以r=3位半径的圆，与圆心像素值之比
            // 课本方案： 【连续】圆上的点连续N个点 Im > Ip+T / Im < Ip-T
            for (int k = 0; k < 16; ++k) {
                if (img.ptr<uchar>(i + mask[2 * k])[j + mask[2 * k + 1]] > (1 + percent) * img.ptr<uchar>(i)[j])
                    ++num_differ[0];
                if (img.ptr<uchar>(i + mask[2 * k])[j + mask[2 * k + 1]] < (1 - percent) * img.ptr<uchar>(i)[j])
                    ++num_differ[1];
                else
                    ++num_differ[2];
            }

            if (num_differ[0] >= FAST_N) {
                ++keypoint_num;
                KeyPoint keypoint_(j, i, keypoint_num, num_differ[0]); //列-行-num
                keypoints_.push_back(keypoint_);
            }
            if (num_differ[1] >= FAST_N) {
                ++keypoint_num;
                KeyPoint keypoint_(j, i, keypoint_num, num_differ[1]);
                keypoints_.push_back(keypoint_);
            }
            
        }
    }

//    cout<<"FAST detect is over" << endl;
    return keypoints_;
}


// /** 计算角度函数 || ORB特征改进 -尺寸：金字塔 -旋转：角度
//  *
//  * @param image 图像
//  * @param keypoints 特征点
//  */
// void computeAngle(const cv::Mat &image, std::vector<KeyPoint> &keypoints){
//     int half_path_size = 8;
//     int bad_points = 0;
//     for(auto &kp:keypoints){
//         // start
//         kp.angle = 0;
//         float m01=0,m10=0;

//         //边界检测
//         if(kp.x_ < half_path_size || kp.y_ < half_path_size ||
//            kp.x_ >= (image.cols - half_path_size) || kp.y_ >= (image.rows - half_path_size)){
//             bad_points++;    
//             continue;
//         }else{
//             for(int dx = -half_path_size; dx < half_path_size; ++dx){
//                 for(int dy = -half_path_size; dy < half_path_size; ++dy){
//                     //求取块内每个点的像素
//                     uchar I_pixel = image.at<uchar>(kp.y_ + dy, kp.x_ + dx);
//                     m01 += dy * I_pixel;
//                     m10 += dx * I_pixel;
//                 }
//             }
//         }
//         //opencv:角度制 ； std:atan：弧度制
//         float theta = atan(m01/m10) * 180 / M_PI;
// //        cout<<"theta"<< " "<< theta<<endl;
//         kp.angle = theta;
//         //end
//     }
// }

/** ORB描述子 || 采用固定提取点
 *
 * @param image 对应图像
 * @param keypoint 输入特征点
 *
 * @param vecDes 输出描述子
 */
void computeORBDesc(const cv::Mat &image, std::vector<KeyPoint> &keypoint, std::vector<DescType> &vecDes){
    
    const int half_path_size = 8;
    const int half_boundary = 16;
    int bad_points = 0;
    for(auto &kp:keypoint){
        // start
//        kp.angle = 0;
        float m01=0,m10=0;
        // 1. 计算角度
        //边界检测 || 边框16 、 遍历的就是(-8，+8)
        if(kp.x_ < half_boundary || kp.y_ < half_boundary ||
           kp.x_ >= (image.cols - half_boundary) || kp.y_ >= (image.rows - half_boundary)){
            bad_points++;    
            continue;
        }else{
            for(int dx = -half_path_size; dx < half_path_size; ++dx){
                for(int dy = -half_path_size; dy < half_path_size; ++dy){
                    //求取块内每个点的像素
                    uchar I_pixel = image.at<uchar>(kp.y_ + dy, kp.x_ + dx);
                    m01 += dy * I_pixel;
                    m10 += dx * I_pixel;
                }
            }
        }
        // angle should be arc tan(m01/m10); || 2.求出 夹角
        float m_sqrt = sqrt(m01 * m01 + m10 * m10) + 1e-18; // avoid divide by zero
        float sin_theta = m01 / m_sqrt;
        float cos_theta = m10 / m_sqrt;
        // std::cout << "theta" << sin_theta / cos_theta << std::endl;
        //opencv:角度制 ； std:atan：弧度制
        // float theta = atan(m01/m10) * 180 / M_PI;

        // 2.计算描述子
        DescType d(256, false);
        bool push_flag = true;
        for (int i = 0; i < 256; ++i) {
            //start
            // //还原成弧度制
            // auto sin_theta = float (sin(theta*M_PI/180));
            // auto cos_theta = float (cos(theta*M_PI/180));

            cv::Point2f kp_p = cv::Point2f(kp.x_, kp.y_);
            //计算p' q'
            cv::Point2f pp = cv::Point2f((cos_theta * ORB_pattern[i * 4] - sin_theta * ORB_pattern[i * 4 + 1]),
                                         (sin_theta * ORB_pattern[i * 4] + cos_theta * ORB_pattern[i * 4 + 1])) + kp_p;
            cv::Point2f qq = cv::Point2f((cos_theta * ORB_pattern[i * 4 + 2] - sin_theta * ORB_pattern[i * 4 + 3]),
                                         (sin_theta * ORB_pattern[i * 4 + 2] + cos_theta * ORB_pattern[i * 4 + 3])) + kp_p;

            //计算边界
            if (pp.x < 0 || pp.y < 0 || pp.x > image.cols || pp.y > image.rows||
                qq.x < 0 || qq.y < 0 || qq.x > image.cols || qq.y > image.rows){
                d.clear();
                push_flag = false;
                continue;
            }
            d[i] = image.at<uchar>(pp.y, pp.x) <= image.at<uchar>(qq.y, qq.x) ? 1 : 0;
        }
        if(push_flag){ // 圈内：0or1
            vecDes.push_back(d);
        }else{ //圈外
            vecDes.push_back({});
        }
    }
    std::cout << "bad/total: " << bad_points << "/" << vecDes.size() << std::endl;
    return;
}


/** 描述子之间 暴力匹配
 *
 * @param vecDesc1 输入待比较描述子1
 * @param vecDesc2 输入描述子2
 * @param matches 输出匹配对 id1-id2-distance
 */
void BfMatch(std::vector<DescType> &vecDesc1, std::vector<DescType> &vecDesc2, std::vector<Match> &matches) {
    double starttime = (double) cv::getTickCount();
    int d_max = 50; //30 40 等等

    //start
    for (int i1 = 0; i1 < vecDesc1.size(); ++i1) {
        if (vecDesc1[i1].empty()) continue;
        Match m{i1, 0, 256};//初始化 匹配对+距离

        for (int i2 = 0; i2 < vecDesc2.size(); ++i2) {
            if (vecDesc2[i2].empty()) continue;

            int distance = 0;
            for (int j = 0; j < 256; ++j) {
                distance += vecDesc1[i1][j] ^ vecDesc2[i2][j]; //异或
            }

            if (distance < d_max && distance < m.distance_) { //第一次筛选：筛的是最小距离
                m.distance_ = distance;
                m.id2_ = i2;
            }
        }
        if (m.distance_ < d_max) { //在筛选一次：可能i1中的点没有在i2中找到匹配点。
            matches.push_back(m);
        }
    }
    //end

    // 时间测量
    double times_bfMatch = ((double) cv::getTickCount() - starttime) / cv::getTickFrequency();
    std::cout << "bfMatchtime: " << times_bfMatch << std::endl;

    return;
}


/**
 * @brief 手写特征提取 + 匹配
 * 
 * @param img1 
 * @param img2 
 * @param img3_depth 
 * @param orb_data 
 */
void self_orb(
        cv::Mat &img1,
        cv::Mat &img2,
        cv::Mat &img_depth,
        self_orb_con &orb_data){
    double t = (double)cv::getTickCount();
    
    //---1.FAST角点提取
    orb_data.keypoints1 = FAST_detect(img1, FAST_detect_percent);
    orb_data.keypoints2 = FAST_detect(img2, FAST_detect_percent);
    std::cout << "FAST detect num1： "<< orb_data.keypoints1.size() << " "
         << "FAST detect num2:  "<< orb_data.keypoints2.size() << std::endl;

    // //---2.求出描述子
    // //计算角度【旋转】 || 后面-金字塔【尺寸】
    // computeAngle(img1, orb_data.keypoints1);
    // computeAngle(img2, orb_data.keypoints2);

    // 2 计算描述子 ORB
    std::vector<DescType > vecDes1,vecDes2;
    computeORBDesc(img1, orb_data.keypoints1, vecDes1);
    computeORBDesc(img2, orb_data.keypoints2, vecDes2);

    // 3.进行BfMatch，匹配
    BfMatch(vecDes1, vecDes2, orb_data.matches);
    std::cout<<"matches:  "<< orb_data.matches.size()<<std::endl;
    printf("ORB detect cost %f ms \n", ((cv::getTickCount() - t) / cv::getTickFrequency()));

    /*************************---show：图像展示部分************************************/
//    // ---show  [画出特征点【画点】]
//    for(auto & l : orb_data.keypoints1){ //画出特征点
//        cv::Point p;
//        p.x = l.x_;
//        p.y = l.y_;
//        cv::circle(img1, p, 3, cv::Scalar(0, 0, 255), -1);
//    }
//    for(auto & l : orb_data.keypoints2){
//        cv::Point p;
//        p.x = l.x_;
//        p.y = l.y_;
//        cv::circle(img2, p, 3, cv::Scalar(0, 0, 255), -1);
//    }

    //---show [【匹配点】连线]
    // int k = 0;
//    vector<double> vecdepth;
    for(unsigned i = 0; i< orb_data.matches.size(); ++i){
        uint depth_row = orb_data.keypoints1[orb_data.matches[i].id1_].y_;
        uint depth_col = orb_data.keypoints1[orb_data.matches[i].id1_].x_;

        ushort depth1 = img_depth.ptr<ushort>(depth_row)[depth_col];
        if(depth1 !=0 &&
           !isnan(depth1) &&
           !isinf(depth1)) {
            double d_depth1 = depth1 / 5000.0; //1000 类似归一化
            orb_data.vecdepth.push_back(d_depth1);
            orb_data.matches_ok.push_back(orb_data.matches[i]);
            // ++k;
        }
    }
    std::cout << "一共找到了" << orb_data.matches_ok.size() << "组匹配点" << std::endl;

//    // 图像合并 ，方便对比
//    cv::hconcat(img1, img2, orb_data.combine); //图像合并
//    for(uint i = 0; i < orb_data.matches.size(); ++i){
//        cv::Point p1 = cv::Point (orb_data.keypoints1[orb_data.matches[i].id1_].x_, orb_data.keypoints1[orb_data.matches[i].id1_].y_);
//        cv::Point p2 = cv::Point (orb_data.keypoints2[orb_data.matches[i].id2_].x_ + 640, orb_data.keypoints2[orb_data.matches[i].id2_].y_); //为啥+640: 并排显示
//        cv::line(orb_data.combine,p1, p2, cv::Scalar(200, 0, 0), 1);//BGR成像 cv::Scalar(200, 0, 0) || cv::Scalar::all(-1)
//    }
}


// /**
//  * @brief 手写特征提取 + 匹配
//  * 
//  * @param img1 
//  * @param img2 
//  * @param img3_depth 
//  * @param orb_data 
//  */
// void self_orb_cv(
//         cv::Mat &img1,
//         cv::Mat &img2,
//         cv::Mat &img_depth,
//         self_orb_con &orb_data){
//     double t = (double)cv::getTickCount();
    
//     //---1.FAST角点提取
//     orb_data.keypoints1 = FAST_detect(img1, FAST_detect_percent);
//     orb_data.keypoints2 = FAST_detect(img2, FAST_detect_percent);
//     std::cout << "FAST detect num1： "<< orb_data.keypoints1.size() << " "
//          << "FAST detect num2:  "<< orb_data.keypoints2.size() << std::endl;

//     //---2.求出描述子
//     //计算角度【旋转】 || 后面-金字塔【尺寸】
//     computeAngle(img1, orb_data.keypoints1);
//     computeAngle(img2, orb_data.keypoints2);

//     //计算描述子 ORB
//     std::vector<DescType > vecDes1,vecDes2;
//     computeORBDesc(img1, orb_data.keypoints1, vecDes1);
//     computeORBDesc(img2, orb_data.keypoints2, vecDes2);
//     //进行BfMatch，匹配
//     BfMatch(vecDes1, vecDes2, orb_data.matches);
//     std::cout<<"matches:  "<< orb_data.matches.size()<<std::endl;
//     printf("ORB detect cost %f ms \n", ((cv::getTickCount() - t) / cv::getTickFrequency()));

//     //---show [【匹配点】连线]
//     int k = 0;
// //    vector<double> vecdepth;
//     for(unsigned i = 0; i< orb_data.matches.size(); ++i){
//         uint depth_row = orb_data.keypoints1[orb_data.matches[i].id1_].y_;
//         uint depth_col = orb_data.keypoints1[orb_data.matches[i].id1_].x_;

//         ushort depth1 = img_depth.ptr<ushort>(depth_row)[depth_col];
//         if(depth1 !=0 &&
//            !isnan(depth1) &&
//            !isinf(depth1)) {
//             double d_depth1 = depth1 / 5000.0; //为啥是5000？s
//             orb_data.vecdepth.push_back(d_depth1);
//             orb_data.matches_ok.push_back(orb_data.matches[i]);
//             ++k;
//         }
//     }
//     std::cout << "一共找到了" << orb_data.matches_ok.size() << "组匹配点" << std::endl;

// }
