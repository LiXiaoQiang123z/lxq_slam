//
// Created by lxq on 2021/12/10.
//

#include "myslam/bundleAdjustment.h"

/**计算f(X),计算误差err
 *
 * @param x
 * @param v_p2d
 * @return
 */
Eigen::Matrix<double,Eigen::Dynamic, 1> findCostFunction(
        Eigen::VectorXd  x,
        std::vector<cv::Point2d> v_p2d){
    // err = [u,v] - K * T * P ; 测量值【真实值】 - 估计值
    Eigen::Matrix<double,Eigen::Dynamic, 1> ans;
    int size_P = (int)(x.rows()-6)/3;
    ans.resize(2*size_P,1);
    ans.setZero();

    if(size_P != v_p2d.size()){
        std::cout<<"---ERROR: 3d point size != 2d point size"<<std::endl;
        return ans;
    }
    // 主体
    Eigen::Matrix<double,4,4> T = Sophus::SE3d::exp(x.head(6)).matrix();//李代数->李群 T21
    for (int i = 0; i < size_P; ++i) {
        Eigen::Vector4d point;
        point << x[6 + 3*i],x[6 + 3*i+1],x[6 + 3*i+2],1.0;
        Eigen::Vector4d cam_point = T * point; // P2' = T21 * P1

        ans(2*i,0) = v_p2d[i].x - ((fx * cam_point[0]) / cam_point[2]) - cx; // 真实值 - 估计值 || 估计值 = fx * X‘/Z’ + cx
        ans(2*i +1,0) = v_p2d[i].y - ((fy * cam_point[1]) / cam_point[2]) - cy;
    }
    return ans;
}

/** 求取位姿的雅克比
 *
 * @param T
 * @param p
 * @return
 */
Eigen::Matrix<double ,2,6> findPoseJacobian(
        Eigen::Matrix<double,4,4> &T,
        Eigen::Vector3d p){
    Eigen::Matrix<double,2,6> ans;
    Eigen::Vector4d point;
    point << p[0], p[1], p[2], 1.0;
    Eigen::Vector4d cam2_p = T * point;//图1世界坐标点 投影到 图2上。   即：P‘

    double inv_z = 1 / cam2_p[2];
    double inv_z2 = inv_z * inv_z;
    ans << -fx * inv_z, // 对位姿求导结果->雅克比J
            0,
            fx * cam2_p[0] * inv_z2,
            fx * cam2_p[0] * cam2_p[1] * inv_z2,
            -fx - fx * cam2_p[0] * cam2_p[0] * inv_z2,
            fx * cam2_p[1] * inv_z,

            0,
            -fy * inv_z,
            fy * cam2_p[1] * inv_z,
            fy + fy * cam2_p[1] * cam2_p[1] * inv_z2,
            -fy * cam2_p[0] * cam2_p[1] * inv_z2,
            -fy * cam2_p[0] * inv_z;

    return ans; //
}

/** 求取路标点
 *
 * @param T
 * @param p
 * @return
 */
Eigen::Matrix<double ,2,3> findPointJacobian(
        Eigen::Matrix<double,4,4> &T,
        Eigen::Vector3d p){
    Eigen::Matrix<double ,2,3> ans;
    Eigen::Vector4d point;
    point << p[0], p[1], p[2], 1.0;

    Eigen::Vector4d cam2_p = T * point;//图1世界坐标点 投影到 图2上。   即：P‘
    double inv_z = 1/cam2_p[2];
    double inv_z2 = inv_z * inv_z;
    ans << -fx * inv_z, // 对路标点求导结果->雅克比J
            0,
            fx * cam2_p[0] * inv_z2,

            0,
            -fy * inv_z,
            fy * cam2_p[1] * inv_z;
    return ans*T.block<3,3>(0,0);
}


/** 求整体雅克比：位姿+路标点
 *
 * @param x ：待求更新量
 * @return ：整体的雅克比
 */
Eigen::MatrixXd findJacobian(Eigen::VectorXd &x){
    Eigen::MatrixXd ans;
    int size_P=(int)(x.size()-6)/3;
    ans.resize(2*size_P,6+3*size_P); // (2n , 6m+3n)
    ans.setZero();

    Eigen::Matrix<double,4,4> T = Sophus::SE3d::exp(x.head(6)).matrix(); // 李代数->李群
    for (int i = 0; i < size_P; ++i) {
        //matrix.block(i,j,p,q)->针对动态矩阵,matrix.block<p,q>(i,j)->针对静态矩阵
        ans.block(2*i,0,2,6) = findPoseJacobian(    T,x.segment(6+3*i, 3) ); // T:位姿 ； x：路标点
        ans.block(2*i,6+3*i,2,3) = findPointJacobian(   T,x.segment(6+3*i, 3)  );
    }
    return ans;
}


/** 手写BA，求取雅克比
 *
 * @param v_p3d 图像1的相机坐标系
 * @param v_p2d 图像2像素坐标
 * @param T ： T21 把图像1中点旋转到图2中，与图2中的观测做差
 * @param img
 * @param iterations
 */
void bundleAdjustment(std::vector<cv::Point3d> v_p3d, std::vector<cv::Point2d> v_p2d, Eigen::Matrix<double, 4, 4> &T,
                      Mat &img, const int iterations) {
    std::cout<<"---------BA self by V1.2----------"<<std::endl;
    Eigen::VectorXd x;//需要更新的状态量x
    x.resize(6 + 3 * v_p3d.size()); // 位姿6维 + 3D点坐标
    x.setZero();

    //---状态量x初始化 || 设计李群-李代数转化
    Eigen::Matrix3d R = T.block<3,3>(0,0);
    Eigen::Vector3d t = T.block<3,1>(0,3);
    Sophus::SE3d T_x(R,t); // 李群-T
    x.head(6) = T_x.log(); //李群-李代数
    for (int i = 0; i < v_p3d.size(); ++i) { // 图1中的3d路标点
        x[6 + 3*i] = v_p3d[i].x;
        x[6 + 3*i+1] = v_p3d[i].y;
        x[6 + 3*i+2] = v_p3d[i].z;
    }

    //---开始迭代BA
    for (int i = 1; i <=iterations ; ++i) {
        std::cout<<"**********Starting BA ************"<<std::endl;
        auto time1 = (double)cv::getTickCount();
        //---求解 H,g
        Eigen::MatrixXd err = findCostFunction(x,v_p2d); // 求解残差函数 = 真实值 - 测量值

        Eigen::MatrixXd Jacobian = findJacobian(x);     //求解状态x的雅克比 || (2n , 6m+3n)
        Eigen::MatrixXd H = Jacobian.transpose() * Jacobian;      // 求解H矩阵 || 【信息矩阵没必要添加】
        Eigen::VectorXd g = -1 * Jacobian.transpose() * err;      // 求解g

        //---求解delta x
        std::cout<<"Solving ... ..."<<std::endl;
        Eigen::VectorXd delta_x = H.ldlt().solve(g);

        //更新位姿：路标点-相加 ； 位姿-李群乘法
        Eigen::Matrix4d Pos_Matrix = (Sophus::SE3d::exp(x.head(6))).matrix() * (Sophus::SE3d::exp(delta_x.head(6))).matrix() ; //
        Sophus::SE3d new_Pos_se = Sophus::SE3d(Pos_Matrix.block<3,3>(0,0),Pos_Matrix.block<3,1>(0,3));
        x = x + delta_x; //路标点-更新
        x.head(6)=new_Pos_se.log(); //位姿-【李代数表示】

        printf("BA cost %f ms \n", (1000*((double )cv::getTickCount() - time1) / cv::getTickFrequency()));
        std::cout<<"f(x)[err] is : "<<err.sum()/(double )err.size()<<std::endl;
        //--------------------在原图相上画出观测和预测的坐标-------------------------------------
        Eigen::Matrix<double,4,4> T_21 = Sophus::SE3d::exp(x.head(6)).matrix();
        std::cout<<"POSE:"<<"\n"<<T_21<<std::endl;
        cv::Mat temp_Mat=img.clone();
        /// 投影到图像上，展现优化效果
        for(int j=0;j<v_p3d.size();j++) {
            Eigen::Vector4d Point;
            Point << x[6+3*j],x[6+3*j+1],x[6+3*j+2],1.0;
            Eigen::Vector4d cam_Point = T_21 * Point; // T21 * point1 = point2（图2中估计点-图像坐标系）
            cv::Point2d temp_Point2d;
            temp_Point2d.x=(fx*cam_Point(0,0)/cam_Point(2,0))+cx;
            temp_Point2d.y=(fy*cam_Point(1,0)/cam_Point(2,0))+cy;
            cv::circle(temp_Mat,temp_Point2d,3,cv::Scalar(0,0,255),2);
            cv::circle(temp_Mat,v_p2d[j],    2,cv::Scalar(255,0,0),2);
        }
        cv::imshow("REPROJECTION ERROR DISPLAY",temp_Mat);
        std::cout<<"\033[32m"<<"Iteration： "<<i<<" Finish......"<<"\033[37m"<<std::endl;
        std::cout<<"\033[32m"<<"Blue is observation......"<<"\033[37m"<<std::endl;
        std::cout<<"\033[32m"<<"Red is reprojection......"<<"\033[37m"<<std::endl;
        std::cout<<"\033[32m"<<"Press Any Key to continue......"<<"\033[37m"<<std::endl;
        cv::waitKey(0);
    }

}
