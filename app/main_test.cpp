//
// Created by lxq on 2022/4/1.
//

#include "myslam/common.h"

using namespace std;

/*
 * 00 - ceres - 基本的函数
 * 01 - ceres -一个例子
 * 02 - Ceres slam十四讲例子
 */
#define optional_method 02
//************* Ceres-test  **************
struct CURVE_FITTING_COST // 1.构建残差函数
{
//构建代价函数结构体，residual 为残差。
//last_point_a_为这一帧中的点a，curr_point_b_为点a旋转后和上一帧里最近的点
//curr_point_c_为点b同线或上线号的点，curr_point_d_为点b下线号的点
//b，c，d与a点距离不超过3m ; plane_norm为根据向量bc和bd求出的法向量
/*
 * 激光slam的啊 - 点到面的距离 = 残差
 */
    //类似构造函数
    CURVE_FITTING_COST(Eigen::Vector3d _curr_point_a_, Eigen::Vector3d _last_point_b_,
                       Eigen::Vector3d _last_point_c_, Eigen::Vector3d _last_point_d_):
            curr_point_a_(_curr_point_a_),last_point_b_(_last_point_b_),
            last_point_c_(_last_point_c_),last_point_d_(_last_point_d_)
    {
        plane_norm = (last_point_d_ - last_point_b_).cross(last_point_c_ - last_point_b_); // 法向量
        plane_norm.normalize(); //归一化
    }

    template <typename T> // 函数模板
    //plane_norm点乘向量ab为a点距面bcd的距离，即残差
    bool operator()(const T* q,const T* t,T* residual)const
    {
        Eigen::Matrix<T, 3, 1> p_a_curr{T(curr_point_a_.x()), T(curr_point_a_.y()), T(curr_point_a_.z())};
        Eigen::Matrix<T, 3, 1> p_b_last{T(last_point_b_.x()), T(last_point_b_.y()), T(last_point_b_.z())};
        Eigen::Quaternion<T> rot_q{q[3], q[0], q[1], q[2]};
        Eigen::Matrix<T, 3, 1> rot_t{t[0], t[1], t[2]};
        Eigen::Matrix<T, 3, 1> p_a_last;
        p_a_last=rot_q * p_a_curr + rot_t; // 把当前帧旋转回上一帧吗？ q_i_j
        residual[0]=abs((p_a_last - p_b_last).dot(plane_norm)); // 点乘
        return true;
    }

    const Eigen::Vector3d curr_point_a_,last_point_b_,last_point_c_,last_point_d_;
    Eigen::Vector3d plane_norm;
};
//*************** 02 - Ceres slam十四讲例子 ************
struct CURVE_FITTING_COST2{

    CURVE_FITTING_COST2(double x,double y) : x_(x),y_(y){}

    template<class T>
    bool operator()(
            const T *const abc,
            T *residual )const{
        residual[0] = T(y_) - ceres::exp(abc[0]*T(x_)*T(x_) + abc[1]*T(x_) + abc[2]);
        return true;
    };

    const double x_,y_;
};

int main(int argc, char** argv){

#if optional_method==00
    cout<< "*********** Ceres-test ***********"<<endl;
    //2.构建优化问题
    //优化问题构建
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
    ceres::Problem::Options problem_options;
    ceres::Problem problem(problem_options);
    problem.AddParameterBlock(para_q, 4, q_parameterization); // 添加参数块
    problem.AddParameterBlock(para_t, 3);
    // 每次求出abcd点后，将他们的坐标构建成Eigen::Vector3d数据，添加残差块：
    Eigen::Vector3d curr_point_a(laserCloudIn_plane.points[i].x,
                                 laserCloudIn_plane.points[i].y,
                                 laserCloudIn_plane.points[i].z);

    Eigen::Vector3d last_point_b(laserCloudIn_plane_last.points[closestPointInd].x,
                                 laserCloudIn_plane_last.points[closestPointInd].y,
                                 laserCloudIn_plane_last.points[closestPointInd].z);

    Eigen::Vector3d last_point_c(laserCloudIn_plane_last.points[minPointInd2].x,
                                 laserCloudIn_plane_last.points[minPointInd2].y,
                                 laserCloudIn_plane_last.points[minPointInd2].z);

    Eigen::Vector3d last_point_d(laserCloudIn_plane_last.points[minPointInd3].x,
                                 laserCloudIn_plane_last.points[minPointInd3].y,
                                 laserCloudIn_plane_last.points[minPointInd3].z);

    problem.AddResidualBlock( // 添加残差项
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,4,3>
                    (new CURVE_FITTING_COST(last_point_a,curr_point_b, // 添加残差函数
                                            curr_point_c,curr_point_d)),
                                            loss_function, //核函数
                                            para_q,para_t); //待估计参数

    // 3.执行优化 - 遍历过所有的a点后，就可以优化求解了。
    //所有前一帧里的点都当a点遍历过后，进行优化
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR; // DENSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 5;//迭代数
    options.minimizer_progress_to_stdout = true;//进度是否发到STDOUT

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // 输出结果
    cout << summary.BriefReport() << endl;
    cout << "estimated para_q ， para_t = ";
//    for (auto a:para_q) cout << a << " ";
    cout << endl;

#elif optional_method==01
    cout<< "*********** Ceres-eg ***********"<<endl;
    double para_Pose[7]; //待估计的参数
    para_Pose[0] = 0.0;
    para_Pose[1] = 0.0;
    para_Pose[2] = 0.0;
    para_Pose[6] =  1.0;
    para_Pose[3] =  0.0;
    para_Pose[4] =  0.0;
    para_Pose[5] =  0.0;


    int kNumObservations = cur_pts.size();
    double invDepth[kNumObservations][1];

    ceres::LossFunction *loss_function; // 核函数
    //loss_function = new ceres::HuberLoss(1.0);
    loss_function = new ceres::CauchyLoss(1.0);// 柯西核函数

    ceres::LocalParameterization *local_parameterizationP = new PoseLocalParameterization();
    ceres::Problem::Options problem_options;
    ceres::Problem problem;
    problem.AddParameterBlock(para_Pose, 7, local_parameterizationP);//对Pose重新参数化 || 添加参数块-待优化参数

    for (int i = 0; i < kNumObservations; ++i) {
        invDepth[i][0] = 1;
        problem.AddParameterBlock(invDepth[i], 1); //对深度重新参数化 || 添加参数块-深度信息
        if (!invdepths.empty()&&invdepths[i]>0){
            // cout << "depth observations "<< 1./invdepths[i] <<" "<< invdepths[i] <<endl;
            invDepth[i][0] = invdepths[i];
            problem.SetParameterBlockConstant(invDepth[i]);//把任何参数块设为常数，并且使用SetParameterBlockVariable()来撤销这一操作

            ceres::CostFunction *f_d;
            //自动求导方法，AutoDiffCostFunction
            f_d = new ceres::AutoDiffCostFunction<autoIvDepthFactor, 1,1>(
                    new autoIvDepthFactor(invdepths[i]) ); // 求残差函数
            problem.AddResidualBlock(f_d, loss_function, invDepth[i]); // 这里优化的是深度信息
        }

        ceres::CostFunction *f;
        f = new ceres::AutoDiffCostFunction<autoMonoFactor, 3,7,1>( // 像素去畸变的残差函数
                new autoMonoFactor(Vector3d(un_prev_pts[i].x, un_prev_pts[i].y, 1),Vector3d(un_cur_pts[i].x, un_cur_pts[i].y, 1)) );

        problem.AddResidualBlock(f, loss_function, para_Pose, invDepth[i]); //优化的是重投影误差
    }

    ceres::Solver::Options options;
// options.max_num_iterations = 7;
    options.linear_solver_type = ceres::DENSE_SCHUR; // 稠密schur
    options.trust_region_strategy_type = ceres::DOGLEG; //狗腿法
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;

    TicToc solveTime;
    ceres::Solve(options, &problem, &summary);

#elif optional_method==02
    cout<< "*********** Ceres-test ***********"<<endl;
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值

    double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
    int N = 1000;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值
//    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                                 // OpenCV随机数产生器

    vector<double> x_data, y_data;      // 生成带噪声的数据
    for (int i = 0; i < N; i++) {
        double x = 1.0 * i / N ;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    double abc[3] = {ae, be, ce};
///////////////////////////////////////////////////////【开证】
    ceres::Problem problem; // 创建优化问题 + 损失核函数
    ceres::LossFunction *loss_function;
    loss_function = new ceres::CauchyLoss(1.0); //柯西核函数
//    problem.AddParameterBlock(abc, 3); // 添加优化变量
    for (int i = 0; i < x_data.size(); ++i) {

        problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<CURVE_FITTING_COST2 ,1,3>(
                        new CURVE_FITTING_COST2(x_data[i],y_data[i]) ),
                loss_function,
                abc);
    }

    // 配置求解器 + 执行优化
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now(); // time
    ceres::Solve(options,&problem,&summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);// time
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;// time
    // 输出结果
    cout << summary.BriefReport() << endl;
    cout << "estimated a,b,c = ";
    for (auto a:abc) cout << a << " ";
    cout << endl;
#endif

}