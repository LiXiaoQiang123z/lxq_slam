//
// Created by lxq on 2022/3/18.
//

#include "myslam/common.h"

/**TODO:章节代码选择
 * c: C++部分
 * 00-模板
 * ch3
 * 31:eigen基本使用 + 几何模块的使用
 * 32：坐标变换例子 + 可视化轨迹演示
 * ch4 ... 第一个字符=章节 ； 第二个字符=该章不同程序的分割
 * 41: sophus
 * 42：TrajectoryError
 * 51： opencv基础操作 + 去畸变
 * 52： 双目相机
 * 53： RGB-D相机
 * 61
 * 62  - G2O
 *
 */
#define optional_ch 41

/*****TODO: slam十四讲参数*****/
// ch3 - trajectory
#define Matrix_SIZE 50
std::string trajectory_file = "../data/trajectory.txt";
void DrawTrajectory(std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>);
// ch4 groundtruth_file estimated_file
std::string ch4_groundtruth_file = "../data/ch4-groundtruth.txt";
std::string ch4_estimated_file = "../data/ch4-estimated.txt";
typedef std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);
TrajectoryType ReadTrajectory(const std::string &path);
// ch5 file path
std::string ch5_ubuntu_png = "../data/ch5_ubuntu.png";
std::string ch5_distorted_png = "../data/ch5_distorted.png";
std::string ch5_left_file = "../data/ch5_left.png";
std::string ch5_right_file = "../data/ch5_right.png";
std::string ch5_pose = "../data/ch5_pose.txt";
// 在pangolin中画图，已写好，无需调整
void showPointCloud( const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &pointcloud);
typedef Eigen::Matrix<double, 6, 1> Vector6d;
// 在pangolin中画图，已写好，无需调整
void showPointCloud(const std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char** argv){

#if optional_ch==31
    //TODO:**************** 1.eigen基本使用 ***************
    std::cout<<"**************** 1.eigen基本使用 ***************"<<std::endl;
    Eigen::Matrix<float,2,3> matrix_23;
    Eigen::Vector3d vd_3d;
    Eigen::Matrix<float,3,1> vf_3d;

    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, Eigen::Dynamic,Eigen::Dynamic> matrix_dynamic;
    Eigen::MatrixXd matrix_xd;

    // 1.1 对eigen矩阵的基本操作
    matrix_23 << 1, 2, 3, 4, 5, 6;
    std::cout << "matrix 2x3 from 1 to 6: \n" << matrix_23 << std::endl; //输出矩阵

    std::cout << "print matrix 2x3: " << std::endl;
    for (int i = 0; i < 2; ++i) { // 遍历
        for (int j = 0; j < 3; ++j) {
            std::cout<<matrix_23(i,j)<<std::endl;
        }
    }

    vd_3d << 3,2,1;
    vf_3d << 4,5,6;
    // 不同类型之间不能运算，需要显式的转换
    Eigen::Matrix<double,2,1> result = matrix_23.cast<double>() * vd_3d;
    std::cout << "[1,2,3;4,5,6]*[3,2,1]=" << result.transpose() << std::endl;
    Eigen::Matrix<float,2,1> result2 = matrix_23*vf_3d;
    std::cout << "[1,2,3;4,5,6]*[4,5,6]: " << result2.transpose() << std::endl;

    // 矩阵的维度也不能搞错
//    Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * vd_3d; // error: static assertion failed: YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES

    // 1.2 一些矩阵的基本运算
    matrix_33 = Eigen::Matrix3d::Random();
    std::cout << "random matrix: \n" << matrix_33 <<std::endl;
    std::cout;std::cout << "transpose: \n" << matrix_33.transpose() << std::endl;      // 转置
    std::cout << "sum: " << matrix_33.sum() << std::endl;            // 各元素和
    std::cout << "trace: " << matrix_33.trace() << std::endl;          // 迹
    std::cout << "times 10: \n" << 10 * matrix_33 << std::endl;               // 数乘
    std::cout << "inverse: \n" << matrix_33.inverse() << std::endl;        // 逆
    std::cout << "det: " << matrix_33.determinant() << std::endl;    // 行列式

    // 1.3 特征值 : A^T*A 实对称矩阵
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solve(matrix_33.transpose() * matrix_33); // 计算矩阵的特征值和特征向量
    std::cout << "Eigen values = \n" << eigen_solve.eigenvalues() << std::endl;
    std::cout << "Eigen vectors = \n" << eigen_solve.eigenvectors() << std::endl;

    // 1.4 解方程
    // 我们求解 matrix_NN * x = v_Nd 这个方程 // N的大小在前边的宏里定义，它由随机数生成 // 直接求逆自然是最直接的，但是求逆运算量大
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_NN
            = Eigen::MatrixXd::Random(Matrix_SIZE,Matrix_SIZE);
    matrix_NN = matrix_NN * matrix_NN.transpose(); //保证半正定
    Eigen::Matrix<double, Eigen::Dynamic, 1> v_Nd = Eigen::MatrixXd::Random(Matrix_SIZE,1);

    // 1.4.1.直接求解
    clock_t time1 = clock();
    Eigen::Matrix<double,Eigen::Dynamic,1> x = matrix_NN.inverse() * v_Nd;
    std::cout << "time of normal inverse is "
         << 1000 * (clock() - time1) / (double) CLOCKS_PER_SEC << "ms" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;

    // 1.4.2 利用QR、cholesky、SVD来分解
    time1 = clock();
    x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
    std::cout << "time of normal inverse is "
              << 1000 * (clock() - time1) / (double) CLOCKS_PER_SEC << "ms" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;

    time1 = clock();
    x = matrix_NN.ldlt().solve(v_Nd);
    std::cout << "time of normal inverse is "
              << 1000 * (clock() - time1) / (double) CLOCKS_PER_SEC << "ms" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;

    time1 = clock();
    x = matrix_NN.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(v_Nd); // A.bdcSvd(ComputeThinU | ComputeThinV).solve(b)
    std::cout << "time of normal inverse is "
              << 1000 * (clock() - time1) / (double) CLOCKS_PER_SEC << "ms" << std::endl;
    std::cout << "x = " << x.transpose() << std::endl;

//TODO: **************** 2.eigen几何模块的使用 ***************
    std::cout<<"**************** 2.eigen几何模块的使用 ***************"<<std::endl;
    // 2.1 旋转矩阵R和旋转向量 AngleAxisd
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();//单位矩阵吧
//    std::cout<<"R: "<<rotation_matrix<<std::endl;
    Eigen::AngleAxisd rotation_vector(M_PI/4, Eigen::Vector3d(0,0,1));//初始化为绕z轴旋转pi/4
    std::cout.precision(3); //精确度，到小数点后几位
    std::cout << "rotation matrix =\n" << rotation_vector.matrix() << std::endl;
    rotation_matrix = rotation_vector.toRotationMatrix(); // 赋值
    // 旋转向量进行坐标变换
    Eigen::Vector3d v(1,0,0);
    Eigen::Vector3d v_rotated = rotation_vector * v;
    std::cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << std::endl;
    // 或者用旋转矩阵
    v_rotated = rotation_matrix * v;
    std::cout << "(1,0,0) after rotation (by matrix) = " << v_rotated.transpose() << std::endl;

    //2.2 欧拉角
    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0);// 210-ZYX的顺序
    std::cout << "yaw pitch roll = " << euler_angles.transpose() << std::endl; //返回的范围弧度 ： [0:pi]x[-pi:pi]x[-pi:pi].

    //2.3 欧式变换矩阵 T
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
//    std::cout<<"T: "<< T.matrix() <<std::endl;
    T.rotate(rotation_matrix);
//    T.rotate(rotation_vector);
    T.pretranslate(Eigen::Vector3d(1,3,4));
    Eigen::Vector3d  v_transformed = T * v; // R*v + t
    std::cout << "v tranformed = " << v_transformed.transpose() << std::endl;

    // 2.4 四元数
    Eigen::Quaterniond q = Eigen::Quaterniond(rotation_vector);
    std::cout << "quaternion from rotation vector = " << q.coeffs().transpose()
         << std::endl;   // 请注意【coeffs】的顺序是(x,y,z,w),w为实部，前三者为虚部
    q = Eigen::Quaterniond(rotation_matrix);
    std::cout << "quaternion from rotation vector = " << q.coeffs().transpose() << std::endl;
    Eigen::Quaterniond q2 = Eigen::Quaterniond(0,0,1,1); // Quaterniond（w,x,y,z）
    std::cout << "quaternion from rotation vector = " << q2.coeffs().transpose() << std::endl;
    // 使用四元数旋转向量
    v_rotated = q * v; // qpq^{-1}
    std::cout << "(1,0,0) after rotation = " << v_rotated.transpose() << std::endl;
    // 用常规向量乘法表示，则应该如下计算
    std::cout << "should be equal to " << (q * Eigen::Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << std::endl;

#elif optional_ch==32
    //TODO: **************** 3. 坐标变换  ***************
    std::cout<<"**************** 3. 坐标变换 ***************"<<std::endl;
    // 条件
    Eigen::Quaterniond q1(0.35,0.2,0.3,0.1), q2(-0.5,0.4,-0.1,0.2);
    q1.normalize(); //注意：四元数需要归一化处理的
    q2.normalize();
    Eigen::Vector3d t1(0.3,0.1,0.1), t2(-0.1,0.5,0.3);
    Eigen::Vector3d p1(0.5,0,0.2);

    /*
     * q，t 表示的是 T_Rk,w :世界坐标系到机器人坐标系
     * 所求P_R2 = T_R2,w * T_w,R1 * p_R1
     */
    Eigen::Isometry3d T_R1_w(q1),T_R2_w(q2);
    T_R1_w.pretranslate(t1);
    T_R2_w.pretranslate(t2);
    Eigen::Vector3d p2 = T_R2_w * T_R1_w.inverse() * p1;
    std::cout<<p2.transpose()<<std::endl; //结果：[-0.0309731    0.73499   0.296108]^T


    //TODO: **************** 4. 绘制轨迹  ***************
    std::cout<<"**************** 4. 绘制轨迹 ***************"<<std::endl;
    std::vector<Eigen::Isometry3d,Eigen::aligned_allocator<Eigen::Isometry3d>> poses;
    // aligned_allocator
    // 与需要非标准对齐的类型一起使用的 STL 兼容分配器。
    // 内存按照动态对齐的矩阵/数组类型（例如 MatrixXd）对齐。
    std::ifstream fin(trajectory_file);
    if(!fin){
        std::cout << "cannot find trajectory file at " << trajectory_file << std::endl;
        return 1;
    }

    while (!fin.eof()){ //数据遍历读取
        double time,tx,ty,tz,qx,qy,qz,qw;
        fin>>time >> tx >> ty >> tz >> qx >> qy >> qz >> qw; //数据读入
        Eigen::Isometry3d Twr(Eigen::Quaterniond(qw,qx,qy,qz));
        Twr.pretranslate(Eigen::Vector3d(tx,ty,tz));
        poses.push_back(Twr); //数据保存到poses中
    }
    std::cout << "read total " << poses.size() << " pose entries" << std::endl;

    // draw trajectory in pangolin
    DrawTrajectory(poses);
    return 0;

#elif optional_ch==41
    //TODO: **************** 4.Sophus-李群和李代数 ***************
    std::cout<<"**************** 4.Sophus-三维旋转部分  ***************"<<std::endl;
    Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI_2, Eigen::Vector3d(0,0,1)).toRotationMatrix(); // 旋转向量-矩阵
    Eigen::Quaterniond q(R);
    // 1.李群矩阵
    Sophus::SO3d SO3_R(R);
    Sophus::SO3d SO3_q(q);
    std::cout << "R:\n" << R << std::endl;
    std::cout << "SO(3) from matrix:\n" << SO3_R.matrix() << std::endl;
    std::cout << "SO(3) from quaternion:\n" << SO3_q.matrix() << std::endl;
    std::cout << "they are equal" << std::endl;

    //2.李代数获取|| 李代数部分的运算
    Eigen::Vector3d so3 = SO3_R.log();
    std::cout<< "so3 = " <<so3.transpose()<<std::endl;
    // hat 为向量到反对称矩阵【待记】
    std::cout<< "so3 hat = " <<Sophus::SO3d::hat(so3)<<std::endl;
    // 相对的，vee为反对称到向量 【待记】
    std::cout<< "so3 hat vee = " <<Sophus::SO3d::vee (Sophus::SO3d::hat(so3)).transpose()<<std::endl;

    //3.增量扰动模型的更新
    Eigen::Vector3d update_so3(1e-4,0,0);
    Sophus::SO3d SO3_updata = Sophus::SO3d::exp(update_so3) * SO3_R;
    std::cout<< "SO3 update" << SO3_updata.matrix()<<std::endl;

    std::cout<<"**************** 4.Sophus-三维变换部分  ***************"<<std::endl;
    // 对SE3来说，操作大同小异
    // 1.李群
    Eigen::Vector3d t(1,0,0);
    Sophus::SE3d SE3_Rt(R,t);
    Sophus::SE3d SE3_qt(q,t);
    std::cout << "SE3 from R,t= \n" << SE3_Rt.matrix() << std::endl;
    std::cout << "SE3 from q,t= \n" << SE3_qt.matrix() << std::endl;

    //2.李代数 六维向量
    typedef Eigen::Matrix<double,6,1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    std::cout<<"ser = "<<se3.transpose()<<std::endl;
    // hat vee 同理
    std::cout<<"se3 hat1 = "<<Sophus::SE3d::hat(se3)<<std::endl;
    std::cout<<"se3 hat2 = "<<Sophus::SE3d::hat(se3).matrix()<<std::endl;
    std::cout<<"se3 vee = "<<Sophus::SE3d::vee( Sophus::SE3d::hat(se3)).transpose()<<std::endl;

    // 3.增量扰动的更新
    Vector6d update_se3;
    update_se3.setZero();
    update_se3(0,0) = 1e-4;
    Sophus::SE3d SE3_update = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    std::cout<<"SE3 update"<<SE3_update.matrix()<<std::endl;

#elif optional_ch==42
    //TODO: **************** 4.TrajectoryError***************
    std::cout<<"**************** 4.TrajectoryError ***************"<<std::endl;
    //读取文件
    TrajectoryType groundtruth = ReadTrajectory(ch4_groundtruth_file);
    TrajectoryType estimated = ReadTrajectory(ch4_estimated_file);
    assert(!groundtruth.empty() && !estimated.empty());
    assert(groundtruth.size() == estimated.size());

    // compute RMSE
    double rmse = 0;
    for (size_t i = 0; i < estimated.size(); ++i) {
        Sophus::SE3d p1 = estimated[i], p2 = groundtruth[i]; // SE3d=变换矩阵T(R,t)
        double error = (p2.inverse() * p1).log().norm(); //矩阵->向量->norm取大小
        rmse += error; //求和
    }
    rmse = sqrt( rmse / (double)(estimated.size()) ); //求平均->方根 （求均方根）
    std::cout<<"RMSE = "<<rmse<<std::endl;

    //绘图
    DrawTrajectory(groundtruth,estimated);

#elif optional_ch==51
    //TODO: **************** 51.opencv基础操作 + 去畸变***************
    std::cout<<"**************** 51.opencv基础操作 + 去畸变  ***************"<<std::endl;
    cv::Mat image;
    image = cv::imread(ch5_ubuntu_png);
    if(image.data == nullptr){
        std::cerr<<"file "<<ch5_ubuntu_png<<"不存在"<<std::endl;
        return 0;
    }

    // 0.图片文件基本信息
    std::cout<<"图像宽为" << image.cols << ",高为" << image.rows << ",通道数为" << image.channels() << std::endl;
    cv::imshow("image",image);
    cv::waitKey(0); // 暂停程序，等待一个按键

    // 判断图片的类型
    if(image.type() != CV_8UC1 && image.type() != CV_8UC3 ){ // TODO：8UC1 8UC3后面的数字等于通道数
        // 图像类型不符合要求
        std::cout << "请输入一张彩色图或灰度图." << std::endl;
        return 0;
    }

    // 1. 遍历图像
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < image.rows; ++i) { //行
        unsigned char *row_ptr = image.ptr<unsigned char >(i); // 第i行的头指针
        for (size_t j = 0; j < image.cols; ++j) { //列
            unsigned char *data_ptr = &row_ptr[j * image.channels()]; // 读取对应[i，j]的数据 || x * image.channels()表示单通道一个一个过，三通道一下跳3个 ||
            for (int k = 0; k != image.channels() ; ++k) { //通道值
                unsigned char data = data_ptr[k];
            }
        }
    }
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2-t1);
    std::cout << "遍历图像用时：" << time_used.count() << " 秒。" << std::endl;

    // 2.cv::Mat的拷贝
    cv::Mat image_another = image; // 直接赋值并不会拷贝，而是都指向图片(类似引用把)
    image_another(cv::Rect(0,0,100,100)).setTo(0); // Rect:矩阵块 将0,0起始处100，100大小的块置为零
    cv::imshow("image",image);
    cv::waitKey(0);

    // 使用clone函数拷贝
    cv::Mat image_clone = image.clone();
    image_clone(cv::Rect(0,0,100,100)).setTo(255); //255代表白色
    cv::imshow("image_2",image);
    cv::imshow("image_clone",image_clone);
    cv::waitKey(0);

    // TODO:对于图像还有很多基本的操作,如【剪切,旋转,缩放等】,限于篇幅就不一一介绍了,请参看OpenCV官方文档查询每个函数的调用方法.
    cv::destroyAllWindows();

    std::cout<<"**************** 51.去畸变  ***************"<<std::endl;
    //畸变参数
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // 内参
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    // 0.获取图像信息
    cv::Mat image_d = cv::imread(ch5_distorted_png,cv::IMREAD_GRAYSCALE); //TODO：用灰度图方式去读取
    if(image_d.data == nullptr){
        std::cerr<<"file "<<ch5_distorted_png<<"不存在"<<std::endl;
        return 0;
    }
    int rows = image_d.rows, cols = image_d.cols;
    cv::Mat image_ud = cv::Mat(rows, cols, CV_8UC1);   // 去畸变以后的图


    // 1.计算去畸变信息 [u,v] [x,y] [cols,rows]
    for (int v = 0; v < rows; ++v) { //v- 行
        for (int u = 0; u < cols; ++u) { //u-x-cols 列
            // 像素-归一化坐标系
            double x = (u - cx) / fx, y = (v-cy) / fy;
            // 计算畸变
            double r = sqrt(x*x + y*y), r_2 = r*r, r_4=r_2*r_2;
            double x_dis = x*(1 + k1*r_2 + k2 * r_4) + 2*p1*x*y + p2*(r_2 + 2*x*x);
            double y_dis = y*(1 + k1*r_2 + k2 * r_4) + p1*(r_2 + 2*y*y) + 2*p2*x*y;
            //归一化-像素
            double u_dis = fx * x_dis + cx;
            double v_dis = fy * y_dis + cy;

            // 赋值：
            if(u_dis>=0 && v_dis>=0 && u_dis < cols && v_dis < rows){
                // 去畸变后的坐标 -> 去畸变图像 【u v还需要使用，u_dis等必须定义】
                image_ud.at<uchar>(v,u) = image_d.at<uchar>((int)v_dis, (int)u_dis);
            } else{
                image_ud.at<uchar>(v,u) = 0;  //边界外的给 黑点
            }
        }
    }

    // 2.可视化效果
    cv::imshow("image_dis",image_d);
    cv::imshow("image_undis",image_ud);
    cv::waitKey(0);

#elif optional_ch==52
    //TODO: **************** 52 双目操作 ***************
    std::cout<<"**************** 52 双目操作：从左右目图像出发，计算图像对应的视差图，然后计算像素在相机坐标系下的坐标，它们将构成点云  ***************"<<std::endl;
    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    double b = 0.573; // 基线
    //0.图像信息
    cv::Mat left = cv::imread(ch5_left_file, cv::IMREAD_GRAYSCALE); //灰度图读取
    cv::Mat right = cv::imread(ch5_right_file,cv::IMREAD_GRAYSCALE);
    cv::imshow("left",left);
    cv::imshow("right",right);

    //TODO: SGBM[???] 1.sgbm生成视差图
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
            0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    //OpenCV算法实现的【SGBM】神奇的参数
    cv::Mat disparity_sgbm, disparity; //视差
    sgbm->compute(left,right,disparity_sgbm); //计算视差
    // 计算距离的视差图（CV_32F）和用于肉眼看的视差图（CV_8U）使用的格式不同，并且用于计算的视差图无需进行裁剪和归一化
    // 所以在sgbm得到视差的基础上，在除以 16
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f); //

    // 生成点云
    std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> pointcloud; //aligned_allocator对齐操作

    // 如果你的机器慢，请把后面的v++和u++改成v+=2, u+=2
    // 2.【目的：像素->相机坐标 => 点云】
    for (int v = 0; v < left.rows; ++v)
        for (int u = 0; u < left.cols; ++u) {
            // 去除外点
            if(disparity.at<float>(v,u)<=0.0 || disparity.at<float>(v,u) >= 96.0) continue;

            Eigen::Vector4d point(0,0,0, left.at<uchar>(v,u) / 255.0); // 第四维为灰度值比重把
            double x = (u - cx) / fx; //归一化坐标系
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v,u)) ;
            point[0] = x * depth; // 相机坐标系
            point[1] = y * depth;
            point[2] = depth;

            pointcloud.push_back(point);
        }



    cv::imshow("disparity",disparity/96.0); //归一化
    cv::waitKey(0);
    showPointCloud(pointcloud); //绘制点云

#elif optional_ch==53
    //TODO: **************** 53 RGB-D操作 ***************
    std::cout<<"************ 53 1.根据内参计算一对RGBD图像对应的点云（像素-相机(世界)生成点云）"
               "2.根据个张图的相机位姿（外参），把点云加起来，组成地图***********"<<std::endl;
    std::vector<cv::Mat> colorImgs, depthImgs; //彩色图 + 深度图
    TrajectoryType poses;

    std::ifstream fin(ch5_pose);
    if(!fin){
        std::cerr << "请在有pose.txt的目录下运行此程序" << std::endl;
        return 1;
    }

    // 0.读取参数 TODO: 多个图像的读取方法
    for (int i = 0; i < 5; ++i) {
        boost::format fmt("../data/%s/%d.%s"); //图像文件格式 || ./字符串/数字.字符串
        colorImgs.push_back(cv::imread((fmt % "ch5_color" % (i + 1) % "png").str()));//读取./color/1-5.png documents
        depthImgs.push_back(cv::imread((fmt % "ch5_depth" % (i + 1) % "pgm").str(), -1));// 使用-1读取原始图像
        if (colorImgs.empty() || depthImgs.empty()) {
            std::cerr << "数据为空" << std::endl;
        }

        double data[7] = {0};
        for (auto &d:data)
            fin >> d; //读取参数
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),//【四元数、t向量 || 变换矩阵T】
                          Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(pose);
    }
    // 1.计算点云并拼接
    // 相机内参
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
    double depthScale = 1000.0; // 归一化范围
    std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000); // 预留空间 || reserve是容器预留空间，但在空间内不真正创建元素对象，所以在没有添加新的对象之前，不能引用容器内的元素

    for (int i = 0; i < 5; ++i) {
        std::cout << "转换图像中: " << i + 1 << std::endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Sophus::SE3d T = poses[i]; // Twc
        for (int v = 0; v < color.rows; ++v)
            for (int u = 0; u < color.cols; ++u) {
                unsigned int d = depth.ptr<unsigned short >(v)[u]; //深度值
                if(d == 0) continue;
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale; // 类似以归一化
                point[0] = (u - cx) * point[2] / fx; //相机坐标系(内参)
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point; // Twc 世界坐标系(外参)

                Vector6d p;
                p.head<3>() = pointWorld; // 世界坐标
                p[5] = color.data[v * color.step + u * color.channels()]; // bule - green - red
                p[4] = color.data[v * color.step + u * color.channels() + 1];
                p[3] = color.data[v * color.step + u * color.channels() + 2];
                pointcloud.push_back(p);
            }
    }

    std::cout<< "点云共有" << pointcloud.size() << "个点." << std::endl;
    showPointCloud(pointcloud);

#elif optional_ch==62
    /**
 * BA Example
 * Author: Xiang Gao
 * Date: 2016.3
 * Email: gaoxiang12@mails.tsinghua.edu.cn
 *
 * 在这个程序中，我们读取两张图像，进行特征匹配。然后根据匹配得到的特征，计算相机运动以及特征点的位置。这是一个典型的Bundle Adjustment，我们用g2o进行优化。
 */

// for std
#include <iostream>
// for opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/concept_check.hpp>
// for g2o
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>


using namespace std;

// 寻找两个图像中的对应点，像素坐标系
// 输入：img1, img2 两张图像
// 输出：points1, points2, 两组对应的2D点
int     findCorrespondingPoints( const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2 );

// 相机内参
double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;

int main( int argc, char** argv )
{
    // 调用格式：命令 [第一个图] [第二个图]
    if (argc != 3)
    {
        cout<<"Usage: ba_example img1, img2"<<endl;
        exit(1);
    }

    // 读取图像
    cv::Mat img1 = cv::imread( argv[1] );
    cv::Mat img2 = cv::imread( argv[2] );

    // 找到对应点
    vector<cv::Point2f> pts1, pts2;
    if ( findCorrespondingPoints( img1, img2, pts1, pts2 ) == false )
    {
        cout<<"匹配点不够！"<<endl;
        return 0;
    }
    cout<<"找到了"<<pts1.size()<<"组对应特征点。"<<endl;
    // 构造g2o中的图
    // 先构造求解器
    g2o::SparseOptimizer    optimizer;
    // 使用Cholmod中的线性方程求解器
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new  g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> ();
    // 6*3 的参数
    g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3( linearSolver );
    // L-M 下降
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg( block_solver );

    optimizer.setAlgorithm( algorithm );
    optimizer.setVerbose( false );

    // 添加节点
    // 两个位姿节点
    for ( int i=0; i<2; i++ )
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if ( i == 0)
            v->setFixed( true ); // 第一个点固定为零
        // 预设值为单位Pose，因为我们不知道任何信息
        v->setEstimate( g2o::SE3Quat() );
        optimizer.addVertex( v );
    }
    // 很多个特征点的节点
    // 以第一帧为准
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId( 2 + i );
        // 由于深度不知道，只能把深度设置为1了
        double z = 1;
        double x = ( pts1[i].x - cx ) * z / fx;
        double y = ( pts1[i].y - cy ) * z / fy;
        v->setMarginalized(true);
        v->setEstimate( Eigen::Vector3d(x,y,z) );
        optimizer.addVertex( v );
    }

    // 准备相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );

    // 准备边
    // 第一帧
    vector<g2o::EdgeProjectXYZ2UV*> edges;
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)) );
        edge->setMeasurement( Eigen::Vector2d(pts1[i].x, pts1[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }
    // 第二帧
    for ( size_t i=0; i<pts2.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(1)) );
        edge->setMeasurement( Eigen::Vector2d(pts2[i].x, pts2[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0,0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }

    cout<<"开始优化"<<endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    cout<<"优化完毕"<<endl;

    //我们比较关心两帧之间的变换矩阵
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
    Eigen::Isometry3d pose = v->estimate();
    cout<<"Pose="<<endl<<pose.matrix()<<endl;

    // 以及所有特征点的位置
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2));
        cout<<"vertex id "<<i+2<<", pos = ";
        Eigen::Vector3d pos = v->estimate();
        cout<<pos(0)<<","<<pos(1)<<","<<pos(2)<<endl;
    }

    // 估计inlier的个数
    int inliers = 0;
    for ( auto e:edges )
    {
        e->computeError();
        // chi2 就是 error*\Omega*error, 如果这个数很大，说明此边的值与其他边很不相符
        if ( e->chi2() > 1 )
        {
            cout<<"error = "<<e->chi2()<<endl;
        }
        else
        {
            inliers++;
        }
    }

    cout<<"inliers in total points: "<<inliers<<"/"<<pts1.size()+pts2.size()<<endl;
    optimizer.save("ba.g2o");
    return 0;
}


int     findCorrespondingPoints( const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2 )
{
    cv::ORB orb;
    vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desp1, desp2;
    orb( img1, cv::Mat(), kp1, desp1 );
    orb( img2, cv::Mat(), kp2, desp2 );
    cout<<"分别找到了"<<kp1.size()<<"和"<<kp2.size()<<"个特征点"<<endl;

    cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create( "BruteForce-Hamming");

    double knn_match_ratio=0.8;
    vector< vector<cv::DMatch> > matches_knn;
    matcher->knnMatch( desp1, desp2, matches_knn, 2 );
    vector< cv::DMatch > matches;
    for ( size_t i=0; i<matches_knn.size(); i++ )
    {
        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
            matches.push_back( matches_knn[i][0] );
    }

    if (matches.size() <= 20) //匹配点太少
        return false;

    for ( auto m:matches )
    {
        points1.push_back( kp1[m.queryIdx].pt );
        points2.push_back( kp2[m.trainIdx].pt );
    }

    return true;
}
#endif
    return 0;
}

/*********************************** ch3 : DrawTrajectory************************************/
void DrawTrajectory(std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses) {
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glLineWidth(2);
        //****************** TODO： start********************
        for (size_t i = 0; i < poses.size(); i++) { // 【开始遍历】【作用是啥子偶？？？】
            // 画每个位姿的三个坐标轴
            Eigen::Vector3d Ow = poses[i].translation();
            Eigen::Vector3d Xw = poses[i] * (0.1 *  Eigen::Vector3d(1, 0, 0)); // 缩小 1/10
            Eigen::Vector3d Yw = poses[i] * (0.1 *  Eigen::Vector3d(0, 1, 0));
            Eigen::Vector3d Zw = poses[i] * (0.1 *  Eigen::Vector3d(0, 0, 1));
            glBegin(GL_LINES);
            glColor3f(1.0, 0.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            glColor3f(0.0, 1.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd();
        }
        // 画出连线
        for (size_t i = 0; i < poses.size(); i++) {
            glColor3f(0.0, 0.0, 0.0);
            glBegin(GL_LINES);
            auto p1 = poses[i], p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        //****************** TODO： end********************
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}
/*********************************** ch4 : DrawTrajectory************************************/
TrajectoryType ReadTrajectory(const std::string &path) {
    std::ifstream fin(path); // 输入路径
    TrajectoryType trajectory;
    if (!fin) {
        std::cerr << "trajectory " << path << " not found." << std::endl;
        return trajectory;
    }

    while (!fin.eof()) { //读取数据
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Sophus::SE3d p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
        trajectory.push_back(p1); //save数据
    }
    return trajectory;
}
void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));


    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glLineWidth(2);
        //***********************************
        // 连线
        for (size_t i = 0; i < gt.size() - 1; i++) {
            glColor3f(0.0f, 0.0f, 1.0f);  // blue for ground truth
            glBegin(GL_LINES);
            auto p1 = gt[i], p2 = gt[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        for (size_t i = 0; i < esti.size() - 1; i++) {
            glColor3f(1.0f, 0.0f, 0.0f);  // red for estimated
            glBegin(GL_LINES);
            auto p1 = esti[i], p2 = esti[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();
        }
        //***********************************
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }

}
/*********************************** ch5 showPointCloud ************************************/
void showPointCloud(const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>> &pointcloud) {

    if (pointcloud.empty()) {
        std::cerr << "Point cloud is empty!" << std::endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        //************************[上色吗？？]*************************
        for (auto &p: pointcloud) {
            glColor3f(p[3], p[3], p[3]);
            glVertex3d(p[0], p[1], p[2]);
        }

        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}
void showPointCloud(const std::vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {

    if (pointcloud.empty()) {
        std::cerr << "Point cloud is empty!" << std::endl;
        return;
    }

    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
            pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        //**************************************
        for (auto &p: pointcloud) {
            glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0); // /255
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
    return;
}

