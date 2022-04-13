//
// Created by lxq on 2021/12/10.
//
/****************** third .hpp **********************/
// #include <opencv2/imgproc.hpp> // imread 等
/****************** my .hpp **********************/
#include "myslam/common.h"
#include "myslam/orb_self.h"
#include "myslam/bundleAdjustment.h"
#include "myslam/BA-g2o.h"

/*******| 定义全局变量 + 函数/函数声明 |***********/
//【可能需要修改的参数-调参】
// 图片-所在位置
std::string first_file = "../data/1.png";
std::string second_file = "../data/2.png";
std::string depth_file = "../data/1_depth.png";


// BA最大迭代次数
const int iterations = 5;

//  相机内参
Eigen::Matrix3d K_cam;
double fx = 520.9; //这一组是 十四讲ch5 照片对应相机内参
double fy = 521.0;
double cx = 325.1;
double cy = 249.7;
/******************************* 手写特征-可视化 ******************************/
// 手写特征 - 变量共享
self_orb_con orb_data;

 /** 手写orb画特征点，以及匹配点连线部分
  *
  * @param img1_color
  * @param img2_color
  */
 void show_orb(
     Mat img1_color,
     Mat img2_color)
 {
     // 画出特征点|| 画点
     for (auto &l : orb_data.keypoints1)
     { //画出特征点
         cv::Point p;
         p.x = l.x_;
         p.y = l.y_;
         /**
            图像	绘制圆圈的图像。
            中央	圆的中心。
            半径	圆的半径。
            颜色	圆圈颜色。
            厚度	圆形轮廓的厚度（如果为正）。负值，如FILLED，表示要绘制一个实心圆。
          */
         cv::circle(img1_color, p, 3, cv::Scalar(0, 0, 255), -1);
     }
     for (auto &l : orb_data.keypoints2)
     {
         cv::Point p;
         p.x = l.x_;
         p.y = l.y_;
         cv::circle(img2_color, p, 3, cv::Scalar(0, 0, 255), -1);
     }

     // ---匹配
     /**
      * @brief 将水平串联应用于给定矩阵。
               该函数水平连接两个或多个cv::Mat矩阵（具有相同的行数）。

        src1	要考虑进行水平连接的第一个输入数组。
        src2	水平连接要考虑的第二个输入数组。
        夏令时	输出数组。它具有与 src1 和 src2 相同的行数和深度，以及 src1 和 src2 的列数之和。
      */
     cv::hconcat(img1_color, img2_color, orb_data.combine); //图像合并
     for (uint i = 0; i < orb_data.matches.size(); ++i)
     {
         cv::Point p1 = cv::Point(orb_data.keypoints1[orb_data.matches[i].id1_].x_, orb_data.keypoints1[orb_data.matches[i].id1_].y_);
         cv::Point p2 = cv::Point(orb_data.keypoints2[orb_data.matches[i].id2_].x_ + 640, orb_data.keypoints2[orb_data.matches[i].id2_].y_); //为啥+640: 并排显示
         cv::line(orb_data.combine, p1, p2, cv::Scalar(200, 0, 0), 1);                                                                       //BGR成像
     }
 }

/******************************* Ceres-构建残差块 ******************************/
/*
 * 输入： 待估计的位姿 q，t ； 2D点point1 + 2D点point2
 * 输出：残差的计算
 */
/*
//template <class T1>
//struct ceres_PnP_32{
//    Eigen::Matrix<T1,3,1> point1;
//    Eigen::Matrix<T1,2,1> point2;
////    Eigen::Matrix<T1,4,1> q_21;
////    Eigen::Matrix<T1,3,1> t_21;
//
//    ceres_PnP_32(    const Eigen::Matrix<T1,3,1> point1_in, // 1.初值获取
//                     const Eigen::Matrix<T1,2,1> point2_in
////                     Eigen::Matrix<T1,4,1> q_21_in = Eigen::Matrix<T1,4,1>(1,0,0,0),
////                     Eigen::Matrix<T1,3,1> t_21_in = Eigen::Matrix<T1,3,1>(0,0,0)
//    ) : point1(point1_in), point2(point2_in){};
//
//    template<class T>
//    bool operator()(const T* q21, const T* t21, T* residual) const{ // 2.残差函数计算
//        Eigen::Quaternion<T> q21_out{(q21[0]),q21[1],q21[2],q21[3]};
//        Eigen::Matrix<T,3,1> t21_out {t21[0], t21[1], t21[2]};
//        Eigen::Matrix<double,3,1>point2_1 = q21_out * point1 + t21_out;
//        residual[0] = point2[0] -  K_cam * point2_1[0] / point1[2];
//        residual[1] = point2[1] -  K_cam * point2_1[1] / point1[2];
//        return true;
//    };
//
//    // 3.生成代价函数
//    static ceres::CostFunction *Create(
//            const Eigen::Matrix<T1,3,1> point1_in, // 1.初值获取
//            const Eigen::Matrix<T1,2,1> point2_in
////            Eigen::Matrix<T1,4,1> q_21_in = Eigen::Matrix<T1,4,1>(1,0,0,0),
////            Eigen::Matrix<T1,3,1> t_21_in = Eigen::Matrix<T1,3,1>(0,0,0)
//    )  {
//        return (new ceres::AutoDiffCostFunction<ceres_PnP_32, 3,2, 4,3>
//                (new ceres_PnP_32(point1_in,point2_in)) //输入
//        );
//    };
//
//};
 */
class ceres_PnP_32{
public:
    ceres_PnP_32(    const Eigen::Matrix<double,3,1> point1_in, // 1.初值获取
                     const Eigen::Matrix<double,2,1> point2_in) :
            point1(point1_in), point2(point2_in){}

    template<class T>
    bool operator()(const T* q21, const T* t21, T* residual) const{ // 2.残差函数计算
        Eigen::Quaternion<T> q21_out{(q21[0]),q21[1],q21[2],q21[3]};
        Eigen::Matrix<T,3,1> t21_out {t21[0], t21[1], t21[2]};
        Eigen::Matrix<double,3,1>point2_1 = q21_out * point1 + t21_out;
//        residual[0] = point2[0] -  (K_cam[0,0] * point2_1[0] / point1[2] + K_cam[0,2]);
//        residual[1] = point2[1] -  (K_cam[1,1] * point2_1[1] / point1[2] + K_cam[1,2]);
        Eigen::Matrix<double,2,3> K_cam23 = K_cam.template block<2,3>(0,0);
        Eigen::Matrix< T, 2, 1 >  residuals( residual );
        residuals.template block< 2, 1 >( 0, 0 ) = point2 - K_cam23 * point2_1.template block<2,1>(0,0) / point2_1[2];
        return true;
    }

    // 3.生成代价函数
    static ceres::CostFunction *Create(
            const Eigen::Matrix<double,3,1> point1_in, // 1.初值获取
            const Eigen::Matrix<double,2,1> point2_in)  {
        return (new ceres::AutoDiffCostFunction<ceres_PnP_32, 3,2, 4,3>
                (new ceres_PnP_32(point1_in,point2_in)) //输入
        );
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
    const Eigen::Matrix<double,3,1> point1;
    const Eigen::Matrix<double,2,1> point2;
    Eigen::Matrix<double,4,1> q_21;
    Eigen::Matrix<double,3,1> t_21;
};


/**
 * @brief main函数：主要的进程
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char **argv)
{

    //------------data 读取-------------
    /**
     * @brief imread ：这里为了简化采用灰度图。
     * IMREAD_UNCHANGED =-1 : 不变-原样返回加载图像
     * IMREAD_GRAYSCALE = 0 ：但通道灰度图像
     * IMREAD_COLOR = 1 ： 3通道BGR彩色图像
     */
    cv::Mat img_1_gray = cv::imread(first_file, cv::IMREAD_GRAYSCALE);
    cv::Mat img_2_gray = cv::imread(second_file, cv::IMREAD_GRAYSCALE);
    cv::Mat img_1 = cv::imread(first_file, cv::IMREAD_COLOR);
    cv::Mat img_2 = cv::imread(second_file, cv::IMREAD_COLOR);
    cv::Mat img_depth = cv::imread(depth_file, cv::IMREAD_COLOR);
    assert(img_1_gray.data != nullptr && img_2_gray.data != nullptr);
    assert(img_1.data != nullptr && img_2.data != nullptr);

    // --- 设置相机内参
    K_cam(0, 0) = fx;
    K_cam(1, 1) = fy;
    K_cam(0, 2) = cx;
    K_cam(1, 2) = cy;
    K_cam(2, 2) = 1.0;
    /********************************************************************/
    //-------ch5-手写特征提取+匹配 orb-self----------TODO: 手写特征对灰度图进行处理
#if (Optional_Program&1) == 0
    self_orb(
            img_1_gray,
            img_2_gray,
            img_depth,
            orb_data);

    //对特征进行描点 + 匹配对连线
    show_orb(img_1, img_2);
    cv::imshow("img1_color", img_1);
    cv::imshow("img2_color", img_2);
    cv::imshow("combine", orb_data.combine);
    //    cv::imwrite("../data/combine_gray.png",combine);
    cv::waitKey(0);

    std::cout << "+------------------------借助opencv实现特征提取+匹配 orb-c--------------------------" << std::endl;
    /********************************************************************/
#elif   (Optional_Program&1) == 1
   //-------借助opencv实现特征提取+匹配 orb-cv----------
   //---初始化
   std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
   cv::Mat descriptors_1, descriptors_2;
   cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(500);
   cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
   cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

   //---第一步：
   auto time1 = (double)cv::getTickCount();
   detector->detect(img_1, keypoints_1);
   detector->detect(img_2, keypoints_2);
   std::cout << "ORB keypoint1: " << keypoints_1.size() << " "
        << "ORB keypoint2: " << keypoints_2.size() << std::endl;
   //-- 第二步:根据角点位置计算 BRIEF 描述子
   descriptor->compute(img_1, keypoints_1, descriptors_1);
   descriptor->compute(img_2, keypoints_2, descriptors_2);
   //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
   std::vector<cv::DMatch> matches;
   matcher->match(descriptors_1, descriptors_2, matches);

   //-- 第四步:匹配点对筛选
   double min_dist = 10000, max_dist = 0;
   //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
   for (int i = 0; i < descriptors_1.rows; i++)
   {
       double dist = matches[i].distance;
       if (dist < min_dist)
           min_dist = dist;
       if (dist > max_dist)
           max_dist = dist;
   }
   printf("-- Max dist : %f \n", max_dist);
   printf("-- Min dist : %f \n", min_dist);

   //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
   std::vector<cv::DMatch> good_matches;
   for (int i = 0; i < descriptors_1.rows; i++)
   {
       if (matches[i].distance <= std::max(2 * min_dist, 30.0))
       {
           good_matches.push_back(matches[i]);
       }
   }
   //第五步，绘制结果
   cv::Mat outimg1,outimg2; // 绘制提取的特征点
   cv::drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
   cv::drawKeypoints(img_2, keypoints_2, outimg2, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
   cv::imshow("ORB features1", outimg1);
   cv::imshow("ORB features2", outimg2);

   cv::Mat img_match; // 绘制匹配结果
   cv::Mat img_goodmatch;
   printf("ORB detect cost %f ms \n", (1000 * ((double )cv::getTickCount() - time1) / cv::getTickFrequency()));
   std::cout << "good_match = " << good_matches.size() << std::endl;
   cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
   cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
   // imshow("all matches", img_match);
   cv::imshow("good matches", img_goodmatch);
   cv::waitKey(0);

  //---2D，3D点 求取
  std::vector<cv::Point2d> points1, points2;
  std::vector<cv::DMatch> good_matches2;
  std::vector<cv::Point3d> points1_3d;
  for (auto & good_matche : good_matches)
  {
      //深度数据 不为空 nan inf
      if (img_depth.at<ushort>(keypoints_1[good_matche.queryIdx].pt) / 5000 != 0 &&
          !isnan(img_depth.at<ushort>(keypoints_1[good_matche.queryIdx].pt)) &&
          !isinf(img_depth.at<ushort>(keypoints_1[good_matche.queryIdx].pt)))
      {
          //数据保存
          points1.push_back(keypoints_1[good_matche.queryIdx].pt);
          points2.push_back(keypoints_2[good_matche.trainIdx].pt);
          good_matches2.emplace_back(good_matche);

          //求解3D坐标，
          cv::Point3d temp;
          double u = keypoints_1[good_matche.queryIdx].pt.x;
          double v = keypoints_1[good_matche.queryIdx].pt.y;
          temp.z = img_depth.at<ushort>(keypoints_1[good_matche.queryIdx].pt) / 5000.0;
          temp.x = (u - K_cam(0, 2)) * temp.z / K_cam(0, 0);
          temp.y = (v - K_cam(1, 2)) * temp.z / K_cam(1, 1);
          points1_3d.emplace_back(temp);
      }
  }
#endif
//  std::cout << "points1_3d size : " << points1_3d.size() << std::endl;
//  // ---使用OpenCV提供的代数法求解：2D-2D || 求解本质矩阵->R,t
//  cv::Point2d principal_point(319.5, 239.5); //相机光心, TUM dataset标定值
//  double focal_length = 525;             //相机焦距, TUM dataset标定值
//  Mat essential_matrix;
//  essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
//  Mat R, tt;
//  recoverPose(essential_matrix, points1, points2, R, tt, focal_length, principal_point);
//  std::cout << "R is "
//       << "\n"
//       << R << std::endl;
//  std::cout << "t is "
//       << "\n"
//       << tt << std::endl;

#if   ((Optional_Program>>1)^(001>>1))==0
  //---------------主要目的：手写BA优化 位姿 + 路标点--------------
  //---开始BA优化求解：3D-2D
  Eigen::Matrix<double, 4, 4> T;
  T.setIdentity();
  /**
   * @brief Construct a new bundle Adjustment object
   * points1_3d  图像1的3d相机坐标
   * points2    图像2的像素坐标
   * 
   */
  bundleAdjustment(points1_3d, points2, T, img_2, iterations);

#elif   ((Optional_Program>>1)^(011>>1))==0
    //---------------主要目的：g2o图优化--------------
/* 代码构建过程
1. 构建图优化，先设定g2o - 维度、求解器类型、优化选择方法-GN、LM、Dogleg等、创建图优化的核心-稀疏优化器（SparseOptimizer optimizer）
   【定义图的顶点和边，并添加到SparseOptimizer中 】
2. 添加顶点： (顶点的类型需要自己定义)；setEstimate    setId等
3. 添加边：(边同理，也需要自己定义类型-误差-雅克比求解等)；setId、setVertex、setMeasurement、setInformation等
4. 执行优化：
     optimizer.initializeOptimization(); //先初始化
     optimizer.optimize(次数);
5. 输出最优值： 也即是顶点
 */
//    std::vector<Vec3> v_p3d;
//    v_p3d.resize(points1_3d.size());
//    for (int i = 0; i < points1_3d.size(); ++i) {
//        v_p3d[i] = Vec3(points1_3d[i].x, points1_3d[i].y, points1_3d[2].z);
//    }
//    std::vector<cv::KeyPoint> position_;
//    position_.resize(points2.size());
//    for (int i = 0; i < points2.size(); ++i) {
//        position_[i].pt.x = points2[i].x;
//        position_[i].pt.y = points2[i].y;
//    }
    Eigen::Matrix<double, 4, 4> T;
    T.setIdentity();
    SE3 cur_pose = Sophus::SE3d (T);

    // 1.构建图优化
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> BlockSolverType; //6-位姿 3-路标点
    typedef g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose( false );

//    2. 添加顶点-位姿点【手动定义自己的Vertex-两个位姿节点】
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

//    for (int i = 0; i < 2; ++i) {
//        auto *vertex_pose = new myVertexPose();
//        vertex_pose->setId(i);
//        if(0 == i)
//            vertex_pose->setFixed(true); // 第一个点固定为零
//        vertex_pose->setEstimate(cur_pose);
//        optimizer.addVertex(vertex_pose);
//    }

//      添加顶点-路标点【手动定义自己的Vertex-有很多个特征点的节点】 || 以第一帧为准
    for ( size_t i=0; i<points1_3d.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId( 2 + i );
        // 由于深度不知道，只能把深度设置为1了
        double z = points1_3d[i].z;
        double x = ( points1_3d[i].x - cx ) * z / fx;
        double y = ( points1_3d[i].y - cy ) * z / fy;
        v->setMarginalized(true);
        v->setEstimate( Eigen::Vector3d(x,y,z) );
        optimizer.addVertex( v );
    }
//    for (int i = 0; i < points1_3d.size(); ++i) {
//        myVertexXYZ* v = new myVertexXYZ();
//        v->setId(2+i);
//        if(0 == i)
//            v->setFixed(true); // 第一个点固定为零
//        v->setEstimate(Vec3 (points1_3d[i].x,
//                             points1_3d[i].y,
//                             points1_3d[i].z));
//        optimizer.addVertex(v);
//    }
//    // 准备相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );

    //3.  添加边-第一帧【二元边】
//    vector<g2o::EdgeProjectXYZ2UV*> edges;
//    for ( size_t i=0; i<pts1.size(); i++ )
//    {
//        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
//        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
//        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)) );
//        edge->setMeasurement( Eigen::Vector2d(pts1[i].x, pts1[i].y ) );
//        edge->setInformation( Eigen::Matrix2d::Identity() );
//        edge->setParameterId(0, 0);
//        // 核函数
//        edge->setRobustKernel( new g2o::RobustKernelHuber() );
//        optimizer.addEdge( edge );
//        edges.push_back(edge);
//    }

    std::vector<EdgeProjection *> edges;
    for (size_t i = 0; i < points1_3d.size(); ++i) {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
//        EdgeProjection *edge = new EdgeProjection(K_cam);

//        edge->setId(index);
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2)));                // 设置连接的顶点
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(0)));
        edge->setMeasurement( Vec2 (points1_3d[i].x,
                                    points1_3d[i].y));
        edge->setInformation( Eigen::Matrix2d::Identity() ); // 信息矩阵：协方差矩阵之逆
        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer.addEdge( edge );
    }
    //  添加边-第二帧【二元边】
    for (size_t i = 0; i < points2.size(); ++i) {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
//        EdgeProjection *edge = new EdgeProjection(K_cam);

        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(i+2)));                // 设置连接的顶点
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*> (optimizer.vertex(1)));
        edge->setMeasurement( Vec2 (points2[i].x,
                                    points2[i].y));
        edge->setInformation( Eigen::Matrix2d::Identity() ); // 信息矩阵：协方差矩阵之逆
        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer.addEdge( edge );
    }

    //4. 执行优化
    cout<<"开始优化"<<endl;
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::cout<<"优化完毕"<<std::endl;

    // 5.输出
    //我们比较关心两帧之间的变换矩阵
    g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
//    Eigen::Isometry3d pose = v->estimate();
//    myVertexPose* v = dynamic_cast<myVertexPose*>( optimizer.vertex(1) );
    Eigen::Isometry3d pose = (Eigen::Isometry3d )v->estimate();
    cout<<"Pose="<<endl<<pose.matrix()<<endl;
    cout<<"Pose inv ="<<endl<<pose.inverse().matrix()<<endl;
    // 以及所有特征点的位置
    for ( size_t i=0; i<points1_3d.size(); i++ )
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

    cout<<"inliers in total points: "<<inliers<<"/"<<points1_3d.size()+points2.size()<<endl;
    optimizer.save("ba.g2o");

#elif   ((Optional_Program>>1)^(031>>1))==0
//    std::vector<Vec3> v_p3d;
//    v_p3d.resize(points1_3d.size());
//    for (int i = 0; i < points1_3d.size(); ++i) {
//        v_p3d[i] = Vec3(points1_3d[i].x, points1_3d[i].y, points1_3d[2].z);
//    }
//    std::vector<cv::KeyPoint> position_;
//    position_.resize(points2.size());
//    for (int i = 0; i < points2.size(); ++i) {
//        position_[i].pt.x = points2[i].x;
//        position_[i].pt.y = points2[i].y;
//    }
    Eigen::Matrix<double, 4, 4> T;
    T.setIdentity();
    SE3 cur_pose = Sophus::SE3d (T);

    // 1.构建图优化
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,3>> BlockSolverType; //6-位姿 3-路标点
    typedef g2o::LinearSolverCholmod<BlockSolverType::PoseMatrixType> LinearSolverType;

    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose( false );

//    2. 添加顶点-位姿点【手动定义自己的Vertex-两个位姿节点】
    for (int i = 0; i < 2; ++i) {
        myVertexPose *vertex_pose = new myVertexPose();
        vertex_pose->setId(i);
        if(0 == i)
            vertex_pose->setFixed(true); // 第一个点固定为零
        vertex_pose->setEstimate(cur_pose);
        optimizer.addVertex(vertex_pose);
    }

//      添加顶点-路标点【手动定义自己的Vertex-有很多个特征点的节点】 || 以第一帧为准
    for (int i = 0; i < points1_3d.size(); ++i) {
        myVertexXYZ* v = new myVertexXYZ();
        v->setId(2+i);
        if(0 == i)
            v->setFixed(true); // 第一个点固定为零
        v->setEstimate(Vec3 (points1_3d[i].x,
                             points1_3d[i].y,
                             points1_3d[i].z));
        optimizer.addVertex(v);
    }
    //3.  添加边-第一帧【二元边】
    std::vector<EdgeProjection *> edges;
    int index = 1;
    for (size_t i = 0; i < points1_3d.size(); ++i) {
//        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        EdgeProjection *edge = new EdgeProjection(K_cam);

        edge->setId(index);
        edge->setVertex( 1, dynamic_cast<myVertexXYZ*> (optimizer.vertex(i+2)));                // 设置连接的顶点
        edge->setVertex( 0, dynamic_cast<myVertexPose*> (optimizer.vertex(0)));
        edge->setMeasurement( Vec2 (points1_3d[i].x,
                                    points1_3d[i].y));
        edge->setInformation( Eigen::Matrix2d::Identity() ); // 信息矩阵：协方差矩阵之逆
//        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer.addEdge( edge );
        index++;
    }
    //  添加边-第二帧【二元边】
    index = 1;
    for (size_t i = 0; i < points2.size(); ++i) {
//        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        EdgeProjection *edge = new EdgeProjection(K_cam);

        edge->setId(index);
        edge->setVertex( 0, dynamic_cast<myVertexXYZ*> (optimizer.vertex(i+2)));                // 设置连接的顶点
        edge->setVertex( 1, dynamic_cast<myVertexPose*> (optimizer.vertex(1)));
        edge->setMeasurement( Vec2 (points2[i].x,
                                    points2[i].y));
        edge->setInformation( Eigen::Matrix2d::Identity() ); // 信息矩阵：协方差矩阵之逆
//        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber);
        optimizer.addEdge( edge );
        index++;
    }

    //4. 执行优化
    cout<<"开始优化"<<endl;
//    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    std::cout<<"优化完毕"<<std::endl;

    // 5.输出
    //我们比较关心两帧之间的变换矩阵
    myVertexPose* v = dynamic_cast<myVertexPose*>( optimizer.vertex(1) ); // SE3-6-向量
    SE3 pose_T21 =  v->estimate() ; // 需要一个3*4矩阵 / 4*4矩阵 || 向量-矩阵
    cout<<"Pose="<<endl<<pose_T21.matrix()<<endl;
    cout<<"Pose inv ="<<endl<<pose_T21.inverse().matrix()<<endl;


#elif   ((Optional_Program>>1)^(021>>1))==0
    //---------------主要目的：Ceres图优化--------------
    /*
1. 构建最小二乘：AddResidualBlock - 自动求导、模板参数、误差类型等。
2. 配置求解器：选择密集增量cholesky
3. ceres::Solve(options, &problem, &summary); 执行优化
     */
//    Eigen::Quaterniond q_21(1,0,0,0);
//    Eigen::Vector3d t_21(0,0,0);
    double T_21[7] = { 0, 0, 0, 1, 0, 0, 0 }; // 虚部+实部 | t吧
//    Eigen::Vector3d point1_ceres = Eigen::Vector3d (points1_3d[0],points1_3d[1],points1_3d[2]);
//    Eigen::Vector2d point2_ceres = Eigen::Vector2d (points2[0],points2[1]);

    std::vector<Eigen::Matrix<double,3,1>> point1_ceres;
    point1_ceres.resize(points1_3d.size());
    for (int i = 0; i < points1_3d.size(); ++i) {
        point1_ceres[i] = Vec3(points1_3d[i].x, points1_3d[i].y, points1_3d[2].z);
    }
    std::vector<Eigen::Matrix<double,2,1>> point2_ceres;
    point1_ceres.resize(points2.size());
    for (int i = 0; i < point2_ceres.size(); ++i) {
        point2_ceres[i] = Vec2(points2[i].x, points2[i].y);
    }

    // 2.1 构建优化问题 + 损失核函数
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
//    loss_function = new ceres::HuberLoss(1.0) ;

    // 2.2 添加参数快，重构参数维度（四元数-三维）
    ceres::LocalParameterization *local_parame = new ceres::EigenQuaternionParameterization();
    problem.AddParameterBlock(T_21, 4, local_parame);
    problem.AddParameterBlock(T_21 + 4,3);

    // 2.3 添加残差块
    ceres::CostFunction *costfunction;
    for (int i = 0; i < points1_3d.size(); ++i) {
        costfunction = ceres_PnP_32::Create(point1_ceres[i],
                                            point2_ceres[i]);

        problem.AddResidualBlock(
                costfunction,
                loss_function,
                T_21,
                T_21 + 4);
    }

    // 2.4 求解:配置求解器+执行优化
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options,&problem,&summary);

    cout << summary.BriefReport() << endl;   //输出优化的简要信息
    cout << "T_21 :"<< T_21 <<endl;


#endif
    return 0;
}
