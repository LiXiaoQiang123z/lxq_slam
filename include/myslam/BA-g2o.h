//
// Created by lxq on 2022/3/25.
//

#ifndef CH9_BA_G2O_H
#define CH9_BA_G2O_H

#include "common.h"

/***************************手写定义-顶点****************************/
// 手动定义自己的顶点【位姿顶点】
/*
 * 1. 初始值获取
 * 2. 如何更新
 */
class myVertexPose: public g2o::BaseVertex<6, SE3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override { _estimate = SE3(); }

    virtual void oplusImpl(const double* update) override
    {
        Vec6 update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = SE3::exp(update_eigen) * _estimate; // 左乘更新 SE3 - 旋转矩阵R
    }

    virtual bool read(std::istream& in) override {return true;}
    virtual bool write(std::ostream& out) const override {return true;}
};
// 手动定义自己的顶点【路标顶点】
class myVertexXYZ: public g2o::BaseVertex<3, Vec3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual void setToOriginImpl() override { _estimate = Vec3::Zero(); }

    virtual void oplusImpl(const double* update) override
    {
        _estimate[0] += update[0];
        _estimate[1] += update[1];
        _estimate[2] += update[2];
    }

    virtual bool read(std::istream& in) override {return true;}
    virtual bool write(std::ostream& out) const override {return true;}
};

/***************************手写定义-边****************************/
// 手动定义自己的边 【模板-一元边】
/*
 * errorDim     2
 * errorType    Vec2
 * 1. 计算残差
 * 2. 计算雅克比
 */
class myEdgeProjectionPoseOnly: public g2o::BaseUnaryEdge<2, Vec2, myVertexPose>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    myEdgeProjectionPoseOnly(const Vec3 &pos, const Mat33 &K)
        :   _pos3d(pos),_K(K)  {}

    virtual bool read(std::istream& in) override {return true;}
    virtual bool write(std::ostream& out) const override {return true;}

    virtual void computeError() override {
        const myVertexPose *v = static_cast<myVertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Vec3 pose_pixel = _K * (T * _pos3d); //旋转到后一帧图像中
        pose_pixel /= pose_pixel[2]; //相机-像素
        _error = _measurement - pose_pixel.head(2); // 真实值 - 估计值 TODO:改了 pos_pixel.head<2>()
    }

    virtual void linearizeOplus() override {  // 求误差对优化变量的偏导数，雅克比矩阵
        const myVertexPose *v = static_cast<myVertexPose *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Vec3 pos_cam = T * _pos3d; // 旋转到下一帧的相机坐标系中
        double fx = _K(0,0);
        double fy = _K(1,1);
//        double X = pos_cam[0];
//        double Y = pos_cam[1];
//        double Z = pos_cam[2];
        double inv_z = 1 / (pos_cam[2]+ 1e-18);
        double inv_z2 = inv_z * inv_z;

        _jacobianOplusXi << -fx * inv_z, // 对位姿求导结果->雅克比J
                0,
                fx * pos_cam[0] * inv_z2,
                fx * pos_cam[0] * pos_cam[1] * inv_z2,
                -fx - fx * pos_cam[0] * pos_cam[0] * inv_z2,
                fx * pos_cam[1] * inv_z,

                0,
                -fy * inv_z,
                fy * pos_cam[1] * inv_z,
                fy + fy * pos_cam[1] * pos_cam[1] * inv_z2,
                -fy * pos_cam[0] * pos_cam[1] * inv_z2,
                -fy * pos_cam[0] * inv_z;
    }
private:
    Vec3 _pos3d;
    Mat33 _K;
};

// 手动定义自己的边 【模板-二元边】 TODO:加入了世界坐标系（相机外参）
/**
 * errorDim     2
 * errorType    Vec2
 * myVertexPose
 * myVertexXYZ
 */
class EdgeProjection
        : public g2o::BaseBinaryEdge<2, Vec2, myVertexPose,myVertexXYZ>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    /// 构造时传入相机内外参
    EdgeProjection( const Mat33 &K) : _K(K) {
//        _cam_ext = cam_ext;
    }

    virtual bool read(std::istream& in) override {return true;}
    virtual bool write(std::ostream& out) const override {return true;}

    virtual void computeError() override {
        const myVertexPose *v0 = static_cast<myVertexPose *>(_vertices[0]);
        const myVertexXYZ *v1 = static_cast<myVertexXYZ *>(_vertices[1]);

        Sophus::SE3d T = v0->estimate();
        Vec3 pose_pixel = _K * (T * v1->estimate()); //旋转到后一帧图像中
        pose_pixel /= pose_pixel[2]; //相机-像素
        _error = _measurement - pose_pixel.head(2); // 真实值 - 估计值 TODO:改了 pos_pixel.head<2>()
    }

    virtual void linearizeOplus() override {  // 求误差对优化变量的偏导数，雅克比矩阵
        const myVertexPose *v0 = static_cast<myVertexPose *>(_vertices[0]);
        const myVertexXYZ *v1 = static_cast<myVertexXYZ *>(_vertices[1]);

        SE3 T = v0->estimate();
        Vec3 pw = v1->estimate();
        Vec3 pos_cam =  T * pw; // 旋转到下一帧的相机坐标系中
        double fx = _K(0,0);
        double fy = _K(1,1);
        double inv_z = 1 / (pos_cam[2] + 1e-18); // TODO:+ 1e-18目的是啥
        double inv_z2 = inv_z * inv_z;

        _jacobianOplusXi << -fx * inv_z, // 对位姿求导结果->雅克比J
                0,
                fx * pos_cam[0] * inv_z2,
                fx * pos_cam[0] * pos_cam[1] * inv_z2,
                -fx - fx * pos_cam[0] * pos_cam[0] * inv_z2,
                fx * pos_cam[1] * inv_z,
                0,
                -fy * inv_z,
                fy * pos_cam[1] * inv_z,
                fy + fy * pos_cam[1] * pos_cam[1] * inv_z2,
                -fy * pos_cam[0] * pos_cam[1] * inv_z2,
                -fy * pos_cam[0] * inv_z;

        _jacobianOplusXj = _jacobianOplusXi.block<2,3>(0,0) * T.rotationMatrix(); // P' * T
    }
private:
//    SE3 _cam_ext; // 相机外参
    Mat33 _K;
};

#endif //CH9_BA_G2O_H
