//
// Created by lxq on 2022/3/26.
//

#ifndef CH9_BA_CERES_H
#define CH9_BA_CERES_H

#include "common.h"

// 代价函数的计算模型  || 【1.定义残差块的类/结构体】 ||e = u-KTP/s
struct CURVE_FITTING_COST {
    // 优化变量
    CURVE_FITTING_COST(SE3 pose_T21) : _pose_T21(pose_T21) {}

    // 残差的计算
    template<typename T>
    bool operator()(
            const T *const abc, // 模型参数，有3维
            T *residual) const {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y-exp(ax^2+bx+c)
        return true;
    }

    const SE3 _pose_T21;
};

#endif //CH9_BA_CERES_H
