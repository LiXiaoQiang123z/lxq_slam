cmake_minimum_required(VERSION 3.4)
project(myslam)

set(CATKIN_TOPLEVEL TRUE) # 嘛意思啊？？ 设置优先级别最高？？

set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

# 设置C++编译选项 ： 
# -std=c++11 -O2 ${SSE_FLAGS} -msse4 ：支持SSE指令集。
#set(CMAKE_CXX_FLAGS "-std=c++11 -O2 ${SSE_FLAGS} -msse4")
set(CMAKE_CXX_STANDARD 14) #c++14标准

# 设置输出位置
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules) #查找第三方库文件
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) #设置文件输出路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib) #设置生成库文件路径
################## find package #####################

# find package
find_package(OpenCV 3 REQUIRED)
find_package(Sophus REQUIRED)
#find_package(fmt REQUIRED)
#set(FMT_LIBRARIES fmt::fmt)
find_package(G2O REQUIRED)
find_package(Ceres REQUIRED)
find_package(Pangolin REQUIRED)
#find_package(CSparse REQUIRED)


# include directions
include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3"
        ${G20_INCLUDE_DIRS}
        ${CERES_INCLUDE_DIRS}
        ${Pangolin_INCLUDE_DIRS} #绘图
#        ${CSPARSE_INCLUDE_DIR}
)
include_directories("/usr/include/suitesparse")

# set libraries = third partly libraries
set(THIRD_PARTLY_LIBS
        ${OpenCV_LIBS}
        ${Sophus_LIBRARIES}
#        ${FMT_LIBRARIES}
#        ${G20_LIBRARIES}
        g2o_core g2o_stuff g2o_types_sba g2o_solver_csparse g2o_csparse_extension
        ${CERES_LIBRARIES}
        ${Pangolin_LIBRARIES} #绘图
#        ${CSPARSE_LIBRARY}
        )

################## add subdirectory #####################
include_directories(${PROJECT_SOURCE_DIR}/include) #声明头文件路径
add_subdirectory(src)
add_subdirectory(app) #主文件所在






