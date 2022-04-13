add_library(
        my_slam_cpp SHARED
        orb_self.cpp
        bundleAdjustment.cpp
)

target_link_libraries(my_slam_cpp ${THIRD_PARTLY_LIBS} ${FMT_LIBRARIES})