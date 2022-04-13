add_executable(lxq_slam1 main.cpp)
target_link_libraries(lxq_slam1 my_slam_cpp ${THIRD_PARTLY_LIBS} )

add_executable(slam_test main_test.cpp)
target_link_libraries(slam_test my_slam_cpp ${THIRD_PARTLY_LIBS} )


add_executable(1_slam_14 1_slam_14.cpp)
target_link_libraries(1_slam_14 my_slam_cpp ${THIRD_PARTLY_LIBS} )


add_executable(0_c_pratice 0_c_pratice.cpp)
#target_link_libraries(c++-pratice my_slam_cpp ${THIRD_PARTLY_LIBS} )