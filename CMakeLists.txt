cmake_minimum_required(VERSION 3.5)
project(camera_calibration)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# add_executable(calibrate instrinsic_cam.cpp)
# target_link_libraries(calibrate ${OpenCV_LIBS})
