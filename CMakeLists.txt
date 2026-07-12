cmake_minimum_required(VERSION 3.15)

set(VCPKG_OVERLAY_TRIPLETS "${CMAKE_CURRENT_SOURCE_DIR}/custom-triplets")

project(HandVolumeControl CXX)

find_package(OpenCV REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)

add_executable(${PROJECT_NAME} 
    src/main.cpp
    src/PalmDetector.cpp
    src/HandLandmarker.cpp
    src/PalmDetector.hpp
    src/HandLandmarker.hpp
    src/npy.hpp
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
    onnxruntime::onnxruntime
    ${OpenCV_LIBS}
)