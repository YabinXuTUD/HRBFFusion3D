cmake_minimum_required(VERSION 2.6.0)

project(HRBFFusion)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
  macro(CANONIFY_BOOL var)
    if(${var})
      set(${var} TRUE)
    else()
      set(${var} FALSE)
    endif()
  endmacro()
endif()

if(WIN32)
  find_package(JPEG REQUIRED)
endif()
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)
find_package(ZLIB REQUIRED)
find_package(Pangolin 0.5 REQUIRED)
find_package(CUDA REQUIRED)
find_package(ORBSLAM2 REQUIRED)
#find_package(OpenNI2 REQUIRED)
find_package(hrbffusion REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(OpenMP)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(OpenCV REQUIRED)
find_package(ORBSLAM2 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
#find_package(G2O REQUIRED)
#find_package(PCL 1.7 REQUIRED)

add_compile_definitions(EIGEN_DONT_VECTORIZE)
add_compile_definitions(EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

if(WIN32)
  find_package(RealSense QUIET)
  CANONIFY_BOOL(REALSENSE_FOUND)
  message(STATUS "librealsense found: ${REALSENSE_FOUND}")
  option(WITH_REALSENSE "Build with Intel RealSense support?" ${REALSENSE_FOUND})
endif()

if(WIN32)
  include_directories(${JPEG_INCLUDE_DIR})
endif()

include_directories(${ZLIB_INCLUDE_DIR})
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${HRBFFUSION_INCLUDE_DIR})
include_directories(${ORBSLAM2_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/../../Core/src/ORB_SLAM2_m)
include_directories(${CMAKE_SOURCE_DIR}/../../Core/src/ORB_SLAM2_m/include)
set(OPENNI2_INCLUDE_DIR /home/yabinxu/library/OpenNI2/Include)
set(OPENNI2_LIBRARY /home/yabinxu/library/OpenNI2/Bin/x64-Release/libOpenNI2.so)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -luuid")
include_directories(${OPENNI2_INCLUDE_DIR})
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

set(ransacDir /home/yabinxu/workspace/hrbf_fusion/Core/src/ransac/RANSAC-1.1)
include_directories(${ransacDir})
set(ransaclib /home/yabinxu/workspace/hrbf_fusion/Core/src/ransac/build/lib3rd_ransac.a)

#add_compile_options(-D WITH_REALSENSE)
if(WITH_REALSENSE)
  set (REALSENSE_INCLUDE_DIR /usr/local/include)
  set (REALSENSE_LIBRARY /usr/local/lib/librealsense.so)
  include_directories(${REALSENSE_INCLUDE_DIR})
endif()

file(GLOB srcs *.cpp *.h)
file(GLOB tools_srcs Tools/*.cpp Tools/*.h)

if(WIN32)
  file(GLOB hdrs *.h)
  file(GLOB tools_hdrs Tools/*.h)
endif()

option(WITH_OPENMP               "Use OpenMP multi-threading"               ON)

if(WIN32)
  add_definitions(-Dlinux=0)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-DNOMINMAX)
  set(ADDITIONAL_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(EXTRA_WINDOWS_LIBS ${EXTRA_WINDOWS_LIBS} ws2_32 ${JPEG_LIBRARY})
else()
  add_definitions(-Dlinux=1)
endif()

set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-O2 -msse2 -msse3 -Wall -std=c++11 -fopenmp")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11")

#for HRBF fusion
add_executable(HRBFfusion
        HRBF_fusion.h HRBF_main.cpp HRBF_fusion.cpp
        ${tools_srcs}
        ${hdrs}
        ${tools_hdrs}
        )

target_link_libraries(HRBFfusion
        ${EXTRA_WINDOWS_LIBS}
        ${ZLIB_LIBRARY}
        ${Pangolin_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${EXTRA_LIBS}
        ${HRBFFUSION_LIBRARY}
        ${OPENNI2_LIBRARY}
        ${REALSENSE_LIBRARY}
        ${SUITESPARSE_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        ${OpenCV_LIBS}
#        ${PCL_LIBRARIES}
        ${ransaclib}
        ${Open3D_LIBRARIES}
        ${ORBSLAM2_LIBRARY}
        ${PROJECT_SOURCE_DIR}/../../Core/src/ORB_SLAM2_m/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/../../Core/src/ORB_SLAM2_m/Thirdparty/g2o/lib/libg2o.so
        boost_system
        boost_filesystem
#        ${G2O_CORE_LIBRARY}
#        ${G2O_STUFF_LIBRARY}
#        ${G2O_SOLVER_CHOLMOD}
#        ${G2O_TYPES_SLAM3D}
        )
target_include_directories(HRBFfusion PUBLIC ${Open3D_INCLUDE_DIRS})
