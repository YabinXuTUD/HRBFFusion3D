cmake_minimum_required(VERSION 2.6.0)

project(HRBFfusionCORE)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

find_package(Pangolin 0.5 REQUIRED)
find_package(CUDA REQUIRED)
find_package(SuiteSparse REQUIRED)
find_package(ORBSLAM2 REQUIRED)
find_package(Eigen3 3.1.0 REQUIRED)
#find_package(PCL 1.7 REQUIRED)
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(OpenCV REQUIRED)
#find_package(G2O REQUIRED)

add_compile_definitions(EIGEN_DONT_VECTORIZE)
add_compile_definitions(EIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(efusion_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Shaders" CACHE PATH "Where the shaders live")
include_directories(${Pangolin_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
#set(EIGEN3_INCLUDE_DIR /home/yabinxu/library/Eigen)
include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${SUITESPARSE_INCLUDE_DIRS})
include_directories(${ORBSLAM2_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/ORB_SLAM2_m)
include_directories(${CMAKE_SOURCE_DIR}/ORB_SLAM2_m/include)
include_directories(.)
#include_directories(${PCL_INCLUDE_DIRS}) Unsupported gpu architecture 'compute_30'

set(ransacDir ${CMAKE_CURRENT_SOURCE_DIR}/ransac/RANSAC-1.1)
include_directories(${ransacDir})
set(ransaclib ${CMAKE_CURRENT_SOURCE_DIR}/ransac/build/lib3rd_ransac.a)

file(GLOB srcs *.cpp)
file(GLOB utils_srcs Utils/*.cpp)
file(GLOB shader_srcs Shaders/*.cpp)
file(GLOB cuda Cuda/*.cu)
file(GLOB containers Cuda/containers/*.cpp)
file(GLOB bundler Bundler/*.cpp)
file(GLOB SIFTGPU Bundler/SIFT/*.cpp)
#file(GLOB Line
#     external/lsd/lsd.cpp
#     Line/lineslam.cpp
#     Line/utils.cpp
#     Line/motion.cpp
#     Line/global_line_constructor.cpp
#     )

if(WIN32)
  file(GLOB hdrs *.h)
  file(GLOB utils_hdrs Utils/*.h)
  file(GLOB shader_hdrs Shaders/*.h)
  file(GLOB cuda_hdrs Cuda/*.cuh)
  file(GLOB containers_hdrs Cuda/containers/*.hpp)
  file(GLOB bundler Bundler/*.cpp)
  file(GLOB SIFTGPU Bundler/SIFT/*.cpp)
endif()

set(CUDA_ARCH_BIN "35 50 52 75" CACHE STRING "Specify 'real' GPU arch to build binaries for, BIN(PTX) format is supported. Example: 1.3 2.1(1.3) or 13 21(13)")
set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")              

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
include(CudaComputeTargetFlags.cmake)                  
APPEND_TARGET_ARCH_FLAGS()

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")           
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false") 

CUDA_COMPILE(cuda_objs ${cuda})

if(WIN32)
  set(ADDITIONAL_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

set(CMAKE_CXX_FLAGS ${ADDITIONAL_CMAKE_CXX_FLAGS} "-O2 -msse2 -msse3 -Wall -std=c++11 -DSHADER_DIR=${efusion_SHADER_DIR}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11 -DSHADER_DIR=${efusion_SHADER_DIR}")
  
if(WIN32)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-DNOMINMAX)
  set (EXTRA_WINDOWS_LIBS ${EXTRA_WINDOWS_LIBS} ws2_32)
endif()

add_library(hrbffusion SHARED
            ${srcs}
            ${utils_srcs}
            ${shader_srcs}
            ${cuda} 
            ${cuda_objs} 
            ${containers}
            ${hdrs}
            ${utils_hdrs}
            ${shader_hdrs}
            ${cuda_hdrs}
            ${containers_hdrs}
            ${Line}
            #${bundler}
            #${SIFTGPU}
)

target_link_libraries(hrbffusion
                      ${Eigen_LIBRARIES}
                      ${Pangolin_LIBRARIES}
                      ${CUDA_LIBRARIES}
                      ${SUITESPARSE_LIBRARIES}
                      ${EXTRA_WINDOWS_LIBS}
#                      ${PCL_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${ransaclib}
                      ${Open3D_LIBRARIES}
                      ${ORBSLAM2_LIBRARY}
                      ${PROJECT_SOURCE_DIR}/ORB_SLAM2_m/Thirdparty/DBoW2/lib/libDBoW2.so
                      ${PROJECT_SOURCE_DIR}/ORB_SLAM2_m/Thirdparty/g2o/lib/libg2o.so
                      ${PROJECT_SOURCE_DIR}/external/levmar-2.6/liblevmar.a
                      boost_system
                      boost_filesystem
#                      ${G2O_CORE_LIBRARY}
#                      ${G2O_STUFF_LIBRARY}
#                      ${G2O_SOLVER_CHOLMOD}
#                      ${G2O_TYPES_SLAM3D}
)

target_include_directories(hrbffusion PUBLIC ${Open3D_INCLUDE_DIRS})

INSTALL(TARGETS hrbffusion
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
)
