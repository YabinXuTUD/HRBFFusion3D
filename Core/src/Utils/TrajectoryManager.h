
#pragma  once

#include <vector>
#include <string>
#include <iostream>
#include "GlobalStateParams.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>
#include<Eigen/StdVector>

#include <iomanip>

//#include <Core/Core.h>
//#include <IO/IO.h>

class TrajectoryManager{
public:
    TrajectoryManager();
    ~TrajectoryManager();

public:

      void LoadFromFile();
      bool SaveTrajectoryToFile();

      Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);

      std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> poses;  //first term is the pose, second term is the submap index
      std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> poses_original;  //first term is the pose, second term is the submap index
      std::vector<unsigned long long int> timstamp;        //for TUM-like dataset timestamp should be provided

      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
