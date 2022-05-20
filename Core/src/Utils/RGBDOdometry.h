/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#ifndef RGBDODOMETRY_H_
#define RGBDODOMETRY_H_


#include "Stopwatch.h"
#include "../GPUTexture.h"
#include "../Cuda/cudafuncs.cuh"
#include "../Utils/GlobalStateParams.h"
#include "OdometryProvider.h"
#include "GPUConfig.h"
#include "Img.h"
#include "../Shaders/Shaders.h"
#include "../PlaneExtraction.h"

#include <stdio.h>
#include <vector>
#include <vector_types.h>

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include<Eigen/Core>
//#include <Core/Core.h>
//#include <IO/IO.h>
//#include <Core/Utility/Eigen.h> // define Eigen matrix

//typedef std::vector<std::pair<std::vector<int>, std::vector<int>>> GroupMatch;

struct host_DataTerm{
     short zero[2];
     short one[2];
     float diff;
     bool valid;
};

typedef std::vector<Eigen::Vector4i> CorrespondenceSetPixelWise;

class RGBDOdometry
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        RGBDOdometry(int width,
                     int height,
                     float cx, float cy, float fx, float fy,
                     float distThresh = 0.1f,
                     float angleThresh = sin(20.f * 3.14159265f / 180.f)
                 );

        virtual ~RGBDOdometry();

        // This map the texture information into the CUDA array for registration use
        //init source file
        void initICP(GPUTexture * filteredDepth, const float depthCutoff, const float mDepthMapFactor);
        void initICP(GPUTexture * predictedVertices, GPUTexture * predictedNormals, const float depthCutoff); //This for model to model registration

        //init model file
        void initICPModel(GPUTexture * predictedVertices, GPUTexture * predictedNormals, const float depthCutoff, const Eigen::Matrix4f & modelPose);

        void initRGB(GPUTexture * rgb);

        void initRGBModel(GPUTexture * rgb);

        void initCurvature(GPUTexture* curvk1, GPUTexture* curvk2);

        void initCurvatureModel(GPUTexture* curvk1Model, GPUTexture* curvk2Model, const Eigen::Matrix4f& modelPose);

        void initICPweight(GPUTexture* icpWeight);

        void initFirstRGB(GPUTexture * rgb); //for the first time

        void correspondPlaneSearch(GPUTexture* depthfiltered, GPUTexture* vertices);

        void correspondPlaneSearch();

        void correspondPlaneSearchRANSAC(const Eigen::Matrix4f& currpose);

        void getIncrementalTransformation(Eigen::Vector3f & trans,
                                          Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
                                          const bool & rgbOnly,
                                          const float & icpWeight,
                                          const bool & pyramid,
                                          const bool & fastOdom,
                                          const bool & so3,
                                          const bool & if_curvature_info,
                                          const int index_frame);

        Eigen::MatrixXd getCovariance(); //Covariance

        void getLastCorrespondence(int2* host_corresp, int rows, int cols);
//        Eigen::Matrix6d CreateInformationMatrix();
        void addToPoseGraph(Eigen::Matrix3f Rprevious, Eigen::Vector3f tprevious, Eigen::Matrix3f Rcurrent, Eigen::Vector3f tcurrent,
                            int index_frame, Eigen::Matrix4d Trans);
        void savePoseGraph(int index_fragment);


        void DownloadGPUMaps();
        void savefilePLY(float* vertices, float* normals, float* curv_max, float* curv_min,
                         short* image_dx,short* image_dy,float* icp_weight, int rows, int cols, std::string type, int paramid);
        void saveCorrepICPsave(int2* corresp,int stride,int frameID, int paramid, int iterative);
        float saveCudaAttrib(float4* corresp,int cols, int rows,int frameID, int paramid, int iterative);
        void saveCudaAttrib(float3* corresp,int cols, int rows,int frameID, int paramid, int iterative);


        float lastICPError; //average geometry error;
        float lastICPCount; //number of valid correspodences;
        float lastRGBError; //RGB error;
        float lastRGBCount; //number of valid correspodences;
        float lastSO3Error; //
        float lastSO3Count; //

        std::vector<float> dist_error_points;

        Eigen::Matrix<double, 6, 6, Eigen::RowMajor> lastA;
        Eigen::Matrix<double, 6, 1> lastb;

        //export pose graph with fragents
        CorrespondenceSetPixelWise correspondence;
//      open3d::PoseGraph pose_graph_;

    private:
        void populateRGBDData(GPUTexture * rgb,
                              DeviceArray2D<float> * destDepths,
                              DeviceArray2D<unsigned char> * destImages);

        std::vector<DeviceArray2D<float>> depth_tmp;  //hierarchy optimization performance

        //as temporary data buffer for vertices maps(from texture to CUDA)
        DeviceArray<float> vmaps_tmp;
        DeviceArray<float> nmaps_tmp;
        DeviceArray<float> ck1maps_tmp;
        DeviceArray<float> ck2maps_tmp;
        DeviceArray<float> icpweight_tmp;

        std::vector<DeviceArray2D<float> > vmaps_g_prev_;  //the previous global(destination) frame the real frame map data in GPU
        std::vector<DeviceArray2D<float> > nmaps_g_prev_;
        std::vector<DeviceArray2D<float> > ck1maps_g_prev_;
        std::vector<DeviceArray2D<float> > ck2maps_g_prev_;

        std::vector<DeviceArray2D<float> > vmaps_curr_;   //the current local(source) frame in GPU
        std::vector<DeviceArray2D<float> > nmaps_curr_;
        std::vector<DeviceArray2D<float> > ck1maps_curr_;
        std::vector<DeviceArray2D<float> > ck2maps_curr_;

        std::vector<DeviceArray2D<float> > icpWeightMap_;

        //this records the corresponding points with same plane
        std::vector<DeviceArray2D<unsigned short>> plane_match_map_g_;
        std::vector<DeviceArray2D<unsigned short>> plane_match_map_curr_;

        CameraModel intr;  //camera intrisic

        DeviceArray<JtJJtrSE3> sumDataSE3;  //SE3, containers in GPU
        DeviceArray<JtJJtrSE3> outDataSE3;
        DeviceArray<int2> sumResidualRGB;

        DeviceArray<JtJJtrSO3> sumDataSO3;  //SO3
        DeviceArray<JtJJtrSO3> outDataSO3;

        const int sobelSize;
        const float sobelScale;
        const float maxDepthDeltaRGB;
        const float maxDepthRGB;

        std::vector<int2> pyrDims; //pyramid dimension

        static const int NUM_PYRS = 3;

        DeviceArray2D<float> lastDepth[NUM_PYRS];
        DeviceArray2D<unsigned char> lastImage[NUM_PYRS];

        DeviceArray2D<float> nextDepth[NUM_PYRS];
        DeviceArray2D<unsigned char> nextImage[NUM_PYRS];
        DeviceArray2D<short> nextdIdx[NUM_PYRS];
        DeviceArray2D<short> nextdIdy[NUM_PYRS];

        DeviceArray2D<unsigned char> lastNextImage[NUM_PYRS];   //for so3
        DeviceArray2D<DataTerm> corresImg[NUM_PYRS];
        DeviceArray2D<int2> corresICP[NUM_PYRS];
        DeviceArray2D<float4> cuda_out_[NUM_PYRS];

        //for sparse ICP test
        DeviceArray2D<float3> z_thrinkMap[NUM_PYRS];
        DeviceArray2D<float3> lambdaMap[NUM_PYRS];

        DeviceArray2D<float3> pointClouds[NUM_PYRS];

        std::vector<int> iterations;  //pyramid iterations
        std::vector<float> minimumGradientMagnitudes;

        float distThres_;
        float angleThres_;
        float curvatureThres_;

        Eigen::Matrix<double, 6, 6> lastCov;

        const int width;
        const int height;
        const float cx, cy, fx, fy;
};

#endif /* RGBDODOMETRY_H_ */
