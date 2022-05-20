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
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */

#ifndef CUDA_CUDAFUNCS_CUH_
#define CUDA_CUDAFUNCS_CUH_

#if __CUDA_ARCH__ < 300
#define MAX_THREADS 512
#else
#define MAX_THREADS 1024
#endif

#include "containers/device_array.hpp"
#include "types.cuh"
#include <cublas_v2.h>

void updateLambdaMap(const mat33& Rcurr,
                     const float3& tcurr,
                     const DeviceArray2D<float>& vmap_curr,
                     const DeviceArray2D<float>& nmap_curr,
                     const mat33& Rprev_inv,
                     const float3& tprev,
                     const CameraModel& intr,
                     const DeviceArray2D<float>& vmap_g_prev,
                     const DeviceArray2D<float>& nmap_g_prev,
                     const DeviceArray2D<int2>& corresICP,
                     DeviceArray2D<float3>& z_thrinkMap,
                     DeviceArray2D<float3>& lambdaMap,
                     float distThres,
                     float angleThres,
                     int threads,
                     int blocks);

void icpStep(const mat33& Rcurr,
             const float3& tcurr,
             const DeviceArray2D<float>& vmap_curr,
             const DeviceArray2D<float>& nmap_curr,
             const DeviceArray2D<float>& ck1maps_curr,
             const DeviceArray2D<float>& ck2maps_curr,
             const DeviceArray2D<unsigned short>& plane_match_map_curr,
             const int icp_weight_layers,
             const mat33& Rprev_inv,
             const float3& tprev,
             const CameraModel& intr,
             const DeviceArray2D<float>& vmap_g_prev,
             const DeviceArray2D<float>& nmap_g_prev,
             const DeviceArray2D<float>& ck1maps_g_prev,
             const DeviceArray2D<float>& ck2maps_g_prev,
             const DeviceArray2D<float>& icpWeightmap_g_prev,
             const DeviceArray2D<unsigned short>& plane_match_map_g,
             DeviceArray2D<int2>& corresICP,
             DeviceArray2D<float4>& cuda_out,
             DeviceArray2D<float3>& z_thrinkMap,
             const DeviceArray2D<float3>& lambdaMap,
             float distThres,
             float angleThres,
             float curvatureThres,
             bool icp_if_use_coorespondence_search,
             int icp_search_radius,
             bool icp_if_use_weight,
             bool use_sparse_icp,
             DeviceArray<JtJJtrSE3> & sum,
             DeviceArray<JtJJtrSE3> & out,
             float * matrixA_host,
             float * vectorB_host,
             float * residual_host,
             int threads,
             int blocks);

void rgbStep(const DeviceArray2D<DataTerm> & corresImg,
             const float & sigma,
             const DeviceArray2D<float3> & cloud,
             const float & fx,
             const float & fy,
             const DeviceArray2D<short> & dIdx,
             const DeviceArray2D<short> & dIdy,
             bool rgb_use_RGBGradient_weight,
             const float & sobelScale,
             DeviceArray<JtJJtrSE3> & sum,
             DeviceArray<JtJJtrSE3> & out,
             float * matrixA_host,
             float * vectorB_host,
             int threads,
             int blocks);

void so3Step(const DeviceArray2D<unsigned char> & lastImage,
             const DeviceArray2D<unsigned char> & nextImage,
             const mat33 & imageBasis,
             const mat33 & kinv,
             const mat33 & krlr,
             DeviceArray<JtJJtrSO3> & sum,
             DeviceArray<JtJJtrSO3> & out,
             float * matrixA_host,
             float * vectorB_host,
             float * residual_host,
             int threads,
             int blocks);

void computeRgbResidual(const float & minScale,
                        const DeviceArray2D<short> & dIdx,
                        const DeviceArray2D<short> & dIdy,
                        const DeviceArray2D<float> & lastDepth,
                        const DeviceArray2D<float> & nextDepth,
                        const DeviceArray2D<unsigned char> & lastImage,
                        const DeviceArray2D<unsigned char> & nextImage,
                        DeviceArray2D<DataTerm> & corresImg,
                        DeviceArray<int2> & sumResidual,
                        const float maxDepthDelta,
                        const float3 & kt,
                        const mat33 & krkinv,
                        int & sigmaSum,
                        int & count,
                        int threads,
                        int blocks);

void createVMap(const CameraModel& intr,
                const DeviceArray2D<float> & depth,
                DeviceArray2D<float> & vmap,
                const float depthCutoff,
                const float mDepthMapFactor);

void createNMap(const DeviceArray2D<float>& vmap,
                DeviceArray2D<float>& nmap);

void tranformMaps(const DeviceArray2D<float>& vmap_src,
                  const DeviceArray2D<float>& nmap_src,
                  const mat33& Rmat,
                  const float3& tvec,
                  DeviceArray2D<float>& vmap_dst,
                  DeviceArray2D<float>& nmap_dst);

void transformCurvMaps(const DeviceArray2D<float>& curvk1_src,
                       const DeviceArray2D<float>& curvk2_src,
                       const mat33& Rmat, const float3& tvec,
                       DeviceArray2D<float>& curvk1_dst,
                       DeviceArray2D<float>& curvk2_dst);

void copyMaps(const DeviceArray<float>& vmap_src,
              const DeviceArray<float>& nmap_src,
              DeviceArray2D<float>& vmap_dst,
              DeviceArray2D<float>& nmap_dst);

void copyCurvatureMap(const DeviceArray<float>& cmap_src,
                      DeviceArray2D<float>& cmap_dst,
                      const float curvatureThreshold);

void copyicpWeightMap(const DeviceArray<float>& icpwmap_src,
                      DeviceArray2D<float>& icpwmap_dst);

void resizeVMap(const DeviceArray2D<float>& input,
                DeviceArray2D<float>& output);

void resizeNMap(const DeviceArray2D<float>& input,
                DeviceArray2D<float>& output);

void resizeCMap(const DeviceArray2D<float>& input,
                DeviceArray2D<float>& output);

void resizeicpWeightMap(const DeviceArray2D<float>& input,
                        DeviceArray2D<float>& output);

void resizePlaneMap(const DeviceArray2D<unsigned short>& input,
                DeviceArray2D<unsigned short>& output);

void imageBGRToIntensity(cudaArray * cuArr,
                         DeviceArray2D<unsigned char> & dst);

void verticesToDepth(DeviceArray<float>& vmap_src,
                     DeviceArray2D<float> & dst,
                     float cutOff);

void projectToPointCloud(const DeviceArray2D<float> & depth,
                         const DeviceArray2D<float3> & cloud,
                         CameraModel & intrinsics,
                         const int & level);

void pyrDown(const DeviceArray2D<float> & src,
             DeviceArray2D<float> & dst);

void pyrDownGaussF(const DeviceArray2D<float> & src,
                   DeviceArray2D<float> & dst);

void pyrDownUcharGauss(const DeviceArray2D<unsigned char>& src,
                       DeviceArray2D<unsigned char> & dst);

//taking discrete derivative
void computeDerivativeImages(DeviceArray2D<unsigned char>& src,
                             DeviceArray2D<short>& dx,
                             DeviceArray2D<short>& dy);

void updateGlobalModel();

#endif /* CUDA_CUDAFUNCS_CUH_ */
