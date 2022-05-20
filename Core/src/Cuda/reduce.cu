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

#include "cudafuncs.cuh"
#include "convenience.cuh"
#include "operators.cuh"


#if __CUDA_ARCH__ < 300
__inline__ __device__
float __shfl_down(float val, int offset, int width = 32)
{
    static __shared__ float shared[MAX_THREADS];
    int lane = threadIdx.x % 32;
    shared[threadIdx.x] = val;
    __syncthreads();
    val = (lane + offset < width) ? shared[threadIdx.x + offset] : 0;
    __syncthreads();
    return val;
}

__inline__ __device__
int __shfl_down(int val, int offset, int width = 32)
{
    static __shared__ int shared[MAX_THREADS];
    int lane = threadIdx.x % 32;
    shared[threadIdx.x] = val;
    __syncthreads();
    val = (lane + offset < width) ? shared[threadIdx.x + offset] : 0;
    __syncthreads();
    return val;
}
#endif

#if __CUDA_ARCH__ < 350
template<typename T>
__device__ __forceinline__ T __ldg(const T* ptr)
{
    return *ptr;
}
#endif

__inline__  __device__ JtJJtrSE3 warpReduceSum(JtJJtrSE3 val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.aa += __shfl_down(val.aa, offset);
        val.ab += __shfl_down(val.ab, offset);
        val.ac += __shfl_down(val.ac, offset);
        val.ad += __shfl_down(val.ad, offset);
        val.ae += __shfl_down(val.ae, offset);
        val.af += __shfl_down(val.af, offset);
        val.ag += __shfl_down(val.ag, offset);

        val.bb += __shfl_down(val.bb, offset);
        val.bc += __shfl_down(val.bc, offset);
        val.bd += __shfl_down(val.bd, offset);
        val.be += __shfl_down(val.be, offset);
        val.bf += __shfl_down(val.bf, offset);
        val.bg += __shfl_down(val.bg, offset);

        val.cc += __shfl_down(val.cc, offset);
        val.cd += __shfl_down(val.cd, offset);
        val.ce += __shfl_down(val.ce, offset);
        val.cf += __shfl_down(val.cf, offset);
        val.cg += __shfl_down(val.cg, offset);

        val.dd += __shfl_down(val.dd, offset);
        val.de += __shfl_down(val.de, offset);
        val.df += __shfl_down(val.df, offset);
        val.dg += __shfl_down(val.dg, offset);

        val.ee += __shfl_down(val.ee, offset);
        val.ef += __shfl_down(val.ef, offset);
        val.eg += __shfl_down(val.eg, offset);

        val.ff += __shfl_down(val.ff, offset);
        val.fg += __shfl_down(val.fg, offset);

        val.residual += __shfl_down(val.residual, offset);
        val.inliers += __shfl_down(val.inliers, offset);
    }

    return val;
}

__inline__  __device__ JtJJtrSE3 blockReduceSum(JtJJtrSE3 val)
{
    static __shared__ JtJJtrSE3 shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const JtJJtrSE3 zero = {0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0};

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        val = warpReduceSum(val);
    }

    return val;
}

__global__ void reduceSum(JtJJtrSE3 * in, JtJJtrSE3 * out, int N)
{
    JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.add(in[i]);
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

__inline__  __device__ JtJJtrSO3 warpReduceSum(JtJJtrSO3 val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.aa += __shfl_down(val.aa, offset);
        val.ab += __shfl_down(val.ab, offset);
        val.ac += __shfl_down(val.ac, offset);
        val.ad += __shfl_down(val.ad, offset);

        val.bb += __shfl_down(val.bb, offset);
        val.bc += __shfl_down(val.bc, offset);
        val.bd += __shfl_down(val.bd, offset);

        val.cc += __shfl_down(val.cc, offset);
        val.cd += __shfl_down(val.cd, offset);

        val.residual += __shfl_down(val.residual, offset);
        val.inliers += __shfl_down(val.inliers, offset);
    }

    return val;
}

__inline__  __device__ JtJJtrSO3 blockReduceSum(JtJJtrSO3 val)
{
    static __shared__ JtJJtrSO3 shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const JtJJtrSO3 zero = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void reduceSum(JtJJtrSO3 * in, JtJJtrSO3 * out, int N)
{
    JtJJtrSO3 sum = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.add(in[i]);
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

struct ICPReduction
{
    mat33 Rcurr;
    float3 tcurr;

    PtrStep<float> vmap_curr;
    PtrStep<float> nmap_curr;
    PtrStep<float> ck1maps_curr;
    PtrStep<float> ck2maps_curr;
    PtrStep<unsigned short> plane_match_map_curr;

    mat33 Rprev_inv;
    float3 tprev;
    bool icp_if_use_coorespondence_search;
    int icp_radius;
    int icp_weight_layers;
    bool icp_if_use_weight;
    bool use_sparse_icp;

    CameraModel intr;

    PtrStep<float> vmap_g_prev;
    PtrStep<float> nmap_g_prev;
    PtrStep<float> ck1maps_g_prev;
    PtrStep<float> ck2maps_g_prev;
    PtrStep<float> icpWeightmap_g_prev;
    PtrStep<unsigned short> plane_match_map_g;

    mutable PtrStepSz<int2> corresICP;
    mutable PtrStepSz<float4> cuda_out;

    mutable PtrStepSz<float3> z_thrinkMap;
    PtrStepSz<float3> lambdaMap;

    float distThres;
    float angleThres;
    float curvatureThres;

    int cols;
    int rows;
    int N;

    JtJJtrSE3 * out;

    float mu;          //Parameter for ICP step 2.1, default 10
    float p;           //We use the norm L_p, default 0.5
    int nbIterShrink;  //number of iterations for the thrink part

    //for sparse ICP
    __device__ __forceinline__ float3 thrink (float3 h) const
    {
        float alpha_a = pow((2.0f / mu) * (1.0f - p), 1.0f / (2.0f - p));
        float hTilde = alpha_a + (p / mu) * pow(alpha_a, p - 1);
        float hNorm = norm(h);

        if(hNorm <= hTilde)
            return 0 * h;

        float beta = ((alpha_a) / hNorm + 1.0f) / 2.0f;
        for(int i = 0;i< nbIterShrink; i++)
            beta = 1 - (p / mu) * pow(hNorm, p - 2.0f) * pow(beta, p - 1.0f);
        return beta * h;
    }

    __device__ __forceinline__ bool
    search (int & x, int & y, float3& n, float3& d, float3& s, int2& corres, float4& cuda_out) const
    {

        float3 vcurr = make_float3(0.0f, 0.0f, 0.0f);
        vcurr.x = vmap_curr.ptr (y)[x];
        vcurr.y = vmap_curr.ptr (y + rows)[x];
        vcurr.z = vmap_curr.ptr (y + 2 * rows)[x];

        float3 vcurr_g = Rcurr * vcurr + tcurr;
        float3 vcurr_cp = Rprev_inv * (vcurr_g - tprev);

        int2 ukr;
        ukr.x = __float2int_rn (vcurr_cp.x * intr.fx / vcurr_cp.z + intr.cx);
        ukr.y = __float2int_rn (vcurr_cp.y * intr.fy / vcurr_cp.z + intr.cy);

        if(ukr.x < 0 || ukr.y < 0 || ukr.x >= cols || ukr.y >= rows || vcurr_cp.z < 0)
            return false;

        float3 ncurr;
        ncurr.x = nmap_curr.ptr (y)[x];
        ncurr.y = nmap_curr.ptr (y + rows)[x];
        ncurr.z = nmap_curr.ptr (y + 2 * rows)[x];

        float3 ncurr_g = Rcurr * ncurr;
        float ck1curr_value = ck1maps_curr.ptr(y + 3 * rows)[x];
        float ck2curr_value = ck2maps_curr.ptr(y + 3 * rows)[x];

        //cuda_out.x = vcurr.x;  cuda_out.y = vcurr.y; cuda_out.z = vcurr.z; cuda_out.w = ck2curr_value;
        if(isnan(vcurr.x) || isnan(ncurr.x) ||isnan(ck1curr_value) || isnan(ck2curr_value))
            return false;

        //search in the neighbor for valid points, ignore points out of the minimun condition
        int2 ukr_nb[25];
        float3 vprev_g_nb[25];
        float3 nprev_g_nb[25];
        //float c_m[25];
        //float3 ck1g_prev_vector_nb[25], ck2g_prev_vector_nb[25];
        float ck1g_prev_value_nb[25], ck2g_prev_value_nb[25];
        //neighbor count
        const int R = icp_if_use_coorespondence_search ? icp_radius : 0;
        const int D = R * 2 + 1;

        int count_nb = 0;
        float D_p_R = -1e8;
        for(int cy = int(ukr.y - D / 2) ; cy < int(ukr.y + D / 2 + 1) ; ++cy){
            for(int cx = int(ukr.x - D / 2); cx < int(ukr.x + D / 2 + 1); ++cx){
                if(cx < 0 || cy < 0 || cx >= int(cols) || cy >= int(rows))
                    continue;
                float3 v_prev = make_float3(0.0f, 0.0f, 0.0f);
                v_prev.x = __ldg(&vmap_g_prev.ptr (cy)[cx]);
                v_prev.y = __ldg(&vmap_g_prev.ptr (cy + rows)[cx]);
                v_prev.z = __ldg(&vmap_g_prev.ptr (cy + 2 * rows)[cx]);
                float3 n_prev = make_float3(0.0f, 0.0f, 0.0f);
                n_prev.x = __ldg(&nmap_g_prev.ptr (cy)[cx]);
                n_prev.y = __ldg(&nmap_g_prev.ptr (cy + rows)[cx]);
                n_prev.z = __ldg(&nmap_g_prev.ptr (cy + 2 * rows)[cx]);

                float ck1_val = __ldg(&ck1maps_g_prev.ptr(cy + 3 * rows)[cx]);
                float ck2_val = __ldg(&ck2maps_g_prev.ptr(cy + 3 * rows)[cx]);

                float dist = norm(v_prev - vcurr_g);
                float sine = norm(cross(ncurr_g, n_prev));
                //global map points with invalid curvature
                if(isnan(v_prev.x) || isnan(n_prev.x) ||isnan(ck1_val) || isnan(ck2_val))
                    continue;
                else if(sine > angleThres  /* 20 degree */ || dist > distThres /*0.1*/
                        /*|| fabs(ck1_val - ck1curr_value) > curvatureThres
                        || fabs(ck2_val - ck2curr_value) > curvatureThres*/){
                    continue;
                }
                ukr_nb[count_nb].x = cx;
                ukr_nb[count_nb].y = cy;
                vprev_g_nb[count_nb] = v_prev;
                nprev_g_nb[count_nb] = n_prev;
                ck1g_prev_value_nb[count_nb] = ck1_val;
                ck2g_prev_value_nb[count_nb] = ck2_val;
                if(dist > D_p_R)
                    D_p_R = dist;
                count_nb++;
            }
        }
        //if neighbors not find, return false
        if(count_nb == 0)
            return false;

        float3 vprev_g;
        float3 nprev_g;
        bool if_found = false;
        float p_smallest = 1e8;

        //select the most similar one
        for(int i = 0; i < count_nb; i++){
            float dist = norm (vprev_g_nb[i] - vcurr_g);
            float sine = norm (cross(ncurr_g, nprev_g_nb[i]));
            float cose = dot(nprev_g_nb[i], ncurr_g);

            //search correspondence on a k * k window
            float ckmax = fabs(ck1g_prev_value_nb[i]) > fabs(ck2g_prev_value_nb[i]) ? fabs(ck1g_prev_value_nb[i]) : fabs(ck2g_prev_value_nb[i]);
            float D_p = dist / D_p_R;
            float D_n = 1 - dot(nprev_g_nb[i], ncurr_g);
            float D_c = 1 - exp(-fabs(ck1g_prev_value_nb[i] - ck1curr_value) / ckmax) * exp(-fabs(ck2g_prev_value_nb[i] - ck2curr_value) / ckmax);
            float w1 = 0.333;
            float w2 = 0.333;
            float w3 = 0.333;
            float p = icp_if_use_coorespondence_search ? w1 * D_p + w2 * D_n + w3 * D_c : 1;
            if(p < p_smallest){
                corres = ukr_nb[i];
                vprev_g = vprev_g_nb[i];
                nprev_g = nprev_g_nb[i];
                p_smallest = p;
            }
            if_found = true;
        }
        n = nprev_g;
        d = vprev_g;
        s = vcurr_g;
        return if_found;
    }

    __device__ __forceinline__ JtJJtrSE3
    getProducts(int & i) const
    {
        int y = i / cols;
        int x = i - (y * cols);

        float3 n_cp, d_cp, s_cp;

        int2 corresICP_;

        corresICP_.x = -1;
        corresICP_.y = -1;

        float4 cuda_out_f;
        float weight = 1.0;

        //seach correspondences, a target point pose, and cooresponding normal
        bool found_coresp = search (x, y, n_cp, d_cp, s_cp, corresICP_, cuda_out_f);

        corresICP.ptr(y)[x].x = corresICP_.x;       //output coorespondence
        corresICP.ptr(y)[x].y = corresICP_.y;
////        cuda_out.ptr(y)[x].x = vmap_curr.ptr(y)[x];
////        cuda_out.ptr(y)[x].y = vmap_curr.ptr (y + rows)[x];
////        cuda_out.ptr(y)[x].z = vmap_curr.ptr (y + rows * 2)[x];
////        cuda_out.ptr(y)[x].w = ck1maps_curr.ptr(y)[x];
//        //weight = 1.0;
//        cuda_out.ptr(y)[x].x = vmap_g_prev.ptr(y)[x];
//        cuda_out.ptr(y)[x].y = vmap_g_prev.ptr (y + rows)[x];
//        cuda_out.ptr(y)[x].z = vmap_g_prev.ptr (y + rows * 2)[x];
//        //cuda_out.ptr(y)[x].w = icpWeightmap_g_prev.ptr(y)[x];
//        cuda_out.ptr(y)[x].w = cuda_out_f.w;
        //weight  = cuda_out_f.w;
        float row[7] = {0, 0, 0, 0, 0, 0, 0};

        z_thrinkMap.ptr(y)[x] = make_float3(0.0f, 0.0f, 0.0f);
        if(found_coresp)
        {
            s_cp = Rprev_inv * (s_cp - tprev);
            d_cp = Rprev_inv * (d_cp - tprev);
            n_cp = Rprev_inv * (n_cp);

            //use sparse ICP
            if(use_sparse_icp)
            {
                float3 lambda = lambdaMap.ptr(y)[x];
                float3 h = s_cp - d_cp + lambda / mu;
                float3 z = thrink(h);

                d_cp = d_cp + z - lambda / mu;
                z_thrinkMap.ptr(y)[x] = z;

                cuda_out.ptr(y)[x].x = lambda.x;
                cuda_out.ptr(y)[x].y = lambda.y;
                cuda_out.ptr(y)[x].z = lambda.z;
                cuda_out.ptr(y)[x].w = norm(z);
            }

            if(icp_if_use_weight)
            {
                float w = icpWeightmap_g_prev.ptr(corresICP_.y)[corresICP_.x];
                if(!isnan(w))
                      weight = w;
                else{
                      weight = 0.0;
                    }
            }
            //A
            *(float3*)&row[0] = n_cp;
            *(float3*)&row[3] = cross(s_cp, n_cp);
            //b
            row[6] = dot(n_cp, s_cp - d_cp);
            cuda_out.ptr(y)[x].x = norm(s_cp - d_cp);
        }

        JtJJtrSE3 values = {weight  * row[0] * row[0],
                            weight  * row[0] * row[1],
                            weight  * row[0] * row[2],
                            weight  * row[0] * row[3],
                            weight  * row[0] * row[4],
                            weight  * row[0] * row[5],
                            weight  * row[0] * row[6],

                            weight  * row[1] * row[1],
                            weight  * row[1] * row[2],
                            weight  * row[1] * row[3],
                            weight  * row[1] * row[4],
                            weight  * row[1] * row[5],
                            weight  * row[1] * row[6],

                            weight  * row[2] * row[2],
                            weight  * row[2] * row[3],
                            weight  * row[2] * row[4],
                            weight  * row[2] * row[5],
                            weight  * row[2] * row[6],

                            weight  * row[3] * row[3],
                            weight  * row[3] * row[4],
                            weight  * row[3] * row[5],
                            weight  * row[3] * row[6],

                            weight  * row[4] * row[4],
                            weight  * row[4] * row[5],
                            weight  * row[4] * row[6],

                            weight  * row[5] * row[5],
                            weight  * row[5] * row[6],

                            weight  * row[6] * row[6],
                            found_coresp};

        return values;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {

            JtJJtrSE3 val = getProducts(i);

            sum.add(val);
        }

        sum = blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void icpKernel(const ICPReduction icp)
{
    icp();
}

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
             int blocks)
{
    int cols = vmap_curr.cols ();
    int rows = vmap_curr.rows () / 4;

    ICPReduction icp;

    icp.Rcurr = Rcurr;
    icp.tcurr = tcurr;

    icp.vmap_curr = vmap_curr;
    icp.nmap_curr = nmap_curr;
    icp.ck1maps_curr = ck1maps_curr;
    icp.ck2maps_curr = ck2maps_curr;
    icp.plane_match_map_curr = plane_match_map_curr;
    icp.icp_weight_layers = icp_weight_layers;

    icp.Rprev_inv = Rprev_inv;
    icp.tprev = tprev;

    icp.intr = intr;

    icp.vmap_g_prev = vmap_g_prev;
    icp.nmap_g_prev = nmap_g_prev;
    icp.ck1maps_g_prev = ck1maps_g_prev;
    icp.ck2maps_g_prev = ck2maps_g_prev;
    icp.icpWeightmap_g_prev = icpWeightmap_g_prev;
    icp.plane_match_map_g = plane_match_map_g;

    icp.distThres = distThres;
    icp.angleThres = angleThres;
    icp.curvatureThres = curvatureThres;

    icp.corresICP = corresICP;
    icp.cuda_out = cuda_out;

    icp.z_thrinkMap = z_thrinkMap;
    icp.lambdaMap = lambdaMap;
    icp.p = 0.5;
    icp.mu = 10.0;
    icp.nbIterShrink = 3;

    icp.icp_if_use_coorespondence_search = icp_if_use_coorespondence_search;
    icp.icp_radius = icp_search_radius;
    icp.icp_if_use_weight = icp_if_use_weight;
    icp.use_sparse_icp = use_sparse_icp;

    icp.cols = cols;
    icp.rows = rows;

    icp.N = cols * rows;
    icp.out = sum;

    icpKernel<<<blocks, threads>>>(icp);

    reduceSum<<<1, MAX_THREADS>>>(sum, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    float host_data[32];
    out.download((JtJJtrSE3 *)&host_data[0]);

    int shift = 0;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = i; j < 7; ++j)
        {
            float value = host_data[shift++];
            if (j == 6)
                vectorB_host[i] = value;
            else
                //symetric matrix
                matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
        }
    }

    residual_host[0] = host_data[27];
    residual_host[1] = host_data[28];
}

#define FLT_EPSILON ((float)1.19209290E-07F)

struct RGBReduction
{
    //computed at the rasiual step
    PtrStepSz<DataTerm> corresImg;

    float sigma;
    PtrStepSz<float3> cloud; //xyz
    float fx;  //focal length
    float fy;  //focal length
    PtrStepSz<short> dIdx;
    PtrStepSz<short> dIdy;
    float sobelScale;

    bool rgb_use_RGBGradient_weight;

    int cols;
    int rows;
    int N;

    JtJJtrSE3 * out;

    __device__ __forceinline__ JtJJtrSE3
    getProducts(int & i) const
    {
        const DataTerm & corresp = corresImg.data[i];

        //valid correpondence
        bool found_coresp = corresp.valid;

        float row[7];

        float rgb_weight = 1.0;

        if(found_coresp)
        {
            float w = sigma + std::abs(corresp.diff);

            w = w > FLT_EPSILON ? 1.0f / w : 1.0f;

            //Signals RGB only tracking, so we should only
            if(sigma == -1)
            {
                w = 1;
            }

            row[6] = -w * corresp.diff;  //residual

            float3 cloudPoint = {cloud.ptr(corresp.zero.y)[corresp.zero.x].x,
                                 cloud.ptr(corresp.zero.y)[corresp.zero.x].y,
                                 cloud.ptr(corresp.zero.y)[corresp.zero.x].z};

            float invz = 1.0 / cloudPoint.z;
            float dI_dx_val = w * sobelScale * dIdx.ptr(corresp.one.y)[corresp.one.x];
            float dI_dy_val = w * sobelScale * dIdy.ptr(corresp.one.y)[corresp.one.x];
            float v0 = dI_dx_val * fx * invz;
            float v1 = dI_dy_val * fy * invz;
            float v2 = -(v0 * cloudPoint.x + v1 * cloudPoint.y) * invz;

            row[0] = v0;
            row[1] = v1;
            row[2] = v2;
            row[3] = -cloudPoint.z * v1 + cloudPoint.y * v2;
            row[4] =  cloudPoint.z * v0 - cloudPoint.x * v2;
            row[5] = -cloudPoint.y * v0 + cloudPoint.x * v1;

            if(rgb_use_RGBGradient_weight)
            {
               float grad_mag = sqrt(dI_dx_val * dI_dx_val + dI_dy_val * dI_dy_val);
               rgb_weight = exp(-0.5 * (10 / grad_mag) * (10 / grad_mag));
            }

        }
        else
        {
            row[0] = row[1] = row[2] = row[3] = row[4] = row[5] = row[6] = 0.f;
        }

        JtJJtrSE3 values = {rgb_weight * row[0] * row[0],
                            rgb_weight * row[0] * row[1],
                            rgb_weight * row[0] * row[2],
                            rgb_weight * row[0] * row[3],
                            rgb_weight * row[0] * row[4],
                            rgb_weight * row[0] * row[5],
                            rgb_weight * row[0] * row[6],

                            rgb_weight * row[1] * row[1],
                            rgb_weight * row[1] * row[2],
                            rgb_weight * row[1] * row[3],
                            rgb_weight * row[1] * row[4],
                            rgb_weight * row[1] * row[5],
                            rgb_weight * row[1] * row[6],

                            rgb_weight * row[2] * row[2],
                            rgb_weight * row[2] * row[3],
                            rgb_weight * row[2] * row[4],
                            rgb_weight * row[2] * row[5],
                            rgb_weight * row[2] * row[6],

                            rgb_weight * row[3] * row[3],
                            rgb_weight * row[3] * row[4],
                            rgb_weight * row[3] * row[5],
                            rgb_weight * row[3] * row[6],

                            rgb_weight * row[4] * row[4],
                            rgb_weight * row[4] * row[5],
                            rgb_weight * row[4] * row[6],

                            rgb_weight * row[5] * row[5],
                            rgb_weight * row[5] * row[6],

                            rgb_weight * row[6] * row[6],
                            found_coresp};

        return values;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        JtJJtrSE3 sum = {0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            JtJJtrSE3 val = getProducts(i);

            sum.add(val);
        }

        sum = blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void rgbKernel (const RGBReduction rgb)
{
    rgb();
}

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
             int blocks)
{
    RGBReduction rgb;

    rgb.corresImg = corresImg;
    rgb.cols = corresImg.cols();
    rgb.rows = corresImg.rows();
    rgb.sigma = sigma;
    rgb.cloud = cloud;
    rgb.fx = fx;
    rgb.fy = fy;
    rgb.dIdx = dIdx;
    rgb.dIdy = dIdy;
    rgb.rgb_use_RGBGradient_weight = rgb_use_RGBGradient_weight;
    rgb.sobelScale = sobelScale;
    rgb.N = rgb.cols * rgb.rows;
    rgb.out = sum;

    rgbKernel<<<blocks, threads>>>(rgb);

    reduceSum<<<1, MAX_THREADS>>>(sum, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    float host_data[32];
    out.download((JtJJtrSE3 *)&host_data[0]);

    int shift = 0;
    for (int i = 0; i < 6; ++i)
    {
        for (int j = i; j < 7; ++j)
        {
            float value = host_data[shift++];
            if (j == 6)
                vectorB_host[i] = value;
            else
                matrixA_host[j * 6 + i] = matrixA_host[i * 6 + j] = value;
        }
    }
}

__inline__  __device__ int2 warpReduceSum(int2 val)
{
    for(int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        val.x += __shfl_down(val.x, offset);
        val.y += __shfl_down(val.y, offset);
    }

    return val;
}

__inline__  __device__ int2 blockReduceSum(int2 val)
{
    static __shared__ int2 shared[32];

    int lane = threadIdx.x % warpSize;

    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    //write reduced value to shared memory
    if(lane == 0)
    {
        shared[wid] = val;
    }
    __syncthreads();

    const int2 zero = {0, 0};

    //ensure we only grab a value from shared memory if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : zero;

    if(wid == 0)
    {
        val = warpReduceSum(val);
    }

    return val;
}

__global__ void reduceSum(int2 * in, int2 * out, int N)
{
    int2 sum = {0, 0};

    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        sum.x += in[i].x;
        sum.y += in[i].y;
    }

    sum = blockReduceSum(sum);

    if(threadIdx.x == 0)
    {
        out[blockIdx.x] = sum;
    }
}

struct RGBResidual
{
    float minScale;

    PtrStepSz<short> dIdx;
    PtrStepSz<short> dIdy;

    PtrStepSz<float> lastDepth;
    PtrStepSz<float> nextDepth;

    PtrStepSz<unsigned char> lastImage;
    PtrStepSz<unsigned char> nextImage;

    mutable PtrStepSz<DataTerm> corresImg;

    float maxDepthDelta;

    float3 kt;
    mat33 krkinv;

    int cols;
    int rows;
    int N;

    int pitch;
    int imgPitch;

    int2 * out;

    __device__ __forceinline__ int2
    getProducts(int k) const
    {
        int i = k / cols;
        int j0 = k - (i * cols);

        int2 value = {0, 0};

        DataTerm corres;

        corres.valid = false;

        if(i >= 0 && i < rows && j0 >= 0 && j0 < cols)
        {
            if(j0 < cols - 5 && i < rows - 1)
            {
                bool valid = true;

                //this is not a isolated pixel./
                for(int u = max(i - 2, 0); u < min(i + 2, rows); u++)
                {
                    for(int v = max(j0 - 2, 0); v < min(j0 + 2, cols); v++)
                    {
                        valid = valid && (nextImage.ptr(u)[v] > 0);
                    }
                }

                if(valid)
                {
                    short * ptr_input_x = (short*) ((unsigned char*) dIdx.data + i * pitch);
                    short * ptr_input_y = (short*) ((unsigned char*) dIdy.data + i * pitch);

                    short valx = ptr_input_x[j0];
                    short valy = ptr_input_y[j0];
                    float mTwo = (valx * valx) + (valy * valy);

                    if(mTwo >= minScale)
                    {
                        int y = i;
                        int x = j0;

                        float d1 = nextDepth.ptr(y)[x];

                        if(!isnan(d1))
                        {
                            float transformed_d1 = (float)(d1 * (krkinv.data[2].x * x + krkinv.data[2].y * y + krkinv.data[2].z) + kt.z);
                            int u0 = __float2int_rn((d1 * (krkinv.data[0].x * x + krkinv.data[0].y * y + krkinv.data[0].z) + kt.x) / transformed_d1);
                            int v0 = __float2int_rn((d1 * (krkinv.data[1].x * x + krkinv.data[1].y * y + krkinv.data[1].z) + kt.y) / transformed_d1);

                            if(u0 >= 0 && v0 >= 0 && u0 < lastDepth.cols && v0 < lastDepth.rows)
                            {
                                float d0 = lastDepth.ptr(v0)[u0];

                                if(d0 > 0 && std::abs(transformed_d1 - d0) <= maxDepthDelta && lastImage.ptr(v0)[u0] != 0)
                                {
                                    corres.zero.x = u0;
                                    corres.zero.y = v0;
                                    corres.one.x = x;
                                    corres.one.y = y;
                                    corres.diff = static_cast<float>(nextImage.ptr(y)[x]) - static_cast<float>(lastImage.ptr(v0)[u0]);
                                    corres.valid = true;
                                    value.x = 1;
                                    value.y = corres.diff * corres.diff;
                                }
                            }
                        }
                    }
                }
            }
        }

        corresImg.data[k] = corres;

        return value;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        int2 sum = {0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            int2 val = getProducts(i);
            sum.x += val.x;
            sum.y += val.y;
        }

        sum = blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void residualKernel (const RGBResidual rgb)
{
    rgb();
}

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
                        int blocks)
{
    int cols = nextImage.cols ();
    int rows = nextImage.rows ();

    RGBResidual rgb;

    rgb.minScale = minScale;

    rgb.dIdx = dIdx;
    rgb.dIdy = dIdy;

    rgb.lastDepth = lastDepth;
    rgb.nextDepth = nextDepth;

    rgb.lastImage = lastImage;
    rgb.nextImage = nextImage;

    rgb.corresImg = corresImg;

    rgb.maxDepthDelta = maxDepthDelta;

    rgb.kt = kt;
    rgb.krkinv = krkinv;

    rgb.cols = cols;
    rgb.rows = rows;
    rgb.pitch = dIdx.step();
    rgb.imgPitch = nextImage.step();

    rgb.N = cols * rows;
    rgb.out = sumResidual;

    residualKernel<<<blocks, threads>>>(rgb);

    int2 out_host = {0, 0};
    int2 * out;

    cudaMalloc(&out, sizeof(int2));
    cudaMemcpy(out, &out_host, sizeof(int2), cudaMemcpyHostToDevice);

    reduceSum<<<1, MAX_THREADS>>>(sumResidual, out, blocks);  //thread block

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    cudaMemcpy(&out_host, out, sizeof(int2), cudaMemcpyDeviceToHost);
    cudaFree(out);

    count = out_host.x;
    sigmaSum = out_host.y;
}

struct SO3Reduction
{
    PtrStepSz<unsigned char> lastImage;
    PtrStepSz<unsigned char> nextImage;

    mat33 imageBasis;
    mat33 kinv;
    mat33 krlr;
    bool gradCheck;

    int cols;
    int rows;
    int N;

    JtJJtrSO3 * out;

    __device__ __forceinline__ float2
    getGradient(const PtrStepSz<unsigned char> img, int x, int y) const
    {
        float2 gradient;

        float actu = static_cast<float>(img.ptr(y)[x]);

        float back = static_cast<float>(img.ptr(y)[x - 1]);
        float fore = static_cast<float>(img.ptr(y)[x + 1]);
        gradient.x = ((back + actu) / 2.0f) - ((fore + actu) / 2.0f);

        back = static_cast<float>(img.ptr(y - 1)[x]);
        fore = static_cast<float>(img.ptr(y + 1)[x]);
        gradient.y = ((back + actu) / 2.0f) - ((fore + actu) / 2.0f);

        return gradient;
    }

    __device__ __forceinline__ JtJJtrSO3
    getProducts(int k) const
    {
        int y = k / cols;
        int x = k - (y * cols);

        bool found_coresp = false; // for image, no need to find the correspondence

        float3 unwarpedReferencePoint = {x, y, 1.0f};

        float3 warpedReferencePoint = imageBasis * unwarpedReferencePoint;  //warped image pixel coordinate, find it and calculate the deviation

        int2 warpedReferencePixel = {__float2int_rn(warpedReferencePoint.x / warpedReferencePoint.z),
                                     __float2int_rn(warpedReferencePoint.y / warpedReferencePoint.z)};

        if(warpedReferencePixel.x >= 1 &&
           warpedReferencePixel.x < cols - 1 &&
           warpedReferencePixel.y >= 1 &&
           warpedReferencePixel.y < rows - 1 &&
           x >= 1 &&
           x < cols - 1 &&
           y >= 1 &&
           y < rows - 1)
        {
            found_coresp = true;  //in the image range, we get find correspondence in pictures
        }

        float row[4];
        row[0] = row[1] = row[2] = row[3] = 0.f;

        if(found_coresp)
        {
            float2 gradNext = getGradient(nextImage, warpedReferencePixel.x, warpedReferencePixel.y); //calculate image correpondence
            float2 gradLast = getGradient(lastImage, x, y);

            float gx = (gradNext.x + gradLast.x) / 2.0f;
            float gy = (gradNext.y + gradLast.y) / 2.0f;

            float3 point = kinv * unwarpedReferencePoint;  //get the point in space based on camera intrisics

            float z2 = point.z * point.z;

            float a = krlr.data[0].x;
            float b = krlr.data[0].y;
            float c = krlr.data[0].z;

            float d = krlr.data[1].x;
            float e = krlr.data[1].y;
            float f = krlr.data[1].z;

            float g = krlr.data[2].x;
            float h = krlr.data[2].y;
            float i = krlr.data[2].z;

            //Aren't jacobians great fun
            float3 leftProduct = {((point.z * (d * gy + a * gx)) - (gy * g * y) - (gx * g * x)) / z2,
                                  ((point.z * (e * gy + b * gx)) - (gy * h * y) - (gx * h * x)) / z2,
                                  ((point.z * (f * gy + c * gx)) - (gy * i * y) - (gx * i * x)) / z2};

            float3 jacRow = cross(leftProduct, point);

            row[0] = jacRow.x;
            row[1] = jacRow.y;
            row[2] = jacRow.z;
            row[3] = -(static_cast<float>(nextImage.ptr(warpedReferencePixel.y)[warpedReferencePixel.x]) - static_cast<float>(lastImage.ptr(y)[x]));
        }

        JtJJtrSO3 values = {row[0] * row[0],
                            row[0] * row[1],
                            row[0] * row[2],
                            row[0] * row[3],

                            row[1] * row[1],
                            row[1] * row[2],
                            row[1] * row[3],

                            row[2] * row[2],
                            row[2] * row[3],

                            row[3] * row[3],
                            found_coresp};

        return values;
    }

    __device__ __forceinline__ void
    operator () () const
    {
        JtJJtrSO3 sum = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
        {
            JtJJtrSO3 val = getProducts(i);

            sum.add(val);
        }

        sum = blockReduceSum(sum);

        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = sum;
        }
    }
};

__global__ void so3Kernel (const SO3Reduction so3)
{
    so3();
}

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
             int blocks)
{
    int cols = nextImage.cols();
    int rows = nextImage.rows();

    SO3Reduction so3;

    so3.lastImage = lastImage;

    so3.nextImage = nextImage;

    so3.imageBasis = imageBasis;
    so3.kinv = kinv;
    so3.krlr = krlr;

    so3.cols = cols;
    so3.rows = rows;

    so3.N = cols * rows;

    so3.out = sum;

    so3Kernel<<<blocks, threads>>>(so3);

    reduceSum<<<1, MAX_THREADS>>>(sum, out, blocks);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());

    float host_data[11];
    out.download((JtJJtrSO3 *)&host_data[0]);

    int shift = 0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = i; j < 4; ++j)
        {
            float value = host_data[shift++];
            if (j == 3)
                vectorB_host[i] = value;
            else
                matrixA_host[j * 3 + i] = matrixA_host[i * 3 + j] = value;
        }
    }

    residual_host[0] = host_data[9];
    residual_host[1] = host_data[10];
}
