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


__global__ void pyrDownGaussKernel (const PtrStepSz<float> src, PtrStepSz<float> dst, float sigma_color)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;   //to form a 5*5 patch

    float center = src.ptr (2 * y)[2 * x];

    int x_mi = max(0, 2*x - D/2) - 2*x;
    int y_mi = max(0, 2*y - D/2) - 2*y;

    int x_ma = min(src.cols, 2*x -D/2+D) - 2*x;
    int y_ma = min(src.rows, 2*y -D/2+D) - 2*y;

    float sum = 0;
    float wall = 0;

    float weights[] = {0.375f, 0.25f, 0.0625f};

    for(int yi = y_mi; yi < y_ma; ++yi)
        for(int xi = x_mi; xi < x_ma; ++xi)
        {
            float val = src.ptr (2*y + yi)[2*x + xi];

            if (abs (val - center) < 3 * sigma_color)
            {
                sum += val * weights[abs(xi)] * weights[abs(yi)];
                wall += weights[abs(xi)] * weights[abs(yi)];
            }
        }


    dst.ptr (y)[x] = sum / wall;
}

void pyrDown(const DeviceArray2D<float> & src, DeviceArray2D<float> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    const float sigma_color = 30;

    pyrDownGaussKernel<<<grid, block>>>(src, dst, sigma_color);
    cudaSafeCall(cudaGetLastError ());
};

__global__ void computeVmapKernel(const PtrStepSz<float> depth, PtrStep<float> vmap, float fx_inv, float fy_inv, float cx, float cy, float depthCutoff, float mDepthMapFactor)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if(u < depth.cols && v < depth.rows)
    {
        float z = depth.ptr (v)[u] * mDepthMapFactor;    // load and convert: mm -> meters

        if(z != 0 && z < depthCutoff)
        {
            float vx = z * (u - cx) * fx_inv;
            float vy = z * (v - cy) * fy_inv;
            float vz = z;

            vmap.ptr (v                 )[u] = vx;
            vmap.ptr (v + depth.rows    )[u] = vy;
            vmap.ptr (v + depth.rows * 2)[u] = vz;
            //add confidence here;
            vmap.ptr (v + depth.rows * 3)[u] = 1.0;
        }
        else
        {
            //just make the first as nan
            vmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        }
    }
}

void createVMap(const CameraModel& intr, const DeviceArray2D<float> & depth, DeviceArray2D<float> & vmap, const float depthCutoff, const float mDepthMapFactor)
{
    vmap.create (depth.rows () * 4, depth.cols ());

    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (depth.cols (), block.x);
    grid.y = getGridDim (depth.rows (), block.y);

    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    computeVmapKernel<<<grid, block>>>(depth, vmap, 1.f / fx, 1.f / fy, cx, cy, depthCutoff, mDepthMapFactor);
    cudaSafeCall (cudaGetLastError ());
}

__global__ void computeNmapKernel(int rows, int cols, const PtrStep<float> vmap, PtrStep<float> nmap)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    int v = threadIdx.y + blockIdx.y * blockDim.y;

    if (u >= cols || v >= rows)
        return;

    if (u == cols - 1 || v == rows - 1)
    {
        nmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
        return;
    }

    //this is the forward difference
    float3 v00, v01, v10;
    v00.x = vmap.ptr (v  )[u];
    v01.x = vmap.ptr (v  )[u + 1];
    v10.x = vmap.ptr (v + 1)[u];

    if (!isnan (v00.x) && !isnan (v01.x) && !isnan (v10.x))
    {
        v00.y = vmap.ptr (v + rows)[u];
        v01.y = vmap.ptr (v + rows)[u + 1];
        v10.y = vmap.ptr (v + 1 + rows)[u];

        v00.z = vmap.ptr (v + 2 * rows)[u];
        v01.z = vmap.ptr (v + 2 * rows)[u + 1];
        v10.z = vmap.ptr (v + 1 + 2 * rows)[u];

        float3 r = normalized (cross (v01 - v00, v10 - v00));

        nmap.ptr (v       )[u] = r.x;
        nmap.ptr (v + rows)[u] = r.y;
        nmap.ptr (v + 2 * rows)[u] = r.z;

        //add radius here
        nmap.ptr (v + 3 * rows)[u] = 1.0;
    }
    else
        nmap.ptr (v)[u] = __int_as_float(0x7fffffff); /*CUDART_NAN_F*/
}

void createNMap(const DeviceArray2D<float>& vmap, DeviceArray2D<float>& nmap)
{
    nmap.create (vmap.rows (), vmap.cols ());

    int rows = vmap.rows () / 4;
    int cols = vmap.cols ();

    dim3 block (32, 8);
    dim3 grid (1, 1, 1);
    grid.x = getGridDim (cols, block.x);
    grid.y = getGridDim (rows, block.y);

    computeNmapKernel<<<grid, block>>>(rows, cols, vmap, nmap);
    cudaSafeCall (cudaGetLastError ());
}

__global__ void tranformMapsKernel(int rows, int cols, const PtrStep<float> vmap_src, const PtrStep<float> nmap_src,
                                   const mat33 Rmat, const float3 tvec, PtrStepSz<float> vmap_dst, PtrStep<float> nmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        //vertices
        float3 vsrc, vdst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        vsrc.x = vmap_src.ptr (y)[x];

        if (!isnan (vsrc.x))
        {
            vsrc.y = vmap_src.ptr (y + rows)[x];
            vsrc.z = vmap_src.ptr (y + 2 * rows)[x];

            vdst = Rmat * vsrc + tvec;

            vmap_dst.ptr (y + rows)[x] = vdst.y;
            vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;
            vmap_dst.ptr (y + 3 * rows)[x] = vmap_src.ptr (y + 3 * rows)[x];
        }

        vmap_dst.ptr (y)[x] = vdst.x;

        //normals
        float3 nsrc, ndst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        nsrc.x = nmap_src.ptr (y)[x];

        if (!isnan (nsrc.x))
        {
            nsrc.y = nmap_src.ptr (y + rows)[x];
            nsrc.z = nmap_src.ptr (y + 2 * rows)[x];

            ndst = Rmat * nsrc;

            nmap_dst.ptr (y + rows)[x] = ndst.y;
            nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
            nmap_dst.ptr (y + 3 * rows)[x] = nmap_src.ptr (y + 3 * rows)[x];
        }

        nmap_dst.ptr (y)[x] = ndst.x;
    }
}

void tranformMaps(const DeviceArray2D<float>& vmap_src,
                  const DeviceArray2D<float>& nmap_src,
                  const mat33& Rmat, const float3& tvec,
                  DeviceArray2D<float>& vmap_dst, DeviceArray2D<float>& nmap_dst)
{
    int cols = vmap_src.cols();
    int rows = vmap_src.rows() / 4;

    vmap_dst.create(rows * 4, cols);
    nmap_dst.create(rows * 4, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    tranformMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, Rmat, tvec, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());
}

__global__ void tranformCurvMapsKernel(int rows, int cols, const PtrStep<float> curvk1_src, const PtrStep<float> curvk2_src,
                                   const mat33 Rmat, const float3 tvec, PtrStepSz<float> curvk1_dst, PtrStep<float> curvk2_dst)
{
    //data index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        float3 curvk1_vec_src, curvk1_vec_dst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        float1 curvk1_val_src, curvk1_val_dst = make_float1(__int_as_float(0x7fffffff));
        curvk1_vec_src.x = curvk1_src.ptr(y)[x];
        if(!isnan(curvk1_vec_src.x))
        {
            curvk1_vec_src.y = curvk1_src.ptr(y + rows)[x];
            curvk1_vec_src.z = curvk1_src.ptr(y + 2 * rows)[x];
            curvk1_val_src.x = curvk1_src.ptr(y + 3 * rows)[x];

            curvk1_vec_dst = Rmat * curvk1_vec_src;

            curvk1_dst.ptr(y + rows)[x] = curvk1_vec_dst.y;
            curvk1_dst.ptr(y + 2 * rows)[x] = curvk1_vec_dst.z;
            curvk1_dst.ptr(y + 3 * rows)[x] = curvk1_val_src.x;
        }
        curvk1_dst.ptr(y)[x] = curvk1_vec_dst.x;

        float3 curvk2_vec_src, curvk2_vec_dst = make_float3 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));
        float1 curvk2_val_src, curvk2_val_dst = make_float1(__int_as_float(0x7fffffff));
        curvk2_vec_src.x = curvk2_src.ptr(y)[x];
        if(!isnan(curvk2_vec_src.x))
        {
            curvk2_vec_src.y = curvk2_src.ptr(y + rows)[x];
            curvk2_vec_src.z = curvk2_src.ptr(y + 2 * rows)[x];
            curvk2_val_src.x = curvk2_src.ptr(y + 3 * rows)[x];

            curvk2_vec_dst = Rmat * curvk2_vec_src;

            curvk2_dst.ptr(y + rows)[x] = curvk2_vec_dst.y;
            curvk2_dst.ptr(y + 2 * rows)[x] = curvk2_vec_dst.z;
            curvk2_dst.ptr(y + 3 * rows)[x] = curvk2_val_src.x;
        }
        curvk2_dst.ptr(y)[x] = curvk2_vec_dst.x;
    }
}

void transformCurvMaps(const DeviceArray2D<float>& curvk1_src,
                       const DeviceArray2D<float>& curvk2_src,
                       const mat33& Rmat, const float3& tvec,
                       DeviceArray2D<float>& curvk1_dst, DeviceArray2D<float>& curvk2_dst)
{
    int cols = curvk1_src.cols();
    int rows = curvk1_src.rows() / 4;

    curvk1_dst.create(rows * 4, cols);
    curvk2_dst.create(rows * 4, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    tranformCurvMapsKernel<<<grid, block>>>(rows, cols, curvk1_src, curvk2_src, Rmat, tvec, curvk1_dst, curvk2_dst);
    cudaSafeCall(cudaGetLastError());
}

__global__ void copyMapsKernel(int rows, int cols, const float * vmap_src, const float * nmap_src,
                               PtrStepSz<float> vmap_dst, PtrStep<float> nmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < cols && y < rows)
    {
        //vertex
        float4 vsrc, vdst = make_float4 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        vsrc.x = vmap_src[y * cols * 4 + (x * 4) + 0];
        vsrc.y = vmap_src[y * cols * 4 + (x * 4) + 1];
        vsrc.z = vmap_src[y * cols * 4 + (x * 4) + 2];
        vsrc.w = vmap_src[y * cols * 4 + (x * 4) + 3];

        float4 nsrc, ndst = make_float4 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        nsrc.x = nmap_src[y * cols * 4 + (x * 4) + 0];
        nsrc.y = nmap_src[y * cols * 4 + (x * 4) + 1];
        nsrc.z = nmap_src[y * cols * 4 + (x * 4) + 2];
        nsrc.w = nmap_src[y * cols * 4 + (x * 4) + 3];
        //valid normal, length less than 1
        if(!(vsrc.z == 0) && nsrc.w > 0)
        {
            vdst = vsrc;
            ndst = nsrc;
        }

        vmap_dst.ptr (y)[x] = vdst.x;
        vmap_dst.ptr (y + rows)[x] = vdst.y;
        vmap_dst.ptr (y + 2 * rows)[x] = vdst.z;
        vmap_dst.ptr (y + 3 * rows)[x] = vdst.w;

        nmap_dst.ptr (y)[x] = ndst.x;
        nmap_dst.ptr (y + rows)[x] = ndst.y;
        nmap_dst.ptr (y + 2 * rows)[x] = ndst.z;
        nmap_dst.ptr (y + 3 * rows)[x] = ndst.w;
    }
}

void copyMaps(const DeviceArray<float>& vmap_src,
              const DeviceArray<float>& nmap_src,
              DeviceArray2D<float>& vmap_dst,
              DeviceArray2D<float>& nmap_dst)
{
    int cols = vmap_dst.cols();
    int rows = vmap_dst.rows() / 4;

    vmap_dst.create(rows * 4, cols);
    nmap_dst.create(rows * 4, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    copyMapsKernel<<<grid, block>>>(rows, cols, vmap_src, nmap_src, vmap_dst, nmap_dst);
    cudaSafeCall(cudaGetLastError());
}

__global__ void copyCurvatureMapKernel(int rows, int cols, const float * cmap_src,
                                       PtrStepSz<float> cmap_dst, float curvatureThreshold)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < cols && y < rows)
    {
        float4 csrc, cdst = make_float4 (__int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff), __int_as_float(0x7fffffff));

        //first principal curvature vector * 3, then principal curvature value
        csrc.x = cmap_src[y * cols * 4 + (x * 4) + 0];
        csrc.y = cmap_src[y * cols * 4 + (x * 4) + 1];
        csrc.z = cmap_src[y * cols * 4 + (x * 4) + 2];
        csrc.w = cmap_src[y * cols * 4 + (x * 4) + 3];

        //we just keep curvature value between range [-300, 300]
        if(csrc.w < curvatureThreshold && csrc.w > - curvatureThreshold && !isnan(csrc.w))
        {
            cdst = csrc;
        }
        cmap_dst.ptr(y)[x] = cdst.x;
        cmap_dst.ptr(y + rows)[x] = cdst.y;
        cmap_dst.ptr(y + 2 * rows)[x] = cdst.z;
        cmap_dst.ptr(y + 3 * rows)[x] = cdst.w;
    }
}

void copyCurvatureMap(const DeviceArray<float>& cmap_src,
                      DeviceArray2D<float>& cmap_dst,
                      const float curvatureThreshold)
{
    int cols = cmap_dst.cols();
    int rows = cmap_dst.rows() / 4;

    cmap_dst.create(rows * 4, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    copyCurvatureMapKernel<<<grid, block>>>(rows, cols, cmap_src, cmap_dst, curvatureThreshold);
    cudaSafeCall(cudaGetLastError());
}


__global__ void copyicpWeightMapKernel(int rows, int cols, const float * icpwmap_src,
                                       PtrStepSz<float> icpwmap_dst)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x < cols && y < rows)
    {
       float1 icp_src, icp_dst = make_float1(__int_as_float(0x7fffffff));

       icp_src.x = icpwmap_src[y * cols + x + 0];

       if(icp_src.x > 0)
       {
           icp_dst.x = icp_src.x;
       }
       icpwmap_dst.ptr(y)[x] = icp_dst.x;
    }
}


void copyicpWeightMap(const DeviceArray<float>& icpwmap_src,
                      DeviceArray2D<float>& icpwmap_dst)
{
    int cols = icpwmap_dst.cols();
    int rows = icpwmap_dst.rows();

    icpwmap_dst.create(rows, cols);

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);

    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    copyicpWeightMapKernel<<<grid, block>>> (rows, cols, icpwmap_src, icpwmap_dst);

    cudaSafeCall(cudaGetLastError());

}

__global__ void pyrDownKernelGaussF(const PtrStepSz<float> src, PtrStepSz<float> dst, float * gaussKernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;

    float center = src.ptr (2 * y)[2 * x];

    int tx = min (2 * x - D / 2 + D, src.cols - 1);
    int ty = min (2 * y - D / 2 + D, src.rows - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
    {
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
            if(!isnan(src.ptr (cy)[cx]))
            {
                sum += src.ptr (cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
            }
        }
    }
    dst.ptr (y)[x] = (float)(sum / (float)count);
}

template<bool normalize>
__global__ void resizeMapKernel(int drows, int dcols, int srows,
                                const PtrStep<float> input, PtrStep<float> output)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dcols || y >= drows)
        return;

    const float qnan = __int_as_float(0x7fffffff);

    int xs = x * 2;
    int ys = y * 2;

    float x00 = input.ptr (ys + 0)[xs + 0];
    float x01 = input.ptr (ys + 0)[xs + 1];
    float x10 = input.ptr (ys + 1)[xs + 0];
    float x11 = input.ptr (ys + 1)[xs + 1];

    if (isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11))
    {
        output.ptr (y)[x] = qnan;
        return;
    }
    else
    {
        float3 n;

        n.x = (x00 + x01 + x10 + x11) / 4;

        float y00 = input.ptr (ys + srows + 0)[xs + 0];
        float y01 = input.ptr (ys + srows + 0)[xs + 1];
        float y10 = input.ptr (ys + srows + 1)[xs + 0];
        float y11 = input.ptr (ys + srows + 1)[xs + 1];

        n.y = (y00 + y01 + y10 + y11) / 4;

        float z00 = input.ptr (ys + 2 * srows + 0)[xs + 0];
        float z01 = input.ptr (ys + 2 * srows + 0)[xs + 1];
        float z10 = input.ptr (ys + 2 * srows + 1)[xs + 0];
        float z11 = input.ptr (ys + 2 * srows + 1)[xs + 1];

        n.z = (z00 + z01 + z10 + z11) / 4;

        float w00 = input.ptr (ys + 3 * srows + 0)[xs + 0];
        float w01 = input.ptr (ys + 3 * srows + 0)[xs + 1];
        float w10 = input.ptr (ys + 3 * srows + 1)[xs + 0];
        float w11 = input.ptr (ys + 3 * srows + 1)[xs + 1];

        //in vmap 'w' is confidence while in nmap 'w' is radius
        float w = (w00 + w01 + w10 + w11) / 4;

        if (normalize)
            n = normalized (n);

        output.ptr (y        )[x] = n.x;
        output.ptr (y + drows)[x] = n.y;
        output.ptr (y + 2 * drows)[x] = n.z;
        output.ptr (y + 3 * drows)[x] = w;
    }
}

template<bool normalize>
void resizeMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    int in_cols = input.cols ();
    int in_rows = input.rows () / 4;

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    output.create (out_rows * 4, out_cols);

    dim3 block (32, 8);
    dim3 grid (getGridDim (out_cols, block.x), getGridDim (out_rows, block.y));
    resizeMapKernel<normalize><< < grid, block>>>(out_rows, out_cols, in_rows, input, output);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

void resizeVMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    resizeMap<false>(input, output);
}

void resizeNMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    resizeMap<true>(input, output);
}


__global__ void resizeCMapKernel(int drows, int dcols, int srows,
                                const PtrStep<float> input, PtrStep<float> output)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dcols || y >= drows)
        return;

    const float qnan = __int_as_float(0x7fffffff);

    int xs = x * 2;
    int ys = y * 2;

    float x00 = input.ptr (ys + 0)[xs + 0];
    float x01 = input.ptr (ys + 0)[xs + 1];
    float x10 = input.ptr (ys + 1)[xs + 0];
    float x11 = input.ptr (ys + 1)[xs + 1];

    float w00 = input.ptr (ys + 3 * srows + 0)[xs + 0];
    float w01 = input.ptr (ys + 3 * srows + 0)[xs + 1];
    float w10 = input.ptr (ys + 3 * srows + 1)[xs + 0];
    float w11 = input.ptr (ys + 3 * srows + 1)[xs + 1];

    if (//isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11) ||
        isnan(w00) || isnan(w01) || isnan(w10) || isnan(w11))
    {
        output.ptr (y)[x] = qnan;
        output.ptr (y + 3 * drows)[x] = qnan;
        return;
    }
    else
    {
        float4 n;

        n.x = (x00 + x01 + x10 + x11) / 4;

        float y00 = input.ptr (ys + srows + 0)[xs + 0];
        float y01 = input.ptr (ys + srows + 0)[xs + 1];
        float y10 = input.ptr (ys + srows + 1)[xs + 0];
        float y11 = input.ptr (ys + srows + 1)[xs + 1];

        n.y = (y00 + y01 + y10 + y11) / 4;

        float z00 = input.ptr (ys + 2 * srows + 0)[xs + 0];
        float z01 = input.ptr (ys + 2 * srows + 0)[xs + 1];
        float z10 = input.ptr (ys + 2 * srows + 1)[xs + 0];
        float z11 = input.ptr (ys + 2 * srows + 1)[xs + 1];

        n.z = (z00 + z01 + z10 + z11) / 4;
        n.w = (w00 + w01 + w10 + w11) / 4;
        output.ptr (y        )[x] = n.x;
        output.ptr (y + drows)[x] = n.y;
        output.ptr (y + 2 * drows)[x] = n.z;
        output.ptr (y + 3 * drows)[x] = n.w;
    }
}


void resizeCMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    int in_cols = input.cols();
    int in_rows = input.rows() / 4;

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    output.create(out_rows * 4, out_cols);

    dim3 block(32, 8);
    dim3 grid(getGridDim(out_cols, block.x), getGridDim (out_rows, block.y));
    resizeCMapKernel<<<grid, block>>>(out_rows, out_cols, in_rows, input, output);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void resizeicpWeightMapKernel(int drows, int dcols, int srows,
                                         const PtrStep<float> input, PtrStep<float> output)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= dcols || y >= drows)
        return;

    const float qnan = __int_as_float(0x7fffffff);

    int xs = x * 2;
    int ys = y * 2;

    float x00 = input.ptr (ys + 0)[xs + 0];
    float x01 = input.ptr (ys + 0)[xs + 1];
    float x10 = input.ptr (ys + 1)[xs + 0];
    float x11 = input.ptr (ys + 1)[xs + 1];

//    float x00 = 1.0;
//    float x01 = 1.0;
//    float x10 = 1.0;
//    float x11 = 1.0;

    if (isnan (x00) || isnan (x01) || isnan (x10) || isnan (x11))
    {
        output.ptr (y)[x] = qnan;
        return;
    }else
    {
        output.ptr (y)[x] = (x00 + x01 + x10 + x11) / 4;
    }
}

void resizeicpWeightMap(const DeviceArray2D<float>& input, DeviceArray2D<float>& output)
{
    int in_cols = input.cols();
    int in_rows = input.rows();

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    output.create(out_rows, out_cols);
    dim3 block(32, 8);
    dim3 grid(getGridDim(out_cols, block.x), getGridDim (out_rows, block.y));

    resizeicpWeightMapKernel<<<grid, block>>>(out_rows, out_cols, in_rows, input, output);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}



__global__ void resizePlaneMapKernel(int drows, int dcols, int srows,
                     const PtrStep<unsigned short> input, PtrStep<unsigned short> output)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if(x >= dcols || y >= drows)
    {
        return;
    }

    int xs = x * 2;
    int ys = y * 2;

    unsigned short p00 = input.ptr (ys + 0)[xs + 0];
    unsigned short p01 = input.ptr (ys + 0)[xs + 1];
    unsigned short p10 = input.ptr (ys + 1)[xs + 0];
    unsigned short p11 = input.ptr (ys + 1)[xs + 1];

    if(p00 == p01 && p00 == p10 && p00 == p11)
    {
        output.ptr(y)[x] = p00;
    }else{
        //while 0 represents it belongs to no plane
        output.ptr(y)[x] = 0U;
    }

}

void resizePlaneMap(const DeviceArray2D<unsigned short>& input, DeviceArray2D<unsigned short>& output)
{
    //just a plane ID is saved
    int in_cols = input.cols();
    int in_rows = input.rows();

    int out_cols = in_cols / 2;
    int out_rows = in_rows / 2;

    output.create(out_rows, out_cols);
    dim3 block(32, 8);
    dim3 grid(getGridDim(out_cols, block.x), getGridDim (out_rows, block.y));
    resizePlaneMapKernel<<<grid, block>>>(out_rows, out_cols, in_rows, input, output);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}


void pyrDownGaussF(const DeviceArray2D<float>& src, DeviceArray2D<float> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    const float gaussKernel[25] = {1, 4, 6, 4, 1,
                                   4, 16, 24, 16, 4,
                                   6, 24, 36, 24, 6,
                                   4, 16, 24, 16, 4,
                                   1, 4, 6, 4, 1};

    float * gauss_cuda;

    cudaMalloc((void**) &gauss_cuda, sizeof(float) * 25);
    cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);

    pyrDownKernelGaussF<<<grid, block>>>(src, dst, gauss_cuda);
    cudaSafeCall ( cudaGetLastError () );

    cudaFree(gauss_cuda);
};

__global__ void pyrDownKernelIntensityGauss(const PtrStepSz<unsigned char> src, PtrStepSz<unsigned char> dst, float * gaussKernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    const int D = 5;

    int center = src.ptr (2 * y)[2 * x];

    int tx = min (2 * x - D / 2 + D, src.cols - 1);
    int ty = min (2 * y - D / 2 + D, src.rows - 1);
    int cy = max (0, 2 * y - D / 2);

    float sum = 0;
    int count = 0;

    for (; cy < ty; ++cy)
        for (int cx = max (0, 2 * x - D / 2); cx < tx; ++cx)
        {
            //This might not be right, but it stops incomplete model images from making up colors
            if(src.ptr (cy)[cx] > 0)
            {
                sum += src.ptr (cy)[cx] * gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
                count += gaussKernel[(ty - cy - 1) * 5 + (tx - cx - 1)];
            }
        }
    dst.ptr (y)[x] = (sum / (float)count);
}

void pyrDownUcharGauss(const DeviceArray2D<unsigned char>& src, DeviceArray2D<unsigned char> & dst)
{
    dst.create (src.rows () / 2, src.cols () / 2);

    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    const float gaussKernel[25] = {1, 4, 6, 4, 1,
                                   4, 16, 24, 16, 4,
                                   6, 24, 36, 24, 6,
                                   4, 16, 24, 16, 4,
                                   1, 4, 6, 4, 1};

    float * gauss_cuda;

    cudaMalloc((void**) &gauss_cuda, sizeof(float) * 25);
    cudaMemcpy(gauss_cuda, &gaussKernel[0], sizeof(float) * 25, cudaMemcpyHostToDevice);

    pyrDownKernelIntensityGauss<<<grid, block>>>(src, dst, gauss_cuda);
    cudaSafeCall ( cudaGetLastError () );

    cudaFree(gauss_cuda);
};

__global__ void verticesToDepthKernel(const float * vmap_src, PtrStepSz<float> dst, float cutOff)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    float z = vmap_src[y * dst.cols * 4 + (x * 4) + 2];

    dst.ptr(y)[x] = z > cutOff || z <= 0 ? __int_as_float(0x7fffffff)/*CUDART_NAN_F*/ : z;
}

void verticesToDepth(DeviceArray<float>& vmap_src, DeviceArray2D<float> & dst, float cutOff)
{
    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows (), block.y));

    verticesToDepthKernel<<<grid, block>>>(vmap_src, dst, cutOff);
    cudaSafeCall ( cudaGetLastError () );
};

texture<uchar4, 2, cudaReadModeElementType> inTex;

__global__ void bgr2IntensityKernel(PtrStepSz<unsigned char> dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dst.cols || y >= dst.rows)
        return;

    uchar4 src = tex2D(inTex, x, y);

    int value = (float)src.x * 0.114f + (float)src.y * 0.299f + (float)src.z * 0.587f;

    dst.ptr (y)[x] = value;
}

void imageBGRToIntensity(cudaArray * cuArr, DeviceArray2D<unsigned char> & dst)
{
    dim3 block (32, 8);
    dim3 grid (getGridDim (dst.cols (), block.x), getGridDim (dst.rows(), block.y));

    cudaSafeCall(cudaBindTextureToArray(inTex, cuArr));

    bgr2IntensityKernel<<<grid, block>>>(dst);

    cudaSafeCall(cudaGetLastError());

    cudaSafeCall(cudaUnbindTexture(inTex));
};

__constant__ float gsobel_x3x3[9];
__constant__ float gsobel_y3x3[9];

__global__ void applyKernel(const PtrStepSz<unsigned char> src, PtrStep<short> dx, PtrStep<short> dy)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;

  if(x >= src.cols || y >= src.rows)
    return;

  float dxVal = 0;
  float dyVal = 0;

  int kernelIndex = 8;
  for(int j = max(y - 1, 0); j <= min(y + 1, src.rows - 1); j++)
  {
      for(int i = max(x - 1, 0); i <= min(x + 1, src.cols - 1); i++)
      {
          dxVal += (float)src.ptr(j)[i] * gsobel_x3x3[kernelIndex];
          dyVal += (float)src.ptr(j)[i] * gsobel_y3x3[kernelIndex];
          --kernelIndex;
      }
  }

  dx.ptr(y)[x] = dxVal;
  dy.ptr(y)[x] = dyVal;
}

void computeDerivativeImages(DeviceArray2D<unsigned char>& src, DeviceArray2D<short>& dx, DeviceArray2D<short>& dy)
{
    static bool once = false;

    if(!once)
    {
//         float gsx3x3[9] = {0.52201,  0.00000, -0.52201,
//                            0.79451, -0.00000, -0.79451,
//                            0.52201,  0.00000, -0.52201};

//         float gsy3x3[9] = {0.52201, 0.79451, 0.52201,
//                            0.00000, 0.00000, 0.00000,
//                            -0.52201, -0.79451, -0.52201};

        float gsx3x3[9] = {1,  0.00000, -1,
                           2, -0.00000, -2,
                           1,  0.00000, -1};
 
         float gsy3x3[9] = {1, 2, 1,
                            0.00000, 0.00000, 0.00000,
                            -1, -2, -1};

        cudaMemcpyToSymbol(gsobel_x3x3, gsx3x3, sizeof(float) * 9);
        cudaMemcpyToSymbol(gsobel_y3x3, gsy3x3, sizeof(float) * 9);

        cudaSafeCall(cudaGetLastError());
        cudaSafeCall(cudaDeviceSynchronize());
        once = true;
    }

    dim3 block(32, 8);
    dim3 grid(getGridDim (src.cols (), block.x), getGridDim (src.rows (), block.y));

    applyKernel<<<grid, block>>>(src, dx, dy);

    cudaSafeCall(cudaGetLastError());
    cudaSafeCall(cudaDeviceSynchronize());
}

__global__ void projectPointsKernel(const PtrStepSz<float> depth,
                                    PtrStepSz<float3> cloud,
                                    const float invFx,
                                    const float invFy,
                                    const float cx,
                                    const float cy)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= depth.cols || y >= depth.rows)
        return;

    float z = depth.ptr(y)[x];

    cloud.ptr(y)[x].x = (float)((x - cx) * z * invFx);
    cloud.ptr(y)[x].y = (float)((y - cy) * z * invFy);
    cloud.ptr(y)[x].z = z;
}

void projectToPointCloud(const DeviceArray2D<float> & depth,
                         const DeviceArray2D<float3> & cloud,
                         CameraModel & intrinsics,
                         const int & level)
{
    dim3 block (32, 8);
    dim3 grid (getGridDim (depth.cols (), block.x), getGridDim (depth.rows (), block.y));

    CameraModel intrinsicsLevel = intrinsics(level);

    projectPointsKernel<<<grid, block>>>(depth, cloud, 1.0f / intrinsicsLevel.fx, 1.0f / intrinsicsLevel.fy, intrinsicsLevel.cx, intrinsicsLevel.cy);
    cudaSafeCall ( cudaGetLastError () );
    cudaSafeCall (cudaDeviceSynchronize ());
}

__global__ void updateLambdaMapKernel(int rows, int cols, const PtrStep<float> vmap_curr, const PtrStep<float> vmap_g_prev, const PtrStepSz<int2> Correp_icp,
                                      const PtrStepSz<float3> z_thrinkMap, PtrStepSz<float3> lambdaMap,
                                      const mat33 Rcurr,  const float3 tcurr, const mat33 Rprev_inv, const float3 tprev)
{
    //data index
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

//    if(x < cols && y < rows)
//    {
    if (x >= cols || y >= rows)
        return;

        float mu = 10.0f;

        float3 v_curr = make_float3(0.0f, 0.0f, 0.0f);
        float3 v_prev = make_float3(0.0f, 0.0f, 0.0f);
        float3 vcurr_g = make_float3(0.0f, 0.0f, 0.0f);
        int2 corresp = Correp_icp.ptr(y)[x];  //please initilize it immediately

        // if find correspondence then update lambda, otherwise set lambda to 0;
        //if(found_coresp)
        float3 z = z_thrinkMap.ptr(y)[x];
        if(corresp.x > 0)
        {
           v_curr.x = vmap_curr.ptr(y       )[x];
           v_curr.y = vmap_curr.ptr(y + rows)[x];
           v_curr.z = vmap_curr.ptr(y + 2 * rows)[x];

           vcurr_g = Rcurr * v_curr + tcurr;                   //transfor to global coodinate
           float3 vcurr_last_pose = Rprev_inv * (vcurr_g - tprev);    //transform to the last pose frame

           v_prev.x = vmap_g_prev.ptr(corresp.y       )[corresp.x];
           v_prev.y = vmap_g_prev.ptr(corresp.y + rows)[corresp.x];
           v_prev.z = vmap_g_prev.ptr(corresp.y + 2 * rows)[corresp.x];

           v_prev = Rprev_inv * (v_prev - tprev);                    //target point as reference

           float3 Delta = vcurr_last_pose - v_prev - z;
           float3 lamda_update = lambdaMap.ptr(y)[x] + mu * Delta;
           lambdaMap.ptr(y)[x].x = lamda_update.x;                  //Delta.x;
           lambdaMap.ptr(y)[x].y = lamda_update.y;
           lambdaMap.ptr(y)[x].z = lamda_update.z;
//          lambdaMap.ptr(y)[x].x = v_curr.x;
//          lambdaMap.ptr(y)[x].y = 2.0f;
//          lambdaMap.ptr(y)[x].z = 2.0f;
         }
    //}

    //lamSum();
}

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
                     int blocks)
{
    int cols = vmap_curr.cols ();
    int rows = vmap_curr.rows () / 4;

    dim3 block(32, 8);
    dim3 grid(1, 1, 1);
    grid.x = getGridDim(cols, block.x);
    grid.y = getGridDim(rows, block.y);

    updateLambdaMapKernel<<<grid, block>>>(rows, cols, vmap_curr, vmap_g_prev, corresICP, z_thrinkMap, lambdaMap,
                                               Rcurr, tcurr, Rprev_inv, tprev);
    cudaSafeCall(cudaGetLastError());
    cudaSafeCall (cudaDeviceSynchronize());
}


void updateGlobalModel()
{


}
