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

#include <vector_functions.h>

#ifndef CUDA_OPERATORS_CUH_
#define CUDA_OPERATORS_CUH_

__device__ __host__ __forceinline__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ __forceinline__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ __forceinline__ float3 cross(const float3& a, const float3& b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ __host__ __forceinline__ float dot(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ __forceinline__ float norm(const float3& a)
{
    return sqrtf(dot(a, a));
}

__device__ __host__ __forceinline__ float3 normalized(const float3& a)
{
    const float rn = rsqrtf(dot(a, a));
    return make_float3(a.x * rn, a.y * rn, a.z * rn);
}

__device__ __forceinline__ float3 operator*(const mat33& m, const float3& a)
{
    return make_float3(dot(m.data[0], a), dot(m.data[1], a), dot(m.data[2], a));
}

__device__ __forceinline__ float3 operator*(const float m, const float3& a)
{
    return make_float3(m * a.x, m * a.y, m * a.z);
}

__device__ __forceinline__ float3 operator/(const float3& a, const float m)
{
    return make_float3(a.x / m, a.y / m, a.z / m);
}

__device__ __forceinline__ void ppp(float a[][3], float e[], float s[], float v[][3], int m, int n)
{
    int i,j;
    float d;

    if (m>=n) i=n;
    else i=m;
    for (j=1; j<=i-1; j++)
      { a[(j-1)][j-1]=s[j-1];
        a[(j-1)][j]=e[j-1];
      }
    a[(i-1)][i-1]=s[i-1];
    if (m<n) a[(i-1)][i]=e[i-1];
    for (i=1; i<=n-1; i++)
    for (j=i+1; j<=n; j++)
      {
        d=v[i-1][j-1]; v[i-1][j-1]=v[j-1][i-1]; v[j-1][i-1]=d;
      }
}

__device__ __forceinline__ void transpose(float A[][3], float B[][3])
{
    B[0][0] = A[0][0];   B[0][1] = A[1][0];   B[0][2] = A[2][0];
    B[1][0] = A[0][1];   B[1][1] = A[1][1];   B[1][2] = A[2][1];
    B[2][0] = A[0][2];   B[2][1] = A[1][2];   B[2][2] = A[2][2];
}

__device__ __forceinline__ void _multiplyM2M(float A[][3], float B[][3], float C[][3])
{
    int i, j, l;
    for(i = 0; i < 3; i++)
    {
       for(j = 0; j < 3; j++)
       {
          C[i][j] = 0.0;
          for(l = 0; l < 3; l++) C[i][j] = C[i][j] + A[i][l] * B[l][j];
       }
    }
}

__device__ __forceinline__ void _multiplyM2V(float A[][3], float x[], float b[])
{
    b[0] = A[0][0] * x[0] + A[0][1] * x[1] + A[0][2] * x[2];
    b[1] = A[1][0] * x[0] + A[1][1] * x[1] + A[1][2] * x[2];
    b[2] = A[2][0] * x[0] + A[2][1] * x[1] + A[2][2] * x[2];
}

__device__ __forceinline__ void _substractM(float A[][3], float B[][3], float C[][3])
{
    C[0][0] = A[0][0] - B[0][0];
    C[0][1] = A[0][1] - B[0][1];
    C[0][2] = A[0][2] - B[0][2];

    C[1][0] = A[1][0] - B[1][0];
    C[1][1] = A[1][1] - B[1][1];
    C[1][2] = A[1][2] - B[1][2];

    C[2][0] = A[2][0] - B[2][0];
    C[2][1] = A[2][1] - B[2][1];
    C[2][2] = A[2][2] - B[2][2];
}

__device__ __forceinline__ void sss(float fg[], float cs[])
{
        float r,d;
    if ((fabs(fg[0])+fabs(fg[1]))==0.0)
      {cs[0]=1.0; cs[1]=0.0; d=0.0;}
    else
      { d=sqrt(fg[0]*fg[0]+fg[1]*fg[1]);
        if (fabs(fg[0])>fabs(fg[1]))
          { d=fabs(d);
            if (fg[0]<0.0) d=-d;
          }
        if (fabs(fg[1])>=fabs(fg[0]))
          { d=fabs(d);
            if (fg[1]<0.0) d=-d;
          }
        cs[0]=fg[0]/d; cs[1]=fg[1]/d;
      }
    r=1.0;
    if (fabs(fg[0])>fabs(fg[1])) r=cs[1];
    else
      if (cs[0]!=0.0) r=1.0/cs[0];
    fg[0]=d; fg[1]=r;
}

__device__ __forceinline__ bool _singularValueDecomposition(float a[][3], float u[][3], float v[][3]){

    float eps=0.00001;
    int i,j,k,l,it,ll,kk,mm,nn,m1,ks;
    float d,dd,t,sm,sm1,em1,sk,ek,b,c,shh,fg[2],cs[2];
    float s[5],e[5],w[5];
    const int m=3,n=3;

    //ka=((m>n)?m:n)+1;

    for(i=0;i<3;i++) {
            for(j=0;j<3;j++){
               u[i][j]=v[i][j]=0.0;
            }
    }

    it=60; k=n;
    if (m-1<n) k=m-1;
    l=m;
    if (n-2<m) l=n-2;
    if (l<0) l=0;
    ll=k;
    if (l>k) ll=l;
    if (ll>=1)
    { for (kk=1; kk<=ll; kk++)
      { if (kk<=k)
          { d=0.0;
            for (i=kk; i<=m; i++)
              { d=d+a[i-1][kk-1]*a[i-1][kk-1];}
            s[kk-1]=sqrt(d);
            if (s[kk-1]!=0.0)
              {
                if (a[kk-1][kk-1]!=0.0)
                  { s[kk-1]=fabs(s[kk-1]);
                    if (a[kk-1][kk-1]<0.0) s[kk-1]=-s[kk-1];
                  }
                for (i=kk; i<=m; i++)
                  {
                    a[i-1][kk-1]=a[i-1][kk-1]/s[kk-1];
                  }
                a[kk-1][kk-1]=1.0+a[kk-1][kk-1];
              }
            s[kk-1]=-s[kk-1];
          }
        if (n>=kk+1)
          { for (j=kk+1; j<=n; j++)
              { if ((kk<=k)&&(s[kk-1]!=0.0))
                  { d=0.0;
                    for (i=kk; i<=m; i++)
                      {
                        d=d+a[i-1][kk-1]*a[i-1][j-1];
                      }
                    d=-d/a[kk-1][kk-1];
                    for (i=kk; i<=m; i++)
                      {
                        a[i-1][j-1]=a[i-1][j-1]+d*a[i-1][kk-1];
                      }
                  }
                e[j-1]=a[kk-1][j-1];
              }
          }
        if (kk<=k)
          { for (i=kk; i<=m; i++)
              {
                u[i-1][kk-1]=a[i-1][kk-1];
              }
          }
        if (kk<=l)
          { d=0.0;
            for (i=kk+1; i<=n; i++)
              d=d+e[i-1]*e[i-1];
            e[kk-1]=sqrt(d);
            if (e[kk-1]!=0.0)
              { if (e[kk]!=0.0)
                  { e[kk-1]=fabs(e[kk-1]);
                    if (e[kk]<0.0) e[kk-1]=-e[kk-1];
                  }
                for (i=kk+1; i<=n; i++)
                  e[i-1]=e[i-1]/e[kk-1];
                e[kk]=1.0+e[kk];
              }
            e[kk-1]=-e[kk-1];
            if ((kk+1<=m)&&(e[kk-1]!=0.0))
              { for (i=kk+1; i<=m; i++) w[i-1]=0.0;
                for (j=kk+1; j<=n; j++)
                  for (i=kk+1; i<=m; i++)
                    w[i-1]=w[i-1]+e[j-1]*a[i-1][j-1];
                for (j=kk+1; j<=n; j++)
                  for (i=kk+1; i<=m; i++)
                    {
                      a[i-1][j-1]=a[i-1][j-1]-w[i-1]*e[j-1]/e[kk];
                    }
              }
            for (i=kk+1; i<=n; i++)
              v[i-1][kk-1]=e[i-1];
          }
      }
    }
    mm=n;
    if (m+1<n) mm=m+1;
    if (k<n) s[k]=a[k][k];
    if (m<mm) s[mm-1]=0.0;
    if (l+1<mm) e[l]=a[l][mm-1];
    e[mm-1]=0.0;
    nn=m;
    if (m>n) nn=n;
    if (nn>=k+1)
    { for (j=k+1; j<=nn; j++)
      { for (i=1; i<=m; i++)
          u[i-1][j-1]=0.0;
        u[j-1][j-1]=1.0;
      }
    }
    if (k>=1)
    { for (ll=1; ll<=k; ll++)
      { kk=k-ll+1;
        if (s[kk-1]!=0.0)
          { if (nn>=kk+1)
              for (j=kk+1; j<=nn; j++)
                { d=0.0;
                  for (i=kk; i<=m; i++)
                    {
                      d=d+u[i-1][kk-1]*u[i-1][j-1]/u[kk-1][kk-1];
                    }
                  d=-d;
                  for (i=kk; i<=m; i++)
                    {
                      u[i-1][j-1]=u[i-1][j-1]+d*u[i-1][kk-1];
                    }
                }
              for (i=kk; i<=m; i++)
                { u[i-1][kk-1]=-u[i-1][kk-1];}
              u[kk-1][kk-1]=1.0+u[kk-1][kk-1];
              if (kk-1>=1)
                for (i=1; i<=kk-1; i++)
                  u[i-1][kk-1]=0.0;
          }
        else
          { for (i=1; i<=m; i++)
              u[i-1][kk-1]=0.0;
            u[kk-1][kk-1]=1.0;
          }
      }
    }
    for (ll=1; ll<=n; ll++)
    { kk=n-ll+1;
    if ((kk<=l)&&(e[kk-1]!=0.0))
      { for (j=kk+1; j<=n; j++)
          { d=0.0;
            for (i=kk+1; i<=n; i++)
              {
                d=d+v[i-1][kk-1]*v[i-1][j-1]/v[kk][kk-1];
              }
            d=-d;
            for (i=kk+1; i<=n; i++)
              {
                v[i-1][j-1]=v[i-1][j-1]+d*v[i-1][kk-1];
              }
          }
      }
    for (i=1; i<=n; i++)
      v[i-1][kk-1]=0.0;
    v[kk-1][kk-1]=1.0;
    }
    for (i=1; i<=m; i++)
    for (j=1; j<=n; j++)
    a[i-1][j-1]=0.0;
    m1=mm; it=60;
    while (1==1)
    { if (mm==0)
      { ppp(a,e,s,v,m,n); return true;
      }
    if (it==0)
      { ppp(a,e,s,v,m,n); return false;
      }
    kk=mm-1;
    while ((kk!=0)&&(fabs(e[kk-1])!=0.0))
      { d=fabs(s[kk-1])+fabs(s[kk]);
        dd=fabs(e[kk-1]);
        if (dd>eps*d) kk=kk-1;
        else e[kk-1]=0.0;
      }
    if (kk==mm-1)
      { kk=kk+1;
        if (s[kk-1]<0.0)
          { s[kk-1]=-s[kk-1];
            for (i=1; i<=n; i++)
              { v[i-1][kk-1]=-v[i-1][kk-1];}
          }
        while ((kk!=m1)&&(s[kk-1]<s[kk]))
          { d=s[kk-1]; s[kk-1]=s[kk]; s[kk]=d;
            if (kk<n)
              for (i=1; i<=n; i++)
                {
                  d=v[i-1][kk-1]; v[i-1][kk-1]=v[i-1][kk]; v[i-1][kk]=d;
                }
            if (kk<m)
              for (i=1; i<=m; i++)
                {
                  d=u[i-1][kk-1]; u[i-1][kk-1]=u[i-1][kk]; u[i-1][kk]=d;
                }
            kk=kk+1;
          }
        it=60;
        mm=mm-1;
      }
    else
      { ks=mm;
        while ((ks>kk)&&(fabs(s[ks-1])!=0.0))
          { d=0.0;
            if (ks!=mm) d=d+fabs(e[ks-1]);
            if (ks!=kk+1) d=d+fabs(e[ks-2]);
            dd=fabs(s[ks-1]);
            if (dd>eps*d) ks=ks-1;
            else s[ks-1]=0.0;
          }
        if (ks==kk)
          { kk=kk+1;
            d=fabs(s[mm-1]);
            t=fabs(s[mm-2]);
            if (t>d) d=t;
            t=fabs(e[mm-2]);
            if (t>d) d=t;
            t=fabs(s[kk-1]);
            if (t>d) d=t;
            t=fabs(e[kk-1]);
            if (t>d) d=t;
            sm=s[mm-1]/d; sm1=s[mm-2]/d;
            em1=e[mm-2]/d;
            sk=s[kk-1]/d; ek=e[kk-1]/d;
            b=((sm1+sm)*(sm1-sm)+em1*em1)/2.0;
            c=sm*em1; c=c*c; shh=0.0;
            if ((b!=0.0)||(c!=0.0))
              { shh=sqrt(b*b+c);
                if (b<0.0) shh=-shh;
                shh=c/(b+shh);
              }
            fg[0]=(sk+sm)*(sk-sm)-shh;
            fg[1]=sk*ek;
            for (i=kk; i<=mm-1; i++)
              { sss(fg,cs);
                if (i!=kk) e[i-2]=fg[0];
                fg[0]=cs[0]*s[i-1]+cs[1]*e[i-1];
                e[i-1]=cs[0]*e[i-1]-cs[1]*s[i-1];
                fg[1]=cs[1]*s[i];
                s[i]=cs[0]*s[i];
                if ((cs[0]!=1.0)||(cs[1]!=0.0))
                  for (j=1; j<=n; j++)
                    {
                      d=cs[0]*v[j-1][i-1]+cs[1]*v[j-1][i];
                      v[j-1][i]=-cs[1]*v[j-1][i-1]+cs[0]*v[j-1][i];
                      v[j-1][i-1]=d;
                    }
                sss(fg,cs);
                s[i-1]=fg[0];
                fg[0]=cs[0]*e[i-1]+cs[1]*s[i];
                s[i]=-cs[1]*e[i-1]+cs[0]*s[i];
                fg[1]=cs[1]*e[i];
                e[i]=cs[0]*e[i];
                if (i<m)
                  if ((cs[0]!=1.0)||(cs[1]!=0.0))
                    for (j=1; j<=m; j++)
                      {
                        d=cs[0]*u[j-1][i-1]+cs[1]*u[j-1][i];
                        u[j-1][i]=-cs[1]*u[j-1][i-1]+cs[0]*u[j-1][i];
                        u[j-1][i-1]=d;
                      }
              }
            e[mm-2]=fg[0];
            it=it-1;
          }
        else
          { if (ks==mm)
              { kk=kk+1;
                fg[1]=e[mm-2]; e[mm-2]=0.0;
                for (ll=kk; ll<=mm-1; ll++)
                  { i=mm+kk-ll-1;
                    fg[0]=s[i-1];
                    sss(fg,cs);
                    s[i-1]=fg[0];
                    if (i!=kk)
                      { fg[1]=-cs[1]*e[i-2];
                        e[i-2]=cs[0]*e[i-2];
                      }
                    if ((cs[0]!=1.0)||(cs[1]!=0.0))
                      for (j=1; j<=n; j++)
                        {
                          d=cs[0]*v[j-1][i-1]+cs[1]*v[j-1][mm-1];
                          v[j-1][mm-1]=-cs[1]*v[j-1][i-1]+cs[0]*v[j-1][mm-1];
                          v[j-1][i-1]=d;
                        }
                  }
              }
            else
              { kk=ks+1;
                fg[1]=e[kk-2];
                e[kk-2]=0.0;
                for (i=kk; i<=mm; i++)
                  { fg[0]=s[i-1];
                    sss(fg,cs);
                    s[i-1]=fg[0];
                    fg[1]=-cs[1]*e[i-1];
                    e[i-1]=cs[0]*e[i-1];
                    if((cs[0]!=1.0)||(cs[1]!=0.0))
                      for (j=1; j<=m; j++)
                        {
                          d=cs[0]*u[j-1][i-1]+cs[1]*u[j-1][kk-2];
                          u[j-1][kk-2]=-cs[1]*u[j-1][i-1]+cs[0]*u[j-1][kk-2];
                          u[j-1][i-1]=d;
                        }
                  }
              }
          }
      }
    }
    return true;
}

#endif /* CUDA_OPERATORS_CUH_ */
