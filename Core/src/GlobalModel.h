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

#ifndef GLOBALMODEL_H_
#define GLOBALMODEL_H_

#include "Shaders/Shaders.h"
#include "Shaders/Uniform.h"
#include "Shaders/FeedbackBuffer.h"
#include "GPUTexture.h"
#include "Utils/Resolution.h"
#include "IndexMap.h"
#include "Utils/Stopwatch.h"
#include "Utils/Intrinsics.h"
#include "Utils/GlobalStateParams.h"
#include "SubMap.h"
#include <pangolin/gl/gl.h>
#include <Eigen/LU>
#include <time.h>
#include <string>
#include <sstream>

#include "Defines.h"
#include "../Cuda/cudafuncs.cuh"
#include <vector_types.h>
#include "Cuda/cudafuncs.cuh"
#include <Eigen/StdVector>

class GlobalModel
{
    public:
        GlobalModel();
        virtual ~GlobalModel();

        void initialise( GPUTexture * vertexMap,
                         GPUTexture * normalMap,
                         GPUTexture * colorMap,
                         GPUTexture * curv1Map,
                         GPUTexture * curv2Map,
                         GPUTexture * gradientMagMap,
                         const Eigen::Matrix4f& init_pose);

        static const int TEXTURE_DIMENSION;
        static const int MAX_VERTICES;
        static const int NODE_TEXTURE_DIMENSION;
        static const int MAX_NODES;
        static const int MAX_SUBMAPS;
        static const int DELTA_TRANS_DIMENSION;
        static const int ACTIVE_KEYFRAME_DIMENSION;

        EFUSION_API void renderPointCloud(pangolin::OpenGlMatrix mvp,
                              const float threshold,
                              const bool drawUnstable,
                              const bool drawNormals,
                              const bool drawColors,
                              const bool drawPoints,
                              const bool drawWindow,
                              const bool drawTimes,
                              const int time,
                              const int timeDelta);

        EFUSION_API const std::pair<GLuint, GLuint> & model();

        void fuse(const Eigen::Matrix4f & pose,
                  const int & time,
                  GPUTexture * rgb,
                  GPUTexture * depthRaw,
                  GPUTexture * depthFiltered,
                  GPUTexture * curv_map_max,
                  GPUTexture * curv_map_min,
                  GPUTexture * confidence,
                  GPUTexture * indexMap,
                  GPUTexture * vertConfMap,
                  GPUTexture * colorTimeMap,
                  GPUTexture * normRadMap,
                  const float depthCutoff,
                  const float confThreshold,
                  const float weighting,
                  const bool insertSubmap,
                  const float indexsubmap);

        void clean(const Eigen::Matrix4f & pose,
                   const int & time,
                   GPUTexture * indexMap,
                   GPUTexture * vertConfMap,
                   GPUTexture * colorTimeMap,
                   GPUTexture * normRadMap,
                   GPUTexture * depthMap,
                   const float confThreshold,
                   const float maxDepth);

        void updateModel();


        EFUSION_API unsigned int lastCount();

        Eigen::Vector4f * downloadMap();

        std::mutex UpdateGobalModel;

        std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> DeltaTransformKF;

        std::vector<int>lActiveKFID;

    private:
        //First is the vbo, second is the fid, Vertex Buffer Object(VBO) and FeedBack ID
        std::pair<GLuint, GLuint> * vbos;
        int target, renderSource; //target for computation renderSource for index buffer rendering

        std::vector<SubMap*> SubMaps;

        //this vector for the global model transformation
        std::vector<Eigen::Matrix4f> vposeSubMaps;

        const int bufferSize;

        GLuint countQuery;
        unsigned int count;

        std::shared_ptr<Shader> initTProgram;

        std::shared_ptr<Shader> drawProgram;
        std::shared_ptr<Shader> drawSurfelProgram;

        //For supersample fusing
        std::shared_ptr<Shader> dataProgram;
        std::shared_ptr<Shader> updateProgram;
        std::shared_ptr<Shader> unstableProgram;

        std::shared_ptr<Shader> updateDeltaTransProgram;
        //render gobal model in a index map(offscreen operation)
        pangolin::GlRenderBuffer renderBuffer;

        //We render updated vertices vec3 + confidences to one texture
        GPUTexture updateMapVertsConfs;

        //We render updated colors vec3 + timestamps to another
        GPUTexture updateMapColorsTime;

        //We render updated normals vec3 + radii to another
        GPUTexture updateMapNormsRadii;

        //We render update curvature k1 vector + value to another
        GPUTexture updateMapCurvatureK1;

        //We render update curvature k2 vector + value to another
        GPUTexture updateMapCurvatureK2;

        //Matrix vector to update the current 3D model
        GPUTexture DeltaTransVec;

        //KeyFrameIDMap
        GPUTexture KeyFrameIDMap;

        GLuint  newUnstableVbo, newUnstableFid;

        pangolin::GlFramebuffer frameBuffer;
        GLuint uvo;    //uv object, coordinate in pixel
        int uvSize;
};

#endif /* GLOBALMODEL_H_ */
