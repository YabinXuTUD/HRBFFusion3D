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

#ifndef INDEXMAP_H_
#define INDEXMAP_H_

#include "Shaders/Shaders.h"
#include "Shaders/Uniform.h"
#include "Shaders/Vertex.h"
#include "GPUTexture.h"
#include "Utils/Resolution.h"
#include "Utils/Intrinsics.h"
#include "Utils/Img.h"
#include "Utils/GlobalStateParams.h"
#include <pangolin/gl/gl.h>
#include <Eigen/LU>

#include <opencv2/opencv.hpp>
#include "Defines.h"

class IndexMap
{
    public:
        IndexMap();
        virtual ~IndexMap();

        //project vertex to the texture of the mdoel/frame. for point based fusion
        void predictIndices(const Eigen::Matrix4f & pose,
                            const int & time,
                            const int maxTime,
                            const std::pair<GLuint, GLuint> & model,
                            const float depthCutoff,
                            const int insertSubmap,
                            const int indexSubmap);

        EFUSION_API void renderDepth(const float depthCutoff);

        void renderHRBFPrediction(pangolin::OpenGlMatrix mvp, const Eigen::Matrix4f& pose);
        void renderSurfelPrediction(pangolin::OpenGlMatrix mvp, const Eigen::Matrix4f& pose);

        enum Prediction
        {
            ACTIVE,
            INACTIVE
        };

        //to determine the initial pose of the vertex
        void combinedPredict(const Eigen::Matrix4f & pose,
                                     const std::pair<GLuint, GLuint> & model,
                                     const float depthCutoff,
                                     const float confThreshold);

        void predictHRBF(IndexMap::Prediction predictionType);

        void synthesizeInfo(const Eigen::Matrix4f & pose,
                            const std::pair<GLuint, GLuint> & model,
                            const float depthCutoff,
                            const float confThreshold);

        void synthesizeDepth(const Eigen::Matrix4f & pose,
                             const std::pair<GLuint, GLuint> & model,
                             const float depthCutoff,
                             const float confThreshold,
                             const int time,
                             const int maxTime,
                             const int timeDelta);

        //save texture for visualization
        void downloadTexture(const Eigen::Matrix4f& pose, int frameID);

        GPUTexture * indexTex()
        {
            return &indexTexture;
        }

        GPUTexture * vertConfTex()
        {
            return &vertConfTexture;
        }

        GPUTexture * colorTimeTex()
        {
            return &colorTimeTexture;
        }

        GPUTexture * normalRadTex()
        {
            return &normalRadTexture;
        }

        GPUTexture * curvMaxTex()
        {
            return &curv_map_maxTexture;
        }

        GPUTexture * curvMinTex()
        {
            return &curv_map_minTexture;
        }

        GPUTexture * drawTex()
        {
            return &drawTexture;
        }

        GPUTexture * depthTex()
        {
            return &depthTexture;
        }

        GPUTexture * imageTexSurfel()
        {
            return &imageSurfelTexture;

        }

        GPUTexture * vertexTexSurfel()
        {
            return &vertexSurfelTexture;

        }

        GPUTexture * normalTexSurfel()
        {
            return &normalSurfelTexture;
        }

        GPUTexture * imageTexHRBF()
        {
            return &imageTextureHRBF;

        }
        GPUTexture * vertexTexHRBF()
        {
            return &vertexTextureHRBF;

        }
        GPUTexture * normalTexHRBF()
        {
            return &normalTextureHRBF;
        }
        GPUTexture * curvk1TexHRBF()
        {
            return &curvk1TextureHRBF;
        }
        GPUTexture * curvk2TexHRBF()
        {
            return &curvk2TextureHRBF;
        }
        GPUTexture * icpweightTexHRBF()
        {
            return &icpWeightHRBF;
        }

        GPUTexture * oldImageTexHRBF()
        {
           return &oldImageTextureHRBF;
        }

        GPUTexture * oldVertexTexHRBF()
        {
           return &oldVertexTextureHRBF;
        }

        GPUTexture * oldNormalTexHRBF()
        {
            return &oldNormalTextureHRBF;
        }

        GPUTexture * oldTimeTexHRBF()
        {
            return &oldTimeTextureHRBF;
        }

        GPUTexture * oldcurvk1TexHRBF()
        {
            return &oldcurvk1TextureHRBF;
        }
        GPUTexture * oldcurvk2TexHRBF()
        {
            return &oldcurvk2TextureHRBF;
        }
        GPUTexture * oldicpweightTexHRBF()
        {
            return &oldicpWeightHRBF;
        }

        static const int FACTOR;
        static const int FACTORModel;

        static const int ACTIVE_KEYFRAME_DIMENSION;

        std::vector<int>lActiveKFID;

    private:
        std::shared_ptr<Shader> indexProgram; //this is index depth program used for fusuion
        pangolin::GlFramebuffer indexFrameBuffer; //the final rendering destination of the OpenGl pipeline is called framebuffer
        pangolin::GlRenderBuffer indexRenderBuffer; //render buffer object is newly introduced for offscreen rendering
        GPUTexture indexTexture; //These is for fusion part, The global model it's self is prject to a texture, not for tracking
        GPUTexture vertConfTexture;
        GPUTexture colorTimeTexture;
        GPUTexture normalRadTexture;
        GPUTexture curv_map_maxTexture;
        GPUTexture curv_map_minTexture;


        //render depth to a framebuffer
        std::shared_ptr<Shader> drawDepthProgram;
        pangolin::GlFramebuffer drawFrameBuffer;
        pangolin::GlRenderBuffer drawRenderBuffer;
        GPUTexture drawTexture;

        std::shared_ptr<Shader> depthProgram;
        pangolin::GlFramebuffer depthFrameBuffer; //for synthesizeDepth usage
        pangolin::GlRenderBuffer depthRenderBuffer;
        GPUTexture depthTexture;

        std::shared_ptr<Shader> surfelSplattingProgram;
        pangolin::GlFramebuffer surfelSplattingFrameBuffer;
        pangolin::GlRenderBuffer surfelSplattingRenderBuffer;
        GPUTexture imageSurfelTexture;
        GPUTexture vertexSurfelTexture;
        GPUTexture normalSurfelTexture;
        GPUTexture timeSurfelTexture;

        std::shared_ptr<Shader> predictHRBFProgram;
        pangolin::GlFramebuffer predictHRBFFrameBuffer;
        pangolin::GlRenderBuffer predictHRBFRenderBuffer;
        GPUTexture imageTextureHRBF;
        GPUTexture vertexTextureHRBF;
        GPUTexture normalTextureHRBF;
        GPUTexture curvk1TextureHRBF;
        GPUTexture curvk2TextureHRBF;
        GPUTexture timeTextureHRBF;
        GPUTexture icpWeightHRBF;

        pangolin::GlFramebuffer oldFrameBufferHRBF;
        pangolin::GlRenderBuffer oldRenderBufferHRBF;
        GPUTexture oldImageTextureHRBF;
        GPUTexture oldVertexTextureHRBF;
        GPUTexture oldNormalTextureHRBF;
        GPUTexture oldTimeTextureHRBF;
        GPUTexture oldcurvk1TextureHRBF;
        GPUTexture oldcurvk2TextureHRBF;
        GPUTexture oldicpWeightHRBF;

        //KeyFrameIDMap
        GPUTexture KeyFrameIDMap;

        std::shared_ptr<Shader> drawPredictedHRBFVertex;
        GLuint vbo_predicted;
};

#endif /* INDEXMAP_H_ */
