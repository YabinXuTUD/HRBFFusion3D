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

#ifndef FILLIN_H_
#define FILLIN_H_

#include "Shaders.h"
#include "Uniform.h"
#include "../Utils/Resolution.h"
#include "../Utils/Intrinsics.h"
#include "../GPUTexture.h"
#include "../Utils/Img.h"
#include "../Utils/GlobalStateParams.h"

#include <opencv2/opencv.hpp>
//#include "Defines.h"

class FillIn
{
    public:
        FillIn();
        virtual ~FillIn();

        void image(GPUTexture * existingRgb, GPUTexture * rawRgb, bool passthrough);
        void vertex(GPUTexture * existingVertex, GPUTexture * vertexFiltered, GPUTexture * eicpWeight,
                    GPUTexture* rawCurv_k1, GPUTexture* rawCurv_k2, GPUTexture* confidenceMap,
                    int time, float weight, bool passthrough);
        void normal(GPUTexture * existingNormal, GPUTexture * rawNormal, bool passthrough);
        void curvature(GPUTexture * existingcurvk1Curvature, GPUTexture * existingcurvk2Curvature ,
                       GPUTexture * curvk1Curvature, GPUTexture * curvk2Curvature, bool passthrough);

        void downloadtexture(const Eigen::Matrix4f& lastPose,int lastFrames, bool global, std::string groundtruthpose);

        GPUTexture imageTexture;
        GPUTexture vertexTexture;
        GPUTexture normalTexture;
        GPUTexture curvk1Texture;
        GPUTexture curvk2Texture;
        GPUTexture icpweightTexture;

        std::shared_ptr<Shader> imageProgram;
        pangolin::GlRenderBuffer imageRenderBuffer;
        pangolin::GlFramebuffer imageFrameBuffer;

        std::shared_ptr<Shader> vertexProgram;
        pangolin::GlRenderBuffer vertexRenderBuffer;
        pangolin::GlFramebuffer vertexFrameBuffer;

        std::shared_ptr<Shader> normalProgram;
        pangolin::GlRenderBuffer normalRenderBuffer;
        pangolin::GlFramebuffer normalFrameBuffer;

        std::shared_ptr<Shader> curvatureProgram;
        pangolin::GlRenderBuffer curvatureRenderBuffer;
        pangolin::GlFramebuffer curvatureFrameBuffer;      

};

#endif /* FILLIN_H_ */

