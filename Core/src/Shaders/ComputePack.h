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

#ifndef COMPUTEPACK_H_
#define COMPUTEPACK_H_

#include "Shaders.h"
#include "../Utils/Resolution.h"
#include "Uniform.h"
#include <pangolin/gl/gl.h>


//compute packages
class ComputePack
{
    public:
        ComputePack(std::shared_ptr<Shader> program,
                    pangolin::GlTexture * target, pangolin::GlTexture * target1 = NULL, pangolin::GlTexture * target2 = NULL, pangolin::GlTexture * target3 = NULL);

        virtual ~ComputePack();

        static const std::string NORM, FILTER, BILINEAR_FILTER, METRIC, METRIC_FILTERED,
                                 CURVATURE, UPDATE_NORMALRAD, FILTERED_BY_MODEL_POINTS, CONFIDENCE_EVALUATION,
                                 VERTEX_NORMAL_RADIUS, RADIUS_OPTIMIZATION;

        void compute_2input(pangolin::GlTexture * input1, pangolin::GlTexture * input2, const std::vector<Uniform> * const uniforms = 0);
        void compute(pangolin::GlTexture * input, const std::vector<Uniform> * const uniforms = 0);

    private:
        std::shared_ptr<Shader> program;  //for compute on depth map
        pangolin::GlRenderBuffer renderBuffer;
        pangolin::GlTexture * target;
        pangolin::GlTexture * target1;
        pangolin::GlFramebuffer frameBuffer; //
};

#endif /* COMPUTEPACK_H_ */
