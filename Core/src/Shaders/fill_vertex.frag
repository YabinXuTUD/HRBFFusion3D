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

#version 330 core

in vec2 texcoord;

layout(location = 0) out vec4 FragColor;
layout(location = 1) out float icp_weight;

//vertex map
uniform sampler2D eSampler;
uniform sampler2D filteredSampler;
uniform sampler2D rck1Sampler;
uniform sampler2D rck2Sampler;
uniform sampler2D eicpweightSampler;
uniform sampler2D confidenceSampler;
uniform vec4 cam;         //cx, cy, 1/fx, 1/fy
uniform float cols;
uniform float rows;
uniform int passthrough;
uniform float weight;
uniform int time;
uniform float icp_weight_lambda;
uniform float curvature_valid_threshold;

#include"surfels.glsl"
void main()
{
    float x = texcoord.x * cols;
    float y = texcoord.y * rows;

    vec4 sample = texture(eSampler, texcoord, 0.0);
    float icp_weight_sample = float(texture(eicpweightSampler, texcoord, 0.0));

    if(sample.z == 0 || passthrough == 1)
    {
        vec4 filtered_vertex = texture(filteredSampler, texcoord, 0.0);
        vec4 rck1_sample = texture(rck1Sampler, texcoord, 0.0);
        vec4 rck2_sample = texture(rck2Sampler, texcoord, 0.0);

        if(rck1_sample.w > -curvature_valid_threshold && rck1_sample.w < curvature_valid_threshold &&
           rck2_sample.w > -curvature_valid_threshold && rck2_sample.w < curvature_valid_threshold)
           {
              float lambda = icp_weight_lambda;
              float vConf = float(textureLod(confidenceSampler, texcoord, 0.0));
              float cmax = abs(rck1_sample.w) > abs(rck2_sample.w) ? abs(rck1_sample.w):abs(rck2_sample.w);
              icp_weight = (1.0f / (filtered_vertex.z * filtered_vertex.z)) * (vConf / 256.0f + exp(-0.5 * (lambda * lambda) / (cmax * cmax)));  
              FragColor = vec4(filtered_vertex.xyz, vConf); 
            }
    }
    else
    {
        FragColor = sample;
        icp_weight = icp_weight_sample;
    }
}
