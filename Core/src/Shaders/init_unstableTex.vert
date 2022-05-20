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

layout (location = 0) in vec2 texcoord;

out vec4 vPosition;
out vec4 vColor;
out vec4 vNormRad;
out vec4 curv_map_max;
out vec4 curv_map_min;

flat out int valid;

uniform sampler2D vertexSampler;
uniform sampler2D normalSampler;
uniform sampler2D colorSampler;
uniform sampler2D curv1Sampler;
uniform sampler2D curv2Sampler;
uniform sampler2D gradientMagSampler;

uniform float cols;
uniform float rows;
uniform vec4 cam;
uniform float curvature_valid_threshold;

uniform float useConfidenceEvaluation;
uniform float epsilon;

uniform mat4 init_pose;

#include "color.glsl"
#include "surfels.glsl"

void main()
{
    float x = texcoord.x * cols;
    float y = texcoord.y * rows;

    vec4 vPositionLoc = textureLod(vertexSampler, texcoord, 0.0);
    vec4 vPositionGlob = init_pose * vec4(vPositionLoc.xyz, 1.0);
    vPosition = vec4(vPositionGlob.xyz, vPositionLoc.w);
    float gradient_mag = float(textureLod(gradientMagSampler, texcoord, 0.0));

    float max_dist = sqrt((rows * 0.5)*(rows * 0.5) + (cols * 0.5)*(cols * 0.5));
    if(useConfidenceEvaluation > 0.0)
         vPosition.w = confidence(x, y, max_dist, 1.0) * exp(-epsilon/sqrt(gradient_mag)); 
    else
         vPosition.w = confidence(x, y, max_dist, 1.0);

    vec4 vNormRadLoc = textureLod(normalSampler, texcoord, 0.0);
    vNormRad =vec4(mat3(init_pose) * vNormRadLoc.xyz, vNormRadLoc.w);
    vColor = textureLod(colorSampler, texcoord, 0.0);
    curv_map_max = textureLod(curv1Sampler, texcoord, 0.0);
    curv_map_min = textureLod(curv2Sampler, texcoord, 0.0);

    vColor.x = encodeColor(vColor.xyz);
    //used for the submap division
    vColor.y = 0.0;
    //Timestamp
    vColor.w = 1.0;

    if(length(vNormRad.xyz) > 0.5 && 
       curv_map_max.w > - curvature_valid_threshold && curv_map_max.w < curvature_valid_threshold &&
       curv_map_min.w > - curvature_valid_threshold && curv_map_min.w < curvature_valid_threshold)
    {
        valid = 1;
    }else
    {
        valid = 0;
    }

}
