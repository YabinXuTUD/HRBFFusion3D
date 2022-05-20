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

//copy the newly added one and filtering the oulires and noise

#version 330 core


layout (location = 0) in vec4 vPos;
layout (location = 1) in vec4 vCol;
layout (location = 2) in vec4 vNormR;
layout (location = 3) in vec4 vCurvmax;
layout (location = 4) in vec4 vCurvmin;

out vec4 vPosition;
out vec4 vColor;
out vec4 vNormRad;
out vec4 vCurv_map_max;
out vec4 vCurv_map_min;

flat out int test;

uniform int time;
uniform float scale;
uniform mat4 pose;
uniform mat4 t_inv;
uniform vec4 cam;               //cx, cy, fx, fy
uniform float cols;
uniform float rows;
uniform float confThreshold;
uniform float window_multiplier;
uniform float curvature_valid_threshold;

uniform usampler2D indexSampler;
uniform sampler2D vertConfSampler;
uniform sampler2D colorTimeSampler;
uniform sampler2D normRadSampler;

uniform sampler2D KeyFrameIDMap;
uniform float KeyFrameIDDimen;

uniform float maxDepth;

#include "hrbfbase.glsl"
#include "utils.glsl"

void main()
{
    vPosition = vPos;
    vColor = vCol;
    vNormRad = vNormR;

    test = 1;

    //transfer to the local pose
    vec3 localPos = (t_inv * vec4(vPosition.xyz, 1.0f)).xyz;

    //coordinate abtained in pixels
    float x = ((cam.z * localPos.x) / localPos.z) + cam.x;
    float y = ((cam.w * localPos.y) / localPos.z) + cam.y;

    //transfer to the local normal
    vec3 localNorm = normalize(mat3(t_inv) * vNormRad.xyz);

    //update the curvature information
    vCurv_map_max = vCurvmax;
    vCurv_map_min = vCurvmin;

    //search step in index map
    float indexXStep = (1.0f / (cols * scale)) * 0.5f;
    float indexYStep = (1.0f / (rows * scale)) * 0.5f;

    //search window around current pixel coordinate
    float windowMultiplier = window_multiplier;

    //find neighbor points
    vec4 vertConf_neighbor[100];
    vec4 normRad_neighbor[100];
    int index_N = 0;

    int count = 0;
    int zCount = 0;

    uint index_submap = uint(vColor.y);
    float halfPixel = 0.5 / KeyFrameIDDimen;
    float active = float(textureLod(KeyFrameIDMap, vec2(float(index_submap) / KeyFrameIDDimen + halfPixel, 0.5), 0.0));

    //cleanning vertices
    if(localPos.z < maxDepth && localPos.z > 0 && x > 0 && y > 0 && x < cols && y < rows)
    {
        for(float i = x / cols - (scale * indexXStep * windowMultiplier); i < x / cols + (scale * indexXStep * windowMultiplier); i += indexXStep)
        {
           for(float j = y / rows - (scale * indexYStep * windowMultiplier); j < y / rows + (scale * indexYStep * windowMultiplier); j += indexYStep)
           {
              //global model point
              uint current = uint(textureLod(indexSampler, vec2(i, j), 0));
              if(current > 0U)
              {
                vec4 vertConf = textureLod(vertConfSampler, vec2(i, j), 0);
                vec4 colorTime = textureLod(colorTimeSampler, vec2(i, j), 0);
                vec4 normRad = textureLod(normRadSampler, vec2(i, j), 0);

                //delete similar point(for previous global model)
                if(colorTime.z < vColor.z &&
                    vertConf.w > confThreshold && //stable
                    vertConf.z > localPos.z &&    //in front of the index sample
                    vertConf.z - localPos.z < 0.01 && //avoid depth inconsistency
                    sqrt(dot(vertConf.xy - localPos.xy, vertConf.xy - localPos.xy)) < vNormRad.w * 1.4) //radii overlap
                {
                    count++;
                }

                //free-space violation(for newly added vertex)
                if(colorTime.w == time &&  
                    vertConf.w > confThreshold &&
                    vertConf.z > localPos.z &&
                    vertConf.z - localPos.z > 0.01 &&
                    abs(localNorm.z) > 0.85f &&
                    active > 0.0)
                {
                    zCount++;
                }
              }
            }
        }
    }

    if(vCurv_map_max.w < -curvature_valid_threshold || vCurv_map_max.w > curvature_valid_threshold ||
       vCurv_map_min.w < -curvature_valid_threshold || vCurv_map_min.w > curvature_valid_threshold)
    {
        test = 0;       
    }

    if(count > 8 || zCount > 4)
    {
        test = 0;      
    }

    //this for New unstable point from new frame
    if(vColor.w == -2)
    {
        vColor.w = time;
    }

    //Degenerate(has been updated in previous step) case or too unstable
    if(vColor.w == -1 || ((time - vColor.w) > 200 && vPosition.w < confThreshold))
    {
        test = 0;
    }

}
