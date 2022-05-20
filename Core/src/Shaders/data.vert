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
flat out int updateId;

uniform sampler2D cSampler;
uniform sampler2D drSampler;
uniform sampler2D drfSampler;
uniform sampler2D new_curv1_Samp;
uniform sampler2D new_curv2_Samp;
uniform sampler2D new_conf_Samp;

uniform usampler2D indexSampler;
uniform sampler2D vertConfSampler;
uniform sampler2D colorTimeSampler;
uniform sampler2D normRadSampler;

uniform vec4 cam; //cx, cy, 1/fx, 1/fy
uniform float cols;
uniform float rows;
uniform float scale;
uniform float texDim;
uniform mat4 pose;
uniform float maxDepth;
uniform float time;
uniform float weighting;
uniform float RadiusMultiplier;
uniform float curvature_valid_threshold;
uniform float PCAforNormalEstimation;

uniform float indexSubmap;
uniform bool insertSubmap;

#include "surfels.glsl"
#include "color.glsl"
#include "geometry.glsl"
#include "utils.glsl"

void main()
{
    //Should be guaranteed to be in bounds and centred on pixels
    float x = texcoord.x * cols;
    float y = texcoord.y * rows;

    //Vertex position integrated into model transformed to global coords
    vec3 vPosLocal = getVertex(texcoord.xy, x, y, cam, drSampler);
    vPosition = pose * vec4(vPosLocal, 1.0f);

    //Filtered position ONLY used for normal and radius calculation
    vec3 vPosition_f = getVertex(texcoord.xy, x, y, cam, drfSampler);

    //Color for color, obviously
    vColor = textureLod(cSampler, texcoord.xy, 0.0);

    vColor.x = encodeColor(vColor.xyz);
    vColor.y = indexSubmap;
    vColor.z = time;

    //Normal and radius computed with filtered position / depth map transformed to global coords
    vec3 vNormLocal = vec3(0.0f, 0.0f, 0.0f);   
    if(PCAforNormalEstimation > 0.0)
    {
        //using PCA for normal estimation
        vNormLocal = getNormalPCA(vPosition_f, texcoord.xy, x, y, cam, cols, rows, 3.0, drfSampler);
    }else
    {
        //using central-difference for normal estimation
        if(checkNeighbours(texcoord.xy, drSampler))
            vNormLocal = getNormal(vPosition_f, texcoord.xy, x, y, cam, drfSampler);
    }
    
    vNormRad = vec4(mat3(pose) * vNormLocal, RadiusMultiplier * getRadius(vPosition_f.z, vNormLocal.z));

    curv_map_max = textureLod(new_curv1_Samp, texcoord.xy, 0.0);
    curv_map_min = textureLod(new_curv2_Samp, texcoord.xy, 0.0);

    //Confidence
    vPosition.w = float(textureLod(new_conf_Samp, texcoord.xy, 0.0));

    //Timestamp
    //We update this in update.vert, because it's used to signal new points later on
    vColor.w = 0;

    updateId = 0;

    uint best = 0U;

    //If this point is actually a valid vertex (i.e. has depth)
    if(int(x) % 2 == int(time) % 2 && int(y) % 2 == int(time) % 2 && 
    //    angleBetween(vPosLocal, vNormLocal) < 1.30 &&
       length(vNormLocal) > 0.8 &&
       vPosLocal.z > 0.3 &&
       vPosLocal.z <= maxDepth && 
       curv_map_max.w > - 300.0f && curv_map_max.w < 300.0f &&
       curv_map_min.w > - 300.0f && curv_map_min.w < 300.0f)
    {
	    int counter = 0;

	    float indexXStep = (1.0f / (cols * scale)) * 0.5f;
	    float indexYStep = (1.0f / (rows * scale)) * 0.5f;

	    float bestDist = 1000;

	    float windowMultiplier = 2;

        float xl = (x - cam.x) * cam.z;
        float yl = (y - cam.y) * cam.w;

        float lambda = sqrt(xl * xl + yl * yl + 1);

        vec3 ray = vec3(xl, yl, 1);

	    for(float i = texcoord.x - (scale * indexXStep * windowMultiplier); i < texcoord.x + (scale * indexXStep * windowMultiplier); i += indexXStep){
	        for(float j = texcoord.y - (scale * indexYStep * windowMultiplier); j < texcoord.y + (scale * indexYStep * windowMultiplier); j += indexYStep){
	           uint current = uint(textureLod(indexSampler, vec2(i, j), 0.0));
	           if(current > 0U)
	           {
                   vec4 vertConf = textureLod(vertConfSampler, vec2(i, j), 0.0);
                   vec4 colorTime = textureLod(colorTimeSampler, vec2(i, j), 0.0);

                   if(abs((vertConf.z * lambda) - (vPosLocal.z * lambda)) < 0.05 
                      //&& colorTime.y == vColor.y
                      )
                   {
                       float dist = length(cross(ray, vertConf.xyz)) / length(ray);

                       vec4 normRad = textureLod(normRadSampler, vec2(i, j), 0.0);

                       if(dist < bestDist && (abs(normRad.z) < 0.75f ||
                          abs(angleBetween(normRad.xyz, vNormLocal.xyz)) < 0.5f))
                       {
                           counter++;
                           bestDist = dist;
                           best = current;
                       }
                   }
	            }
	        }
	    }

	    //We found a point to merge with
	    if(counter > 0)
	    {
	       updateId = 1;
	       vColor.w = -1;
	    }
	    else
	    {
	       //New unstable vertex
	       updateId = 2;
	       vColor.w = -2;
	    }
    }

    //Output vertex id of the existing point to update
    if(updateId == 1)
    {
	    uint intY = best / uint(texDim);
	    uint intX = best - (intY * uint(texDim));

	    float halfPixel = 0.5 * (1.0f / texDim);

	    //should set gl_Position here to the 2D index for the updated vertex ID
	    gl_Position = vec4(float(int(intX) - (int(texDim) / 2)) / (texDim / 2.0) + halfPixel,
	                       float(int(intY) - (int(texDim) / 2)) / (texDim / 2.0) + halfPixel,
	                       0,
	                       1.0);
    }
    else
    {
        //Either don't render anything, or output a new unstable vertex offscreen
        gl_Position = vec4(-10, -10, 0, 1);
    }
}
