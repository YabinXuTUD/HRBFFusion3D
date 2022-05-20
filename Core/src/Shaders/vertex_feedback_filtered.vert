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

out float zVal;

uniform sampler2D gSampler;
uniform sampler2D cSampler;
uniform vec4 cam; //cx, cy, 1/fx, 1/fy
uniform float cols;
uniform float rows;
uniform int time;
uniform float maxDepth;
uniform float windowMultiply;
uniform float radius_multiplier;

#include "surfels.glsl"
#include "color.glsl"
#include "geometry.glsl"
#include "utils.glsl" 
#include "hrbfbase.glsl"
//compute curvature on this vertex

void main()
{
    //Should be guaranteed to be in bounds, unit: m
    float x = texcoord.x * cols;
    float y = texcoord.y * rows;

    vPosition = vec4(getVertex(texcoord.xy, int(x), int(y), cam, gSampler), 1);
    vColor = textureLod(cSampler, texcoord.xy, 0.0);
    
    curv_map_max = vec4(0, 0, 0, 1000);
    curv_map_min = vec4(0, 0, 0, 1000);

    int index_frame = time;
    if(index_frame == 1)
    {
        float curvature_m = 1000.0f;
        float curvature_g = 1000.0f;
        float k1 = 1000.0f;
        float k2 = 1000.0f;
        vec3 principalc_max = vec3(0.0,0.0,0.0);
        vec3 principalc_min = vec3(0.0,0.0,0.0);

        int index_N = 0;
        vec4 vertConf_arrayNB[100];
        vec4 normRad_arrayNB[100];


        int scale = 1;
        float windowMultiplier = windowMultiply;
        float indexXStep = (1.0 / (cols * scale));
        float indexYStep = (1.0 / (rows * scale));

        //find neighbors
        for(float i = texcoord.x - (scale * indexXStep * windowMultiplier); i < texcoord.x + (scale * indexXStep * windowMultiplier); i += indexXStep){
            for(float j = texcoord.y - (scale * indexYStep * windowMultiplier); j < texcoord.y + (scale * indexYStep * windowMultiplier); j += indexYStep){
                float valueNB = float(texture(gSampler, vec2(i, j)));
                //check neighbor for normal estimation
                if(valueNB > 0.0 && checkNeighbours(vec2(i, j), gSampler)){
                    float cx = i * cols;
                    float cy = j * rows;
                    vec3 vPoseCurrN = getVertex(vec2(i,j), cx, cy, cam, gSampler);
                    vertConf_arrayNB[index_N] = vec4(vPoseCurrN.xyz, 0.0);
                    vec3 vNormCurrN = getNormal(vPoseCurrN.xyz, vec2(i,j), cx, cy ,cam ,gSampler);
                    normRad_arrayNB[index_N] = vec4(vNormCurrN.xyz, radius_multiplier * getRadius(vPoseCurrN.z, vNormCurrN.z));
                    index_N++;
                }
            }
        }

        if(index_N > 10)
        {
            vec3 gradient = hrbfgradient(vPosition.xyz, vertConf_arrayNB, normRad_arrayNB, index_N);
            vec3 gradient1 = gradient;
            gradient1 = normalize(gradient1);
            vec4 normals = vec4(gradient1,getRadius(vPosition.z, gradient1.z));
            float g[9];
            hrbfHessianMatrix(g, vPosition.xyz, vertConf_arrayNB, normRad_arrayNB, index_N);

            float h_x = - gradient[0] / gradient[2];
            float h_y = - gradient[1] / gradient[2];

            float h_xx = (2 * gradient[0] * gradient[2] * g[2] - gradient[0] * gradient[0] * g[8] - gradient[2] * gradient[2] * g[0]) / (gradient[2] * gradient[2] * gradient[2]);
            float h_xy = (gradient[0] * gradient[2] * g[5] + gradient[1] * gradient[2] * g[2] - gradient[0] * gradient[1] * g[8] - gradient[2] * gradient[2] * g[1]) / (gradient[2] * gradient[2] * gradient[2]);
            float h_yy = (2 * gradient[1] * gradient[2] * g[5] - gradient[1] * gradient[1] * g[8] - gradient[2] * gradient[2] * g[4]) / (gradient[2] * gradient[2] * gradient[2]);

            vec3 r_u = vec3(1, 0, h_x);
            vec3 r_v = vec3(0, 1, h_y);

            //first fundamental form coefficients
            float E = 1 + h_x * h_x;
            float F = h_x * h_y;
            float G = 1 + h_y * h_y;

            //second foundamental form coefficients
            float length = sqrt(h_x * h_x + h_y * h_y + 1);
            float L = h_xx / length;
            float M = h_xy / length;
            float N = h_yy / length;

            curvature_g = (L * N - M * M) / (E * G - F * F);
            curvature_m = (E * N + G * L - 2 * F * M) / (2 * (E * G - F * F));

            //compute principal curvature, mean curvature and gaussian curvature should be valid
            if(!isnan(curvature_g) && !isnan(curvature_m))
            {
                float delta = curvature_m * curvature_m - curvature_g;
                if (delta < 0.0)
                {
                    //curvature_m = 0.0f;
                    delta = 0.0;
                }
                k1 = curvature_m + sqrt(delta);
                k2 = curvature_m - sqrt(delta);

                float lamda_max = - (M - k1 * F) / (N - k1 * G);
                float lamda_min = - (M - k2 * F) / (N - k2 * G);

                //principal dierection
                principalc_max = r_u + lamda_max * r_v;
                principalc_max = normalize(principalc_max);
                principalc_min = r_u + lamda_min * r_v;
                principalc_min = normalize(principalc_min);
            }
            curv_map_max = vec4(principalc_max, k1);
            curv_map_min = vec4(principalc_min, k2);
        }
    }
    vec3 vNormLocal = getNormal(vPosition.xyz, texcoord.xy, x, y, cam, gSampler);
    // if(dot(vNormLocal, normalize(vPosition.xyz)) > 0.2){
         vNormRad = vec4(vNormLocal, getRadius(vPosition.z, vNormLocal.z));
    // }else{
    //     vNormRad = vec4(0.0, 0.0, 0.0, 0.0);
    // }

    //to obtain reliable normal, always check neighbors, this is the feed_back
    if(vPosition.z <= 0 || vPosition.z > maxDepth || !checkNeighbours(texcoord, gSampler))
    {
	    zVal = 0;
    }
    else
    {
        zVal = vPosition.z;
    }

    float max_dist = sqrt((rows * 0.5)*(rows * 0.5) + (cols * 0.5)*(cols * 0.5));
    vPosition.w = confidence(x, y, max_dist, 1.0f);
    
    vColor.x = encodeColor(vColor.xyz);
    
    vColor.y = 0;
    //Timestamp
    vColor.w = float(time);

}
