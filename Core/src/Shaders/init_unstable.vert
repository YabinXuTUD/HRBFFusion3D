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

layout (location = 0) in vec4 vPos;
layout (location = 1) in vec4 vCol;
layout (location = 2) in vec4 vNorm;
layout (location = 3) in vec4 curvmax;
layout (location = 4) in vec4 curvmin;

out vec4 vPosition;
out vec4 vColor;
out vec4 vNormRad;
out vec4 curv_map_max;
out vec4 curv_map_min;

uniform float curvature_valid_threshold;

flat out float zVal;

//out float valid;
void main()
{
    if(length(vNorm.xyz) > 0.8 && vPos.z > 0.0 &&
       curvmax.w > -curvature_valid_threshold && curvmax.w < curvature_valid_threshold &&
       curvmin.w > -curvature_valid_threshold && curvmin.w < curvature_valid_threshold)
    {
        vPosition = vPos;
        vColor = vCol;
        vColor.y = 0;     //Mark this as submap index for each vertex
        vColor.z = 1;     //This sets the vertex's initialisation time
        vNormRad = vNorm;
        curv_map_max = curvmax;
        curv_map_min = curvmin;

        zVal = 1;
    }else
    {
        zVal = 0;         //invalid vertex points
    }
}
