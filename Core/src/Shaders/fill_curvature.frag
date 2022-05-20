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

out vec4 curvk1;
out vec4 curvk2;

uniform sampler2D ecurvk1Sampler;
uniform sampler2D ecurvk2Sampler;
uniform sampler2D rcurvk1Sampler;
uniform sampler2D rcurvk2Sampler;
uniform vec4 cam; //cx, cy, 1/fx, 1/fy
uniform float cols;
uniform float rows;
uniform int passthrough;

void main()
{
    vec4 samplecurv1 = vec4(textureLod(ecurvk1Sampler, texcoord, 0.0));
    vec4 samplecurv2 = vec4(textureLod(ecurvk2Sampler, texcoord, 0.0));

    if(samplecurv1.w > 300 || samplecurv2.w > 300 ||
       passthrough == 1)
    {
        curvk1 = textureLod(rcurvk1Sampler, texcoord, 0.0);  
        curvk2 = textureLod(rcurvk2Sampler, texcoord, 0.0);
    }
    else
    {
        curvk1 = samplecurv1;
        curvk2 = samplecurv2;
    }
}
