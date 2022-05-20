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

out vec4 normal_opt;

uniform sampler2D VertexFilteredSampler;
uniform sampler2D NormalSampler;
uniform float cols;
uniform float rows;
uniform vec4 cam;
uniform float radius_multiplier;
uniform int windowSize;

#include "surfels.glsl"
#include "geometry.glsl"
#include "utils.glsl" 

void main()
{
	float x = texcoord.x * cols;
    float y = texcoord.y * rows;

	vec3 vertexFiltered = texture(VertexFilteredSampler, texcoord.xy, 0.0).xyz;
	vec4 normal = texture(NormalSampler, texcoord.xy, 0.0);                     //normal map 

    if(vertexFiltered.z < 0.3 || length(normal.xyz) < 0.8)
	     discard;
	vec3 vec_p_viewer = normalize(vertexFiltered);

	float sum = 0.0;
	float count = 0.0;

	int windowMultiplier = windowSize;
	float indexXstep = 1.0f / cols;
	float indexYstep = 1.0f / rows;

	for(float i = texcoord.x - (indexXstep * windowMultiplier); i <= texcoord.x + (indexXstep * windowMultiplier); i += indexXstep){
        for(float j = texcoord.y - (indexYstep * windowMultiplier); j <= texcoord.y + (indexYstep * windowMultiplier); j += indexYstep){
            if(i == texcoord.x && j == texcoord.y)
			    continue;
			vec3 vertex_n = texture(VertexFilteredSampler, vec2(i, j), 0.0).xyz;
			vec4 normal_n = texture(NormalSampler, vec2(i, j), 0.0);

			vec3 vec_p_n = normalize(vertex_n - vertexFiltered);

			if(vertex_n.z > 0.3 && length(normal_n.xyz) > 0.8 && dot(vec_p_viewer, vec_p_n) < 0.95)
			{
				sum = sum + dot(normalize(vertexFiltered - vertex_n), normal_n.xyz);
				count++;
			}
	   }
	}
	normal_opt = normal;
	// if(count > 0.0)
	//     normal_opt.w = sum / count; //exp(-(sum / count) * (sum / count) / 0.8) * normal.w;  /// count;
}
