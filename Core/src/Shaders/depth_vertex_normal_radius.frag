
#version 330 core

in vec2 texcoord;

layout(location = 0) out vec4  vertex_raw;
layout(location = 1) out vec4  vertex_filtered;
layout(location = 2) out vec4  normal;
layout(location = 3) out float radius;

uniform sampler2D depthRawSampler;
uniform sampler2D depthFilteredSampler;
uniform float cols;
uniform float rows;
uniform vec4 cam;
uniform float radius_multiplier;
uniform float PCAforNormalEstimation;

#include "surfels.glsl"
#include "geometry.glsl"
#include "utils.glsl"

void main()
{
	float x = texcoord.x * cols;
  	float y = texcoord.y * rows;

	vec3 vPosLocal   = getVertex(texcoord.xy, int(x), int(y), cam, depthRawSampler);
  	vec3 vPosLocal_f = getVertex(texcoord.xy, int(x), int(y), cam, depthFilteredSampler);

	vec3 vNormLocal = vec3(0.0, 0.0, 0.0);
	if(PCAforNormalEstimation > 0.0)
	{
		//using PCA for normal estimation
		vNormLocal = getNormalPCA(vPosLocal_f, texcoord.xy, x, y, cam, cols, rows, 3.0, depthFilteredSampler);
	}else
	{
		//using central-difference for normal estimation
		if(checkNeighbours(texcoord, depthRawSampler))
			vNormLocal  = getNormal(vPosLocal_f, texcoord.xy, int(x), int(y), cam, depthFilteredSampler);
	}
	
	//init radius; 3*3 pixel patch length ,default 4
	float radius_init = radius_multiplier * getRadius(vPosLocal_f.z, vNormLocal.z);

	// vec3 vPosLocal   = getVertex(texcoord.xy, x, y, cam, depthRawSampler);
  	// vec3 vPosLocal_f = getVertex(texcoord.xy, x, y, cam, depthFilteredSampler);
	// vec3 vNormLocal  = getNormal(vPosLocal_f, texcoord.xy, x, y, cam, depthFilteredSampler);

	//invalid posisition and normal(we delete depth values on the boundary of the image)
	if(length(vNormLocal) < 0.3 || vPosLocal.z < 0.3 || vPosLocal_f.z < 0.3
	   //to prevent curvature estimation error, we try to discard point with a certain radius
	   //|| x < 6.0f || y < 6.0f || x > (cols - 6) || y > (rows - 6)
	   )
	{
		vPosLocal  = vec3(0.0, 0.0, 0.0);
		vPosLocal_f = vec3(0.0, 0.0, 0.0);
		vNormLocal = vec3(0.0, 0.0, 0.0);
		radius_init = 0.0f;
	}

	//get vertex, normal, and radius.
	float max_dist = sqrt((rows * 0.5) * (rows * 0.5) + (cols * 0.5) * (cols * 0.5));
	vertex_raw = vec4(vPosLocal.xyz, confidence(x, y, max_dist, 1.0));
	vertex_filtered = vec4(vPosLocal_f.xyz, 1.0);
	normal = vec4(vNormLocal.xyz, radius_init);
	radius = radius_init;
}
