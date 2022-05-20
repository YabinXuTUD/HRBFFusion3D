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

out float confidence_hrbf;

uniform sampler2D gradient_mag;
uniform sampler2D depthSampler;
uniform float cols;
uniform float rows;
uniform vec4 cam;
uniform float weighting;

uniform float useConfidenceEvaluation;
uniform float epsilon;

#include "surfels.glsl"

void main()
{
    float x = texcoord.x * cols;
    float y = texcoord.y * rows;

    float gradient_mag_ = float(texture(gradient_mag, texcoord.xy, 0.0));
    float d = float(texture(depthSampler, texcoord.xy, 0.0));

    float max_dist = sqrt((rows * 0.5)*(rows * 0.5) + (cols * 0.5)*(cols * 0.5));
    if(useConfidenceEvaluation > 0.0f)
         confidence_hrbf = confidence(x, y, max_dist, weighting) * exp(-epsilon/sqrt(gradient_mag_));
    else
         confidence_hrbf  = confidence(x, y, max_dist, weighting);
}
