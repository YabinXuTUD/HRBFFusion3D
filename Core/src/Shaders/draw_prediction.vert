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

layout (location = 0) in vec4 position;

uniform mat4 MVP;
uniform mat4 pose;

out vec4 vColor;

void main()
{

    if(position.z > 0.3)
    {
        gl_Position = MVP * pose * vec4(position.xyz, 1.0);
        gl_PointSize = 3;    //diameter of a point
        vColor = vec4(0.0, 0.0, 1.0, 1.0);
    }else
    {
        gl_Position = vec4(-10, -10, 0, 1);
    }
}
