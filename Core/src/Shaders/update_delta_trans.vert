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

//matrix texture for updating the global model
uniform sampler2D DeltaTransformKF;
uniform float DeltaTransDimen;

void main()
{
    uint index_submap = uint(vCol.y);
    vec4 pos_h = vec4(vPos.xyz, 1.0);
    vec3 normal = vNormR.xyz;

    //get current matrix from the texture
    uint matrix_start_position =  index_submap * 16U;
    // //go to the Textcoordinate part
    float onePixel = 1.0 / DeltaTransDimen;
    float halfPixel = 0.5 / DeltaTransDimen;
    //transfer to pixel coordinate
    float start = float(matrix_start_position) / DeltaTransDimen + halfPixel;

    mat4 Trans0;
    for(int i = 0; i < 4; i++)
    {
        float v0 = float(textureLod(DeltaTransformKF, vec2(start + 4 * i * onePixel + 0 * onePixel, 0.5), 0.0));
        float v1 = float(textureLod(DeltaTransformKF, vec2(start + 4 * i * onePixel + 1 * onePixel, 0.5), 0.0));
        float v2 = float(textureLod(DeltaTransformKF, vec2(start + 4 * i * onePixel + 2 * onePixel, 0.5), 0.0));
        float v3 = float(textureLod(DeltaTransformKF, vec2(start + 4 * i * onePixel + 3 * onePixel, 0.5), 0.0));
        Trans0[i] = vec4(v0, v1, v2, v3);
    }

    // //get all 4 column vector of the Transformation Matrix
    // vec4 v0 = textureLod(DeltaTransformKF, vec2(start, 0.5), 0.0);
    // vec4 v1 = textureLod(DeltaTransformKF, vec2(start + 1 * onePixel, 0.5), 0.0);
    // vec4 v2 = textureLod(DeltaTransformKF, vec2(start + 2 * onePixel, 0.5), 0.0);
    // vec4 v3 = textureLod(DeltaTransformKF, vec2(start + 3 * onePixel, 0.5), 0.0);

    // mat4 Trans0; //=  Trans0;
    // Trans0[0] = v0;
    // Trans0[1] = v1;
    // Trans0[2] = v2;
    // Trans0[3] = v3;

    //mat4 Trans = transpose(Trans0);
    mat4 Trans = Trans0;

    mat3 rot = mat3(Trans);
    vec4 pose_update =  Trans * pos_h;
    vec3 noraml_update = rot * normal;

    vPosition = vec4(pose_update.xyz, vPos.w);
    vColor = vCol;
    vNormRad = vec4(noraml_update, vNormR.w);
    vCurv_map_max = vCurvmax;
    vCurv_map_min = vCurvmin;

    test = 1;
}
