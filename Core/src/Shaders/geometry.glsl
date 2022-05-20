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

//Central difference on floating point depth maps, we only use float point depth map
//Cam is //cx, cy, 1 / fx, 1 / fy
vec3 getVertex(vec2 texCoord, float x, float y, vec4 cam, sampler2D depth)
{
    float z = float(textureLod(depth, texCoord, 0.0));
    return vec3((x - cam.x) * z * cam.z, (y - cam.y) * z * cam.w, z); //depth to 3D
}

//use int to compute the 3D coordinates, not the half-pixel coordinates
vec3 getVertex(vec2 texcoord, int x, int y, vec4 cam, sampler2D depth)
{
    float z = float(textureLod(depth, texcoord, 0.0));
    return vec3((x - cam.x) * z * cam.z, (y - cam.y) * z * cam.w, z);
}

//Cam is //cx, cy, 1 / fx, 1 / fy, normal point inwards?
vec3 getNormal(vec3 vPosition, vec2 texCoord, float x, float y, vec4 cam, sampler2D depth)
{
    vec3 vPosition_xf = getVertex(vec2(texCoord.x + (1.0 / cols), texCoord.y), x + 1, y, cam, depth);
    vec3 vPosition_xb = getVertex(vec2(texCoord.x - (1.0 / cols), texCoord.y), x - 1, y, cam, depth);
    
    vec3 vPosition_yf = getVertex(vec2(texCoord.x, texCoord.y + (1.0 / rows)), x, y + 1, cam, depth);
    vec3 vPosition_yb = getVertex(vec2(texCoord.x, texCoord.y - (1.0 / rows)), x, y - 1, cam, depth);
    
    vec3 del_x = ((vPosition_xb + vPosition) / 2) - ((vPosition_xf + vPosition) / 2);
    vec3 del_y = ((vPosition_yb + vPosition) / 2) - ((vPosition_yf + vPosition) / 2);
    
    return normalize(cross(del_x, del_y));
}

vec3 getNormal(vec3 vPosition, vec2 texCoord, int x, int y, vec4 cam, sampler2D depth)
{
    vec3 vPosition_xf = getVertex(vec2(texCoord.x + (1.0 / cols), texCoord.y), x + 1, y, cam, depth);
    vec3 vPosition_xb = getVertex(vec2(texCoord.x - (1.0 / cols), texCoord.y), x - 1, y, cam, depth);
    
    vec3 vPosition_yf = getVertex(vec2(texCoord.x, texCoord.y + (1.0 / rows)), x, y + 1, cam, depth);
    vec3 vPosition_yb = getVertex(vec2(texCoord.x, texCoord.y - (1.0 / rows)), x, y - 1, cam, depth);
    
    vec3 del_x = ((vPosition_xb + vPosition) / 2) - ((vPosition_xf + vPosition) / 2);
    vec3 del_y = ((vPosition_yb + vPosition) / 2) - ((vPosition_yf + vPosition) / 2);
    
    return normalize(cross(del_x, del_y));
}

vec3 computeRoots2(float b, float c)
{
    float d = float (b * b - 4.0 * c);
    if (d < 0.0)  // no real roots ! THIS SHOULD NOT HAPPEN!
        d = 0.0;
    float sd = sqrt(d);
    return vec3(0.0f, 0.5f * (b + sd), 0.5f * (b - sd));
}

vec3 computeRoots(mat3 m)
{
    // The characteristic equation is x^3 - c2*x^2 + c1*x - c0 = 0.  The
    // eigenvalues are the roots to this equation, all guaranteed to be
    // // real-valued, because the matrix is symmetric.
    float c0 = m[0][0] * m[1][1] * m[2][2]
      + 2.0f * m[1][0] * m[2][0] * m[2][1]
             - m[0][0] * m[2][1] * m[2][1]
             - m[1][1] * m[2][0] * m[2][0]
             - m[2][2] * m[1][0] * m[1][0];
    float c1 = m[0][0] * m[1][1] -
               m[1][0] * m[1][0] +
               m[0][0] * m[2][2] -
               m[2][0] * m[2][0] +
               m[1][1] * m[2][2] -
               m[2][1] * m[2][1];
    float c2 = m[0][0] + m[1][1] + m[2][2];

    if(abs(c0) < 0.000001f)
    {
        return computeRoots2(c2, c1);
    }else
    {
        vec3 roots;
        const float s_inv3 = float(1.0f / 3.0f);
        const float s_sqrt3 = sqrt(float (3.0));
        // Construct the parameters used in classifying the roots of the equation
        // and in solving the equation for the roots in closed form.
        float c2_over_3 = c2 * s_inv3;
        float a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;
        if (a_over_3 > float(0.0f))
            a_over_3 = float(0.0f);

        float half_b = float(0.5) * (c0 + c2_over_3 * (float(2.0) * c2_over_3 * c2_over_3 - c1));

        float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
        if (q > float(0))
            q = float(0);

        // Compute the eigenvalues by solving for the roots of the polynomial.
        float rho = sqrt(-a_over_3);
        float theta = atan(sqrt(-q), half_b) * s_inv3;
        float cos_theta = cos(theta);
        float sin_theta = sin(theta);
        roots.x = c2_over_3 + float (2) * rho * cos_theta;
        roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
        roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);

        // Sort in increasing order.
        if (roots.x >= roots.y)
        {
            //swap(roots.x, roots.y)
            float temp = roots.x;
            roots.x = roots.y;
            roots.y = temp;
        }
        if (roots.y >= roots.z)
        {
            // swap(roots.y, roots.z);
            float temp = roots.y;
            roots.y = roots.z;
            roots.z = temp;
            if (roots.x >= roots.y)
            {
                // std::swap(roots.x, roots.y);
                float temp1 = roots.x;
                roots.x = roots.y;
                roots.y = temp1;
            }       
        }
        if (roots.x <= 0)  // eigenval for symetric positive semi-definite matrix can not be negative! Set it to 0
            return computeRoots2 (c2, c1);
        return roots;
    }   
}

mat3 computeMeanAndCovarianceMatrix(vec3 points[100], int pointcloud_size)
{
    float accu[9];
    accu[0] = accu[1] = accu[2] = accu[3] = accu[4] = accu[5] = accu[6] = accu[7] = accu[8] = 0.0f;
    mat3 covariance_matrix;
    int point_count = 0;
    for (int i = 0; i < pointcloud_size; i++)
    {
        ++point_count;
        accu[0] += points[i].x * points[i].x;
        accu[1] += points[i].x * points[i].y;
        accu[2] += points[i].x * points[i].z;
        accu[3] += points[i].y * points[i].y; // 4
        accu[4] += points[i].y * points[i].z; // 5
        accu[5] += points[i].z * points[i].z; // 8
        accu[6] += points[i].x;
        accu[7] += points[i].y;
        accu[8] += points[i].z;
    }
    //remember rules of indexing in shader: mat[col][row]
    accu[0] /= float(point_count);
    accu[1] /= float(point_count);
    accu[2] /= float(point_count);
    accu[3] /= float(point_count);
    accu[4] /= float(point_count);
    accu[5] /= float(point_count);
    accu[6] /= float(point_count);
    accu[7] /= float(point_count);
    accu[8] /= float(point_count);

    covariance_matrix[0][0] = accu[0] - accu[6] * accu[6];
    covariance_matrix[1][0] = accu[1] - accu[6] * accu[7];
    covariance_matrix[2][0] = accu[2] - accu[6] * accu[8];
    covariance_matrix[1][1] = accu[3] - accu[7] * accu[7];
    covariance_matrix[2][1] = accu[4] - accu[7] * accu[8];
    covariance_matrix[2][2] = accu[5] - accu[8] * accu[8];
    covariance_matrix[0][1] = covariance_matrix[1].x;
    covariance_matrix[0][2] = covariance_matrix[2].x;
    covariance_matrix[1][2] = covariance_matrix[2].y;
    return covariance_matrix;
}

vec3 getNormalPCA(vec3 vPosition, vec2 texCoord, float x, float y, vec4 cam, float cols, float rows, float winMultiply, sampler2D depth)
{
    float indexXStep = 1.0f / cols;
    float indexYStep = 1.0f / rows;
    //find neighbors
    float tx_min = max(0.0f, texCoord.x - (indexXStep * winMultiply));
    float tx_max = min(1.0f, texCoord.x + (indexXStep * winMultiply));

    float ty_min = max(0.0f, texCoord.y - (indexYStep * winMultiply));
    float ty_max = min(1.0f, texCoord.y + (indexYStep * winMultiply));
    int points_N = 0;
    vec3 points[100];
    for(float i = tx_min; i <= tx_max; i += indexXStep){
         for(float j = ty_min; j <= ty_max; j += indexYStep){
            vec3 xyz = getVertex(vec2(i, j), i * cols, j * rows, cam, depth);
            if(xyz.z > 0.3 && abs(xyz.z - vPosition.z) < 0.05)
            {
                points[points_N] = xyz;
                points_N++;
            }
         }
    }
    if(points_N < 8)
        return vec3(0.0f, 0.0f, 0.0f);

    mat3 covMatrix = computeMeanAndCovarianceMatrix(points, points_N);
    float scale = max(max(max(covMatrix[0][0], covMatrix[1][0]), max(covMatrix[2][0], covMatrix[1][1]))
                    ,max(covMatrix[2][1], covMatrix[2][2]));
    mat3 scaledMat = covMatrix / scale;
    vec3 eigenvalues = computeRoots(covMatrix);
    //smallest eigenvalue
    float eigenvalue = eigenvalues.x * scale;
    scaledMat[0][0] = scaledMat[0][0] - eigenvalue;
    scaledMat[1][1] = scaledMat[1][1] - eigenvalue; 
    scaledMat[2][2] = scaledMat[2][2] - eigenvalue;
    vec3 row_0 = vec3(scaledMat[0][0], scaledMat[1][0], scaledMat[2][0]);
    vec3 row_1 = vec3(scaledMat[0][1], scaledMat[1][1], scaledMat[2][1]);
    vec3 row_2 = vec3(scaledMat[0][2], scaledMat[1][2], scaledMat[2][2]);
    vec3 vec_1 = cross(row_0, row_1);
    vec3 vec_2 = cross(row_0, row_2);
    vec3 vec_3 = cross(row_1, row_2);
    float len1 = length(vec_1);
    float len2 = length(vec_2);
    float len3 = length(vec_3);
    vec3 normal;
    if (len1 >= len2 && len1 >= len3)
        normal = vec_1;
    else if (len2 >= len1 && len2 >= len3)
        normal = vec_2;
    else
        normal = vec_3;
    if(normal.z < 0)
        normal = -normal;
    return normalize(normal);
}



// //Forward difference on raw depth maps still in ushort mm
// //Cam is //cx, cy, 1 / fx, 1 / fy
// vec3 getVertex(vec2 texcoord, int x, int y, vec4 cam, usampler2D depth)
// {
//     float z = float(textureLod(depth, texcoord, 0.0)) / 1000.0f;
//     return vec3((x - cam.x) * z * cam.z, (y - cam.y) * z * cam.w, z);
// }

// //Cam is //cx, cy, 1 / fx, 1 / fy
// vec3 getNormal(vec3 vPosition, vec2 texcoord, int x, int y, vec4 cam, usampler2D depth)
// {
//     vec3 vPosition_x = getVertex(vec2(texcoord.x + (1.0 / cols), texcoord.y), x + 1, y, cam, depth);
//     vec3 vPosition_y = getVertex(vec2(texcoord.x, texcoord.y + (1.0 / rows)), x, y + 1, cam, depth);
    
//     vec3 del_x = vPosition_x - vPosition;
//     vec3 del_y = vPosition_y - vPosition;
    
//     return normalize(cross(del_x, del_y));
// }
