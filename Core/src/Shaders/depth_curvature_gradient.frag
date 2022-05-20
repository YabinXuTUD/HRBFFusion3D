//Created by Yabin Xu

#version 330 core

in vec2 texcoord;

layout(location = 0) out vec4  principal_curvature_max;
layout(location = 1) out vec4  principal_curvature_min;
layout(location = 2) out float gradient_mag;
layout(location = 3) out vec4  normal_opt;                 //the bounded radius with h parameter

//filtered vertex map
uniform sampler2D VertexFiltered;
//normal map
uniform sampler2D NormalRadSampler;

uniform vec4 cam;                                           //cx, cy, 1/fx, 1/fy
uniform float cols;
uniform float rows;
uniform float maxD;
uniform float winMultiply;

#include "geometry.glsl"
#include "hrbfbase.glsl"
#include "surfels.glsl"
#include "utils.glsl"

void main()
{
   vec4 vertex_filtered = texture(VertexFiltered, texcoord.xy, 0.0);
   vec4 vertex_normal = texture(NormalRadSampler, texcoord.xy, 0.0);
   principal_curvature_max = vec4(0.0f, 0.0f, 0.0f, 1000.0f); 
   principal_curvature_min = vec4(0.0f, 0.0f, 0.0f, 1000.0f);
   gradient_mag = 0.0f;
   normal_opt = vec4(0.0f, 0.0f, 0.0f, 0.0f);

   if(vertex_filtered.z > 0.3 && length(vertex_normal.xyz) > 0.5)
   {
      float x = texcoord.x * cols;
      float y = texcoord.y * rows;
      vec3 vPoseCurr = vertex_filtered.xyz;

      //initialization
      float curvature_m = 1000.0f;
      float curvature_g = 1000.0f;
      float k1 = 1000.0f;
      float k2 = 1000.0f;
      vec3 principalc_max = vec3(0.0, 0.0, 0.0);
      vec3 principalc_min = vec3(0.0, 0.0, 0.0);

      int index_N = 0;
      vec4 vertConf_arrayNB[100];
      vec4 normRad_arrayNB[100];

      float indexXStep = 1.0f / cols;
      float indexYStep = 1.0f / rows;

      vec3 viewing_ray = normalize(vPoseCurr);

      //find neighbors
      float tx_min = max(0.0f, texcoord.x - (indexXStep * winMultiply));
      float tx_max = min(1.0f, texcoord.x + (indexXStep * winMultiply));

      float ty_min = max(0.0f, texcoord.y - (indexYStep * winMultiply));
      float ty_max = min(1.0f, texcoord.y + (indexYStep * winMultiply));

      for(float i = tx_min; i <= tx_max; i += indexXStep){
         for(float j = ty_min; j <= ty_max; j += indexYStep){
               vec4 vertex_filtered_n = texture(VertexFiltered, vec2(i, j), 0.0);
               vec4 normal_n = texture(NormalRadSampler, vec2(i, j), 0.0);
               vec3 delta_n = normalize(vertex_filtered_n.xyz - vPoseCurr);
               
               if(abs(vertex_filtered_n.z - vPoseCurr.z) < 0.10 &&
                  //abs(dot(viewing_ray, delta_n)) < 0.995 &&
                  vertex_filtered_n.z > 0.3 && length(normal_n.xyz) > 0.8){
                  vertConf_arrayNB[index_N] = vec4(vertex_filtered_n.xyz, 1.0);
                  normRad_arrayNB[index_N]  = vec4(normal_n.xyz, normal_n.w);
                  index_N++;
               }
         }
      }

      if(index_N > 15)
      {
         vec3 gradient = hrbfgradient(vPoseCurr, vertConf_arrayNB, normRad_arrayNB, index_N);
         gradient_mag = abs(dot(gradient, vertex_normal.xyz));
         vec3 gradient1 = gradient;
         gradient1 = normalize(gradient1);
         vec4 normals = vec4(gradient1, getRadius(vPoseCurr.z, gradient1.z));
         normal_opt.xyz = normals.xyz;
         normal_opt.w = texture(NormalRadSampler, texcoord.xy, 0.0).w;
         float g[9];
         hrbfHessianMatrix(g, vPoseCurr, vertConf_arrayNB, normRad_arrayNB, index_N);

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
      }
      principal_curvature_max = vec4(principalc_max, k1);
      principal_curvature_min = vec4(principalc_min, k2);
   }
}
