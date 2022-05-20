//Created by Yabin Xu

#version 330

in vec2 texcoord;

uniform usampler2D indexSampler;
uniform sampler2D vertConfSampler;
uniform sampler2D colorTimeSampler;
uniform sampler2D normRadSampler;
uniform sampler2D curv_maxSampler;
uniform sampler2D curv_minSampler;

// uniform sampler2D surfelVertSampler;
// uniform sampler2D surfelNormSampler;

uniform vec4  cam;  //cx, cy ,1/fx, 1/fy
uniform float cols;
uniform float rows;

uniform float scale;
uniform float winMultiply;
uniform int   predict_minimum_neighbors;
uniform int   predict_maximum_neighbors;
uniform float icp_weight_lambda;
uniform float predict_confidence_threshold;

layout(location = 0) out vec4 image;
layout(location = 1) out vec4 vertex;
layout(location = 2) out vec4 normal;
layout(location = 3) out vec4 principal_curvature_max;
layout(location = 4) out vec4 principal_curvature_min;
layout(location = 5) out uint time;
layout(location = 6) out float icp_weight;

#include "hrbfbase.glsl"
#include "color.glsl"
#include "utils.glsl"

void main()
{
   float x = texcoord.x * cols;
   float y = texcoord.y * rows;

   //deproject to virtual image plane
   float xl = (float(x) - cam.x) * cam.z;
   float yl = (float(y) - cam.y) * cam.w;

   //normalized viewing ray
   vec3 ray = normalize(vec3(xl, yl, 1.0));

   //resolution of Index Map
   float indexXStep = (1.0 / (cols * scale));
   float indexYStep = (1.0 / (rows * scale));

   //search window in Index Map
   float windowMultiplier = winMultiply;

   //number of vertices found in searching window
   int index_N = 0;

   //vertices arributes
   vec4 vertConf_arrayNB[100];
   vec4 normRad_arrayNB[100];
   vec4 colorTime_arrayNB[100];
   vec4 curvMax_arrayNB[100];
   vec4 curvMin_arrayNB[100];
   float confidence_arrayNB[100];
   float weighted_sum_w = 0.0f;
   float confidence = 0.0f;
   float radius = 0.0f;

   //sample neighboring vertices
   float epsilon = 0.00001;
   for(float i = 0; i < windowMultiplier + epsilon; i++)
   {
      for(float j = texcoord.x - (scale * indexXStep * i); j < texcoord.x + (scale * indexXStep * i) + epsilon; j+=indexXStep){
         for(float k = texcoord.y - (scale * indexYStep * i); k < texcoord.y + (scale * indexYStep * i) + epsilon; k += indexYStep){
            if((j - (texcoord.x - (scale * indexXStep * i))) < epsilon || (k - (texcoord.y - (scale * indexYStep * i))) < epsilon ||
               (texcoord.x + (scale * indexXStep * i) - j) < epsilon || (texcoord.y + (scale * indexYStep * i) - k) < epsilon)
            {
               if(j < 0.0 || j > 1.0 || k < 0.0 || k > 1.0)
                   continue;
            
               vec4 vPositionConf = textureLod(vertConfSampler, vec2(j,k), 0.0);
               vec4 vNormalRad0   = textureLod(normRadSampler,  vec2(j,k), 0.0);
               vec4 vColorTime    = textureLod(colorTimeSampler,vec2(j,k), 0.0);
               vec4 vCurvmax0     = textureLod(curv_maxSampler, vec2(j,k), 0.0);
               vec4 vCurvmin0     = textureLod(curv_minSampler, vec2(j,k), 0.0);

               // vec4 vertSurfel     = textureLod(surfelVertSampler, vec2(j,k), 0.0);
               // vec4 normSurfel     = textureLod(surfelNormSampler, vec2(j,k), 0.0);
               //select points consistent with present direction of the ray
               if(vPositionConf.z < 0.1 || length(vNormalRad0.xyz) < 0.1 || vPositionConf.w < predict_confidence_threshold
                  //|| dot(ray, vNormalRad0.xyz) < 0.15 //|| dot(vNormalRad0.xyz, normSurfel.xyz) < 0.50
                  ||vNormalRad0.z < 0.0f
                  ){       
                     continue;
                  }
               vertConf_arrayNB[index_N] = vPositionConf;
               confidence_arrayNB[index_N] = vPositionConf.w;
               //weighted_sum_w = weighted_sum_w + vPositionConf.w;
               normRad_arrayNB[index_N] = vNormalRad0;
               colorTime_arrayNB[index_N] = vColorTime;
               curvMax_arrayNB[index_N] = vCurvmax0;
               curvMin_arrayNB[index_N] = vCurvmin0;
               index_N++;
               if(index_N > predict_maximum_neighbors)
                  break;
            }
         }
      }
   }

     
   //initialize vertices and corresponding attributes
   vec4 image_v = vec4(0.0f, 0.0f, 0.0f, 0.0f);
   vec3 p_surface = vec3(0.0f, 0.0f, 0.0f);
   vec3 p_normal = vec3(0.0f, 0.0f, 0.0f);
   vec4 curv_max_nearest = vec4(0.0f, 0.0f, 0.0f, 1000.0f);
   vec4 curv_min_nearest = vec4(0.0f, 0.0f, 0.0f, 1000.0f);
   float icp_weight_ = 0.0f;

   //searching parameters initialization
   vec3 p_temp = vec3(0.0f, 0.0f, 0.0f);
   vec3 normal_temp = vec3(0.0f, 0.0f, 0.0f);
   float f_temp = 0.0f;
   vec3 starting_point = vec3(0.0f, 0.0f, 0.0f);
   vec3 ending_point = vec3(0.0f, 0.0f, 0.0f);

   vec3 closest_point_toRay = vec3(0.0f, 0.0f, 0.0f);
   float projection_length_min_toRay = 1000000.0f;
   //project current vertices to the viewing ray, find closest and furthest point
   for(int i = 0; i < index_N; i++){      
      float pj = abs(dot(vertConf_arrayNB[i].xyz, ray));
      float pk = length(cross(vertConf_arrayNB[i].xyz, ray));
      if(pj < projection_length_min_toRay)
      {
         closest_point_toRay = pj * ray;
         projection_length_min_toRay = pj;
      }
   }

   bool find_interval = false;
   bool startingP_find = false;
   bool endingP_find = false;
   bool find_surfaceP = false;
   float f_str, f_end;

   //bisection to find the surface prediction, we search the surface point start from the point closest to the ray
   int number_support_points = 0;
   if(index_N > predict_minimum_neighbors)
   {
      //initialize closest_point as starting point
      float v0;    
      v0 = hrbfvalue(closest_point_toRay, vertConf_arrayNB, normRad_arrayNB, index_N, number_support_points);
      if(number_support_points > predict_minimum_neighbors) 
      {
         if(v0 > 0)
         {
            //search backward
            ending_point = closest_point_toRay;
            int steps_N = 25;
            for(int i = 0; i < steps_N; i++)
            {
               vec3 p1 = ending_point - 0.004 * i * ray;
               float v1;
               v1 = hrbfvalue(p1, vertConf_arrayNB, normRad_arrayNB, index_N, number_support_points);
               if(v1 < 0)
               {
                  starting_point = p1;
                  f_str = v1;
                  startingP_find = true;
                  break;
               }
            }

            if(startingP_find == true)
            {
               for(int i = 1; i < 11;i++)
               {
                  vec3 p2 = starting_point + 0.0004 * i * ray;
                  float v2;
                  v2 = hrbfvalue(p2, vertConf_arrayNB, normRad_arrayNB, index_N, number_support_points);
                  if(v2 > 0)
                  {
                     ending_point = p2;
                     f_end = v2;
                     find_interval = true;
                     break;
                  }
               }
            }
         }else
         {
            //search forward
            starting_point = closest_point_toRay;
            int steps_N = 25;
            //find ending points first,then find the starting point
            for(int i = 0; i < steps_N; i++){
               vec3 p1 = starting_point +  0.004 * i * ray;
               float v1;
               v1 = hrbfvalue(p1, vertConf_arrayNB, normRad_arrayNB, index_N, number_support_points);
               if(v1 > 0){
                  ending_point = p1;
                  f_end = v1;
                  endingP_find = true;
                  break;
               }
            }

            if(endingP_find == true)
            {
               //search for starting points
               for(int i = 1; i < 11;i++)
               {
                  vec3 p2 = ending_point -  0.0004 * i * ray;
                  float v2;
                  v2 = hrbfvalue(p2, vertConf_arrayNB, normRad_arrayNB, index_N, number_support_points);
                  if(v2 < 0)
                  {
                     starting_point = p2;
                     f_str =  v2;
                     find_interval = true;
                     break;
                  }
               }           
            }   
         }
      }     
   }

   //search under current interval
   if(find_interval == true)
   {               
      for(int j = 0; j < 10; j++)
      {
         vec3 step = ending_point - starting_point;

         if(getlength(step) < 0.00001)
         {
            p_surface = p_temp;
            normal_temp = hrbfgradient(p_surface, vertConf_arrayNB, normRad_arrayNB, index_N);
            find_surfaceP = true;
            break;
         }

         p_temp = starting_point + 0.5 * step;
         f_temp = hrbfvalue(p_temp, vertConf_arrayNB, normRad_arrayNB, index_N, number_support_points);

         if(abs(f_temp) < 0.00001)
         {
            p_surface = p_temp;
            normal_temp = hrbfgradient(p_surface, vertConf_arrayNB, normRad_arrayNB, index_N);
            find_surfaceP = true;
            break;
         }                   

         if(f_temp < 0)
         {
            starting_point = p_temp;
            f_str = f_temp;

         }
         else{
            ending_point = p_temp;
            f_end = f_temp;
         }
      }
   }

   //if intersected points found, find the corresponding normal
   if(find_surfaceP == true)
   {
      //find points and calculate correspondending normal
      p_surface = p_temp;
      p_normal = normalize(normal_temp);

      //get weighted sum of the confidence of the neighbors if valid curvature obtained
      float weight_sum = 0.0;

      //for other vertex attributes, choose the nearest 
      float dist_smallest = 1000000;
      for(int it = 0; it < index_N; it++)
      {
            float dist = sqrt((p_surface.x - vertConf_arrayNB[it].x) * (p_surface.x - vertConf_arrayNB[it].x) +
                              (p_surface.y - vertConf_arrayNB[it].y) * (p_surface.y - vertConf_arrayNB[it].y) +
                              (p_surface.z - vertConf_arrayNB[it].z) * (p_surface.z - vertConf_arrayNB[it].z));
            if(dist < dist_smallest)
            {
               curv_max_nearest = curvMax_arrayNB[it];
               curv_min_nearest = curvMin_arrayNB[it];
               confidence = vertConf_arrayNB[it].w;
               radius = normRad_arrayNB[it].w;
               image_v = vec4(decodeColor(colorTime_arrayNB[it].x), 1);
               time = uint(colorTime_arrayNB[it].z);
               dist_smallest = dist;
            }
      }
      float lambda = icp_weight_lambda;
      float cmax = abs(curv_max_nearest.w) > abs(curv_min_nearest.w) ? abs(curv_max_nearest.w) : abs(curv_min_nearest.w);
      icp_weight_=(1.0f / (p_surface.z * p_surface.z)) * (confidence / 256.0f + exp(-0.5 * (lambda * lambda) / (cmax * cmax)));  
   }

   image = image_v;
   vertex = vec4(p_surface, confidence);
   normal = vec4(p_normal, radius);
   principal_curvature_max = curv_max_nearest;
   principal_curvature_min = curv_min_nearest;
   icp_weight = icp_weight_;
}
