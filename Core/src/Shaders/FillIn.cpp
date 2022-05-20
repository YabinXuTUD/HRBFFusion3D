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

#include "FillIn.h"

FillIn::FillIn()
 : imageTexture(Resolution::getInstance().width(),
                Resolution::getInstance().height(),
                GL_RGBA,
                GL_RGB,
                GL_UNSIGNED_BYTE,
                false,
                true),
   vertexTexture(Resolution::getInstance().width(),
                 Resolution::getInstance().height(),
                 GL_RGBA32F,
                 GL_LUMINANCE,
                 GL_FLOAT,
                 false,
                 true),
   normalTexture(Resolution::getInstance().width(),
                 Resolution::getInstance().height(),
                 GL_RGBA32F,
                 GL_LUMINANCE,
                 GL_FLOAT,
                 false,
                 true),
   curvk1Texture(Resolution::getInstance().width(),
                 Resolution::getInstance().height(),
                 GL_RGBA32F,
                 GL_LUMINANCE,
                 GL_FLOAT,
                 false,
                 true),
   curvk2Texture(Resolution::getInstance().width(),
                 Resolution::getInstance().height(),
                 GL_RGBA32F,
                 GL_LUMINANCE,
                 GL_FLOAT,
                 false,
                 true),
   icpweightTexture(Resolution::getInstance().width(),
                    Resolution::getInstance().height(),
                    GL_LUMINANCE32F_ARB,
                    GL_LUMINANCE,
                    GL_FLOAT,
                    false,
                    true),
   imageProgram(loadProgramFromFile("empty.vert", "fill_rgb.frag", "quad.geom")),
   imageRenderBuffer(Resolution::getInstance().width(), Resolution::getInstance().height()),
   vertexProgram(loadProgramFromFile("empty.vert", "fill_vertex.frag", "quad.geom")),
   vertexRenderBuffer(Resolution::getInstance().width(), Resolution::getInstance().height()),
   normalProgram(loadProgramFromFile("empty.vert", "fill_normal.frag", "quad.geom")),
   normalRenderBuffer(Resolution::getInstance().width(), Resolution::getInstance().height()),
   curvatureProgram(loadProgramFromFile("empty.vert", "fill_curvature.frag", "quad.geom")),
   curvatureRenderBuffer(Resolution::getInstance().width(), Resolution::getInstance().height())
{
    imageFrameBuffer.AttachColour(*imageTexture.texture);
    imageFrameBuffer.AttachDepth(imageRenderBuffer);

    vertexFrameBuffer.AttachColour(*vertexTexture.texture);
    vertexFrameBuffer.AttachColour(*icpweightTexture.texture);
    vertexFrameBuffer.AttachDepth(vertexRenderBuffer);

    normalFrameBuffer.AttachColour(*normalTexture.texture);
    normalFrameBuffer.AttachDepth(normalRenderBuffer);

    curvatureFrameBuffer.AttachColour(*curvk1Texture.texture);
    curvatureFrameBuffer.AttachColour(*curvk2Texture.texture);
    curvatureFrameBuffer.AttachDepth(curvatureRenderBuffer);
}

FillIn::~FillIn()
{

}

void FillIn::image(GPUTexture * existingRgb, GPUTexture * rawRgb, bool passthrough)
{
    imageFrameBuffer.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, imageRenderBuffer.width, imageRenderBuffer.height);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    imageProgram->Bind();

    imageProgram->setUniform(Uniform("eSampler", 0));
    imageProgram->setUniform(Uniform("rSampler", 1));
    imageProgram->setUniform(Uniform("passthrough", (int)passthrough));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, existingRgb->texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, rawRgb->texture->tid);

    glDrawArrays(GL_POINTS, 0, 1);

    imageFrameBuffer.Unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);

    imageProgram->Unbind();

    glPopAttrib();

    glFinish();
}

void FillIn::vertex(GPUTexture * existingVertex, GPUTexture * vertexFiltered, GPUTexture * eicpWeight,
                    GPUTexture * rawCurv_k1, GPUTexture * rawCurv_k2, GPUTexture * confidenceMap,
                    int time, float weight, bool passthrough)
{
    vertexFrameBuffer.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, vertexRenderBuffer.width, vertexRenderBuffer.height);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    vertexProgram->Bind();
    vertexProgram->setUniform(Uniform("eSampler", 0));
    vertexProgram->setUniform(Uniform("filteredSampler", 1));
    vertexProgram->setUniform(Uniform("rck1Sampler", 2));
    vertexProgram->setUniform(Uniform("rck2Sampler", 3));
    vertexProgram->setUniform(Uniform("eicpweightSampler", 4));
    vertexProgram->setUniform(Uniform("confidenceSampler", 5));
    vertexProgram->setUniform(Uniform("passthrough", (int)passthrough));
    vertexProgram->setUniform(Uniform("curvature_valid_threshold", GlobalStateParam::get().preprocessingCurvValidThreshold));

    Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
                          Intrinsics::getInstance().cy(),
                          1.0f / Intrinsics::getInstance().fx(),
                          1.0f / Intrinsics::getInstance().fy());

    vertexProgram->setUniform(Uniform("cam", cam));
    vertexProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
    vertexProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));
    vertexProgram->setUniform(Uniform("weight", weight));
    vertexProgram->setUniform(Uniform("time", time));
    vertexProgram->setUniform(Uniform("icp_weight_lambda", GlobalStateParam::get().registrationICPCurvWeightImpactControl));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, existingVertex->texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, vertexFiltered->texture->tid);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, rawCurv_k1->texture->tid);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, rawCurv_k2->texture->tid);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, eicpWeight->texture->tid);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, confidenceMap->texture->tid);

    glDrawArrays(GL_POINTS, 0, 1);

    vertexFrameBuffer.Unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);

    vertexProgram->Unbind();

    glPopAttrib();

    glFinish();
}

void FillIn::normal(GPUTexture * existingNormal, GPUTexture * rawNormal, bool passthrough)
{
    normalFrameBuffer.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, normalRenderBuffer.width, normalRenderBuffer.height);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    normalProgram->Bind();

    normalProgram->setUniform(Uniform("eSampler", 0));
    normalProgram->setUniform(Uniform("rSampler", 1));
    normalProgram->setUniform(Uniform("passthrough", (int)passthrough));

    Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
                  Intrinsics::getInstance().cy(),
                  1.0f / Intrinsics::getInstance().fx(),
                  1.0f / Intrinsics::getInstance().fy());

    normalProgram->setUniform(Uniform("cam", cam));
    normalProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
    normalProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, existingNormal->texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, rawNormal->texture->tid);

    glDrawArrays(GL_POINTS, 0, 1);

    normalFrameBuffer.Unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);

    normalProgram->Unbind();

    glPopAttrib();

    glFinish();
}

void FillIn::curvature(GPUTexture * existingcurvk1Curvature, GPUTexture * existingcurvk2Curvature, GPUTexture * curvk1Curvature, GPUTexture * curvk2Curvature, bool passthrough)
{
    curvatureFrameBuffer.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, curvatureRenderBuffer.width, curvatureRenderBuffer.height);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    curvatureProgram->Bind();

    curvatureProgram->setUniform(Uniform("ecurvk1Sampler", 0));
    curvatureProgram->setUniform(Uniform("ecurvk2Sampler", 1));
    curvatureProgram->setUniform(Uniform("rcurvk1Sampler", 2));
    curvatureProgram->setUniform(Uniform("rcurvk2Sampler", 3));
    curvatureProgram->setUniform(Uniform("passthrough", (int)passthrough));

    Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
                  Intrinsics::getInstance().cy(),
                  1.0f / Intrinsics::getInstance().fx(),
                  1.0f / Intrinsics::getInstance().fy());

    curvatureProgram->setUniform(Uniform("cam", cam));
    curvatureProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
    curvatureProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, existingcurvk1Curvature->texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, existingcurvk2Curvature->texture->tid);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, curvk1Curvature->texture->tid);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, curvk2Curvature->texture->tid);

    glDrawArrays(GL_POINTS, 0, 1);

    curvatureFrameBuffer.Unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);

    curvatureProgram->Unbind();

    glPopAttrib();
    glFinish();

}

void FillIn::downloadtexture(const Eigen::Matrix4f& lastPose,int lastFrames, bool global, std::string groundtruthpose)
{
     Img<Eigen::Matrix<unsigned char, 4, 1>> image(Resolution::getInstance().rows(),Resolution::getInstance().cols());
     Img<Eigen::Vector4f> fillIn_vertices(Resolution::getInstance().rows(),Resolution::getInstance().cols());
     Img<Eigen::Vector4f> fillIn_normals(Resolution::getInstance().rows(),Resolution::getInstance().cols());
     Img<Eigen::Vector4f> fillIn_curvk1down(Resolution::getInstance().rows(),Resolution::getInstance().cols());
     Img<Eigen::Vector4f> fillIn_curvk2down(Resolution::getInstance().rows(),Resolution::getInstance().cols());
     Img<float> icpWeight(Resolution::getInstance().rows(),Resolution::getInstance().cols());

     imageTexture.texture->Download(image.data, GL_RGBA, GL_UNSIGNED_BYTE);
     vertexTexture.texture->Download(fillIn_vertices.data, GL_RGBA, GL_FLOAT);
     normalTexture.texture->Download(fillIn_normals.data, GL_RGBA, GL_FLOAT);
     curvk1Texture.texture->Download(fillIn_curvk1down.data, GL_RGBA, GL_FLOAT);
     curvk2Texture.texture->Download(fillIn_curvk2down.data, GL_RGBA, GL_FLOAT);
     icpweightTexture.texture->Download(icpWeight.data, GL_LUMINANCE, GL_FLOAT);

     cv::Mat imageCV;
     imageCV.create(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_8UC3);
     for (int i = 0; i < imageCV.rows; ++i){
         for (int j = 0; j < imageCV.cols; ++j){
             cv::Vec3b& bgra = imageCV.at<cv::Vec3b>(i, j);
             if(image.at<Eigen::Matrix<unsigned char, 4, 1>>(i,j)(0) == 0 && 
                image.at<Eigen::Matrix<unsigned char, 4, 1>>(i,j)(1) == 0 && 
                image.at<Eigen::Matrix<unsigned char, 4, 1>>(i,j)(2) == 0)
             {
                 bgra[2] = static_cast<unsigned char>(UCHAR_MAX);
                 bgra[1] = static_cast<unsigned char>(UCHAR_MAX);
                 bgra[0] = static_cast<unsigned char>(UCHAR_MAX);
             }else{
                 bgra[2] = image.at<Eigen::Matrix<unsigned char, 4, 1>>(i,j)(0); // Blue
                 bgra[1] = image.at<Eigen::Matrix<unsigned char, 4, 1>>(i,j)(1); // Green
                 bgra[0] = image.at<Eigen::Matrix<unsigned char, 4, 1>>(i,j)(2); // Red
             }

         }
     }

     std::vector<int> compression_params;
     compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
     compression_params.push_back(0);
     cv::imwrite("fillin_colorMap.png", imageCV, compression_params);
     std::cout << "fillin_colorMap saved successfully" << std::endl;

     //output txt
     std::ofstream textureD;
     std::string filename = "./fillin/fillInTex_";
     if(groundtruthpose == "groundtruth")
     {
         filename += "groundtruth_";
     }
     filename += std::to_string(lastFrames) + ".txt";
     textureD.open(filename);
     for(int i = 0; i < fillIn_curvk1down.rows; i++){
         for(int j = 0; j < fillIn_curvk1down.cols; j++)
         {
             if(fillIn_curvk1down.at<Eigen::Vector4f>(i, j)(3) > 300.0 || 
                std::isnan(fillIn_curvk1down.at<Eigen::Vector4f>(i, j)(3)))
                 continue;
             Eigen::Vector4f v_point(fillIn_vertices.at<Eigen::Vector4f>(i, j)(0), fillIn_vertices.at<Eigen::Vector4f>(i, j)(1), fillIn_vertices.at<Eigen::Vector4f>(i, j)(2), 1.0);
             float confidence = fillIn_vertices.at<Eigen::Vector4f>(i, j)(3);
             Eigen::Vector3f n_point(fillIn_normals.at<Eigen::Vector4f>(i, j)(0),fillIn_normals.at<Eigen::Vector4f>(i, j)(1),fillIn_normals.at<Eigen::Vector4f>(i, j)(2));
             float radius = fillIn_normals.at<Eigen::Vector4f>(i, j)(3);
             Eigen::Matrix3f rot = lastPose.topLeftCorner(3, 3);
             if(global)
             {
                 v_point = lastPose * v_point;
                 n_point = rot * n_point;
             }
             textureD << v_point(0)<<" "<<v_point(1)<<" "<<v_point(2)<<" "
                      << n_point(0)<<" "<<n_point(1)<<" "<<n_point(2)<<" "
                      <<" "<<icpWeight.at<float>(i, j)<<" "
                      /*<< fillIn_curvk1down.at<Eigen::Vector4f>(i, j)(0)<<" "<<fillIn_curvk1down.at<Eigen::Vector4f>(i, j)(1)<<" "<<fillIn_curvk1down.at<Eigen::Vector4f>(i, j)(2)<<" "*/
                      /* << fillIn_curvk2down.at<Eigen::Vector4f>(i, j)(0)<<" "<<fillIn_curvk2down.at<Eigen::Vector4f>(i, j)(1)<<" "<<fillIn_curvk2down.at<Eigen::Vector4f>(i, j)(2)<<" "*/
                      <<fillIn_curvk1down.at<Eigen::Vector4f>(i, j)(3) << " " << fillIn_curvk2down.at<Eigen::Vector4f>(i, j)(3) <<" "
                      <<"\n";
         }
     }
     textureD.close();
     std::cout <<"texture saved"<< std::endl;
}


