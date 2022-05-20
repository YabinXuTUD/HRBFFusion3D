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

#include "IndexMap.h"

const int IndexMap::FACTOR = 1;
const int IndexMap::FACTORModel = 1;

const int IndexMap::ACTIVE_KEYFRAME_DIMENSION = 19200;


IndexMap::IndexMap()
: indexProgram(loadProgramFromFile("index_map.vert", "index_map.frag")),
  indexRenderBuffer(Resolution::getInstance().width() * IndexMap::FACTOR, Resolution::getInstance().height() * IndexMap::FACTOR),
  indexTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
               Resolution::getInstance().height() * IndexMap::FACTOR,
               GL_LUMINANCE32UI_EXT,
               GL_LUMINANCE_INTEGER_EXT,
               GL_UNSIGNED_INT),
  vertConfTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
                  Resolution::getInstance().height() * IndexMap::FACTOR,
                  GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
  colorTimeTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
                   Resolution::getInstance().height() * IndexMap::FACTOR,
                   GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
  normalRadTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
                   Resolution::getInstance().height() * IndexMap::FACTOR,
                   GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
  curv_map_maxTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
                   Resolution::getInstance().height() * IndexMap::FACTOR,
                   GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
  curv_map_minTexture(Resolution::getInstance().width() * IndexMap::FACTOR,
                   Resolution::getInstance().height() * IndexMap::FACTOR,
                   GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
  drawDepthProgram(loadProgramFromFile("empty.vert", "visualise_textures.frag", "quad.geom")),
  drawRenderBuffer(Resolution::getInstance().width(), Resolution::getInstance().height()),
  drawTexture(Resolution::getInstance().width(),
              Resolution::getInstance().height(),
              GL_RGBA,
              GL_RGB,
              GL_UNSIGNED_BYTE,
              false),
  depthProgram(loadProgramFromFile("splat.vert", "depth_splat.frag")),
  depthRenderBuffer(Resolution::getInstance().width(), Resolution::getInstance().height()),
  depthTexture(Resolution::getInstance().width(), Resolution::getInstance().height(),
               GL_LUMINANCE32F_ARB,
               GL_LUMINANCE,
               GL_FLOAT,
               false,
               true),
  surfelSplattingProgram(loadProgramFromFile("splat.vert", "combo_splat.frag")),
  surfelSplattingRenderBuffer(Resolution::getInstance().width()* IndexMap::FACTORModel, Resolution::getInstance().height()* IndexMap::FACTORModel),
  imageSurfelTexture(Resolution::getInstance().width() * IndexMap::FACTORModel,
                     Resolution::getInstance().height() * IndexMap::FACTORModel,
                     GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, false, true),
  vertexSurfelTexture(Resolution::getInstance().width() * IndexMap::FACTORModel,
                      Resolution::getInstance().height() * IndexMap::FACTORModel,
                      GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
  normalSurfelTexture(Resolution::getInstance().width() * IndexMap::FACTORModel,
                      Resolution::getInstance().height() * IndexMap::FACTORModel,
                      GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
  timeSurfelTexture(Resolution::getInstance().width()* IndexMap::FACTORModel,
                    Resolution::getInstance().height()* IndexMap::FACTORModel,
                    GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT, false, true),
  predictHRBFProgram(loadProgramFromFile("empty.vert", "predict_hrbf.frag", "quad.geom")),
  predictHRBFRenderBuffer(Resolution::getInstance().width() * IndexMap::FACTORModel, Resolution::getInstance().height() * IndexMap::FACTORModel),
  imageTextureHRBF(Resolution::getInstance().width() * IndexMap::FACTORModel,
                   Resolution::getInstance().height() * IndexMap::FACTORModel,
                   GL_RGBA, GL_RGB, GL_UNSIGNED_BYTE, false, true),
  vertexTextureHRBF(Resolution::getInstance().width() * IndexMap::FACTORModel,
                Resolution::getInstance().height() * IndexMap::FACTORModel,
                GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
  normalTextureHRBF(Resolution::getInstance().width()* IndexMap::FACTORModel,
                Resolution::getInstance().height()* IndexMap::FACTORModel,
                GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
  curvk1TextureHRBF(Resolution::getInstance().width()* IndexMap::FACTORModel,
                Resolution::getInstance().height()* IndexMap::FACTORModel,
                GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
  curvk2TextureHRBF(Resolution::getInstance().width()* IndexMap::FACTORModel,
                Resolution::getInstance().height()* IndexMap::FACTORModel,
                GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
  timeTextureHRBF(Resolution::getInstance().width()* IndexMap::FACTORModel,
              Resolution::getInstance().height()* IndexMap::FACTORModel,
              GL_LUMINANCE16UI_EXT, GL_LUMINANCE_INTEGER_EXT,
              GL_UNSIGNED_SHORT, false, true),
  icpWeightHRBF(Resolution::getInstance().width()* IndexMap::FACTORModel,
               Resolution::getInstance().height()* IndexMap::FACTORModel,
               GL_LUMINANCE32F_ARB,
               GL_LUMINANCE,
               GL_FLOAT,
               false,
               true),
   oldRenderBufferHRBF(Resolution::getInstance().width(), Resolution::getInstance().height()),
   oldImageTextureHRBF(Resolution::getInstance().width(),
                    Resolution::getInstance().height(),
                    GL_RGBA,
                    GL_RGB,
                    GL_UNSIGNED_BYTE,
                    false,
                    true),
   oldVertexTextureHRBF(Resolution::getInstance().width(),
                     Resolution::getInstance().height(),
                     GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
   oldNormalTextureHRBF(Resolution::getInstance().width(),
                     Resolution::getInstance().height(),
                     GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
   oldTimeTextureHRBF(Resolution::getInstance().width(),
                   Resolution::getInstance().height(),
                   GL_LUMINANCE16UI_EXT,
                   GL_LUMINANCE_INTEGER_EXT,
                   GL_UNSIGNED_SHORT,
                   false,
                   true),
   oldcurvk1TextureHRBF(Resolution::getInstance().width(),
                     Resolution::getInstance().height(),
                     GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
   oldcurvk2TextureHRBF(Resolution::getInstance().width(),
                     Resolution::getInstance().height(),
                     GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, false, true),
   oldicpWeightHRBF(Resolution::getInstance().width(),
                    Resolution::getInstance().height(),
                    GL_LUMINANCE32F_ARB,
                    GL_LUMINANCE,
                    GL_FLOAT,
                    false,
                    true),
   KeyFrameIDMap(ACTIVE_KEYFRAME_DIMENSION, 1, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT),
   drawPredictedHRBFVertex(loadProgramFromFile("draw_prediction.vert", "draw_prediction.frag")),
   lActiveKFID(0)

{
   indexFrameBuffer.AttachColour(*indexTexture.texture);
   indexFrameBuffer.AttachColour(*vertConfTexture.texture);
   indexFrameBuffer.AttachColour(*colorTimeTexture.texture);
   indexFrameBuffer.AttachColour(*normalRadTexture.texture);
   indexFrameBuffer.AttachColour(*curv_map_maxTexture.texture);
   indexFrameBuffer.AttachColour(*curv_map_minTexture.texture);
   indexFrameBuffer.AttachDepth(indexRenderBuffer);

   drawFrameBuffer.AttachColour(*drawTexture.texture);
   drawFrameBuffer.AttachDepth(drawRenderBuffer);

   depthFrameBuffer.AttachColour(*depthTexture.texture);
   depthFrameBuffer.AttachDepth(depthRenderBuffer);

   surfelSplattingFrameBuffer.AttachColour(*imageSurfelTexture.texture);
   surfelSplattingFrameBuffer.AttachColour(*vertexSurfelTexture.texture);
   surfelSplattingFrameBuffer.AttachColour(*normalSurfelTexture.texture);
   surfelSplattingFrameBuffer.AttachColour(*timeSurfelTexture.texture);
   surfelSplattingFrameBuffer.AttachDepth(surfelSplattingRenderBuffer);

   predictHRBFFrameBuffer.AttachColour(*imageTextureHRBF.texture);
   predictHRBFFrameBuffer.AttachColour(*vertexTextureHRBF.texture);
   predictHRBFFrameBuffer.AttachColour(*normalTextureHRBF.texture);
   predictHRBFFrameBuffer.AttachColour(*curvk1TextureHRBF.texture);
   predictHRBFFrameBuffer.AttachColour(*curvk2TextureHRBF.texture);
   predictHRBFFrameBuffer.AttachColour(*timeTextureHRBF.texture);
   predictHRBFFrameBuffer.AttachColour(*icpWeightHRBF.texture);
   predictHRBFFrameBuffer.AttachDepth(predictHRBFRenderBuffer);

   oldFrameBufferHRBF.AttachColour(*oldImageTextureHRBF.texture);
   oldFrameBufferHRBF.AttachColour(*oldVertexTextureHRBF.texture);
   oldFrameBufferHRBF.AttachColour(*oldNormalTextureHRBF.texture);
   oldFrameBufferHRBF.AttachColour(*oldcurvk1TextureHRBF.texture);
   oldFrameBufferHRBF.AttachColour(*oldcurvk2TextureHRBF.texture);
   oldFrameBufferHRBF.AttachColour(*oldTimeTextureHRBF.texture);
   oldFrameBufferHRBF.AttachColour(*oldicpWeightHRBF.texture);
   oldFrameBufferHRBF.AttachDepth(oldRenderBufferHRBF);

   glGenBuffers(1, &vbo_predicted);
}

IndexMap::~IndexMap()
{

}

void IndexMap::predictIndices(const Eigen::Matrix4f & pose,
                              const int & time,
                              const int maxTime,
                              const std::pair<GLuint, GLuint> & model,
                              const float depthCutoff,
                              const int insertSubmap,
                              const int indexSubmap)
{
    indexFrameBuffer.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(0, 0, indexRenderBuffer.width, indexRenderBuffer.height);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    indexProgram->Bind();
    Eigen::Matrix4f t_inv = pose.inverse();        //
    Eigen::Vector4f cam(Intrinsics::getInstance().cx() * IndexMap::FACTOR,
                        Intrinsics::getInstance().cy() * IndexMap::FACTOR,
                        Intrinsics::getInstance().fx() * IndexMap::FACTOR,
                        Intrinsics::getInstance().fy() * IndexMap::FACTOR);
    indexProgram->setUniform(Uniform("t_inv", t_inv));
    indexProgram->setUniform(Uniform("cam", cam));
    indexProgram->setUniform(Uniform("maxDepth", depthCutoff));
    indexProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols() * IndexMap::FACTOR));
    indexProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows() * IndexMap::FACTOR));
    indexProgram->setUniform(Uniform("time", time));
    indexProgram->setUniform(Uniform("insertSubmap", insertSubmap));
    indexProgram->setUniform(Uniform("indexSubmap", indexSubmap));
    indexProgram->setUniform(Uniform("curvature_valid_threshold", GlobalStateParam::get().preprocessingCurvValidThreshold));

    std::vector<float> LAFK;
    LAFK.reserve(ACTIVE_KEYFRAME_DIMENSION);
    for(int i = 0; i < ACTIVE_KEYFRAME_DIMENSION; i++){
        LAFK.push_back(0);
    }
    for(int i = 0; i < lActiveKFID.size(); i++){
        LAFK[lActiveKFID[i]] = 1;
    }

    glBindTexture(GL_TEXTURE_2D, KeyFrameIDMap.texture->tid);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, LAFK.size(), 1, GL_LUMINANCE, GL_FLOAT, LAFK.data());
    indexProgram->setUniform(Uniform("KeyFrameIDMap", 0));
    indexProgram->setUniform(Uniform("KeyFrameIDDimen", static_cast<float>(ACTIVE_KEYFRAME_DIMENSION)));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, KeyFrameIDMap.texture->tid);

    //vbo
    glBindBuffer(GL_ARRAY_BUFFER, model.first);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 3));
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 4));
    glDrawTransformFeedback(GL_POINTS, model.second);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);

    indexFrameBuffer.Unbind();
    indexProgram->Unbind();

    glPopAttrib();
    glFinish();
}

void IndexMap::renderDepth(const float depthCutoff)
{
    drawFrameBuffer.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, drawRenderBuffer.width, drawRenderBuffer.height);

    glClearColor(0, 0, 0, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    drawDepthProgram->Bind();

    drawDepthProgram->setUniform(Uniform("maxDepth", depthCutoff));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, vertexTextureHRBF.texture->tid);

    drawDepthProgram->setUniform(Uniform("texVerts", 0));

    glDrawArrays(GL_POINTS, 0, 1);

    drawFrameBuffer.Unbind();

    drawDepthProgram->Unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    glPopAttrib();

    glFinish();
}

void IndexMap::renderHRBFPrediction(pangolin::OpenGlMatrix mvp, const Eigen::Matrix4f& pose)
{
    //prepare point to render
    Img<Eigen::Vector4f> verticesImag(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    Img<Eigen::Vector4f> normalImag(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    vertexTextureHRBF.texture->Download(verticesImag.data, GL_RGBA, GL_FLOAT);
    normalTextureHRBF.texture->Download(normalImag.data, GL_RGBA, GL_FLOAT);

    drawPredictedHRBFVertex->Bind();
    drawPredictedHRBFVertex->setUniform(Uniform("MVP", mvp));
    drawPredictedHRBFVertex->setUniform(Uniform("pose", pose));

    glBindBuffer(GL_ARRAY_BUFFER, vbo_predicted);
    glBufferData(GL_ARRAY_BUFFER, Resolution::getInstance().rows() * Resolution::getInstance().cols() * 4 * sizeof(float), verticesImag.data, GL_STREAM_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    glDrawArrays(GL_POINTS, 0, Resolution::getInstance().rows() * Resolution::getInstance().cols());
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    drawPredictedHRBFVertex->Unbind();
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_POINT_SPRITE);
}

void IndexMap::renderSurfelPrediction(pangolin::OpenGlMatrix mvp, const Eigen::Matrix4f& pose)
{
    //prepare point to render
    Img<Eigen::Vector4f> verticesImag(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    Img<Eigen::Vector4f> normalImag(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    vertexSurfelTexture.texture->Download(verticesImag.data, GL_RGBA, GL_FLOAT);
    normalSurfelTexture.texture->Download(normalImag.data, GL_RGBA, GL_FLOAT);

    drawPredictedHRBFVertex->Bind();
    drawPredictedHRBFVertex->setUniform(Uniform("MVP", mvp));
    drawPredictedHRBFVertex->setUniform(Uniform("pose", pose));

    glBindBuffer(GL_ARRAY_BUFFER, vbo_predicted);
    glBufferData(GL_ARRAY_BUFFER, Resolution::getInstance().rows() * Resolution::getInstance().cols() * 4 * sizeof(float), verticesImag.data, GL_STREAM_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    glDrawArrays(GL_POINTS, 0, Resolution::getInstance().rows() * Resolution::getInstance().cols());
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    drawPredictedHRBFVertex->Unbind();
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_POINT_SPRITE);
}

void IndexMap::combinedPredict(const Eigen::Matrix4f & pose,
                               const std::pair<GLuint, GLuint> & model,
                               const float depthCutoff,
                               const float confThreshold)
{
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    surfelSplattingFrameBuffer.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(0, 0, surfelSplattingRenderBuffer.width, surfelSplattingRenderBuffer.height);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    surfelSplattingProgram->Bind();
    Eigen::Matrix4f t_inv = pose.inverse();
    Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
                      Intrinsics::getInstance().cy(),
                      Intrinsics::getInstance().fx(),
                      Intrinsics::getInstance().fy());
    surfelSplattingProgram->setUniform(Uniform("t_inv", t_inv));
    surfelSplattingProgram->setUniform(Uniform("cam", cam));
    surfelSplattingProgram->setUniform(Uniform("maxDepth", depthCutoff));
    surfelSplattingProgram->setUniform(Uniform("confThreshold", confThreshold));
    surfelSplattingProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
    surfelSplattingProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));
    surfelSplattingProgram->setUniform(Uniform("preprocessingInitRadiusMultiplier", GlobalStateParam::get().preprocessingInitRadiusMultiplier));

    glBindBuffer(GL_ARRAY_BUFFER, model.first);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 3));
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 4));

    glDrawTransformFeedback(GL_POINTS, model.second);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    surfelSplattingFrameBuffer.Bind();
    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_POINT_SPRITE);

    glPopAttrib();

    glFinish();
}
void IndexMap::predictHRBF(IndexMap::Prediction predictionType)
{

    if(predictionType == IndexMap::ACTIVE)
    {
       predictHRBFFrameBuffer.Bind();
    }
    else if(predictionType == IndexMap::INACTIVE)
    {
       oldFrameBufferHRBF.Bind();
    }
    else
    {
       assert(false);
    }

    glPushAttrib(GL_VIEWPORT_BIT);

    if(predictionType == IndexMap::ACTIVE)
    {
       glViewport(0, 0, predictHRBFRenderBuffer.width, predictHRBFRenderBuffer.height);
    }
    else if(predictionType == IndexMap::INACTIVE)
    {
       glViewport(0, 0, oldRenderBufferHRBF.width, oldRenderBufferHRBF.height);
    }
    else
    {
       assert(false);
    }

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    predictHRBFProgram->Bind();

    Eigen::Vector4f cam(Intrinsics::getInstance().cx() * IndexMap::FACTOR,
                        Intrinsics::getInstance().cy() * IndexMap::FACTOR,
                        1.0 / Intrinsics::getInstance().fx()*IndexMap::FACTOR,
                        1.0 / Intrinsics::getInstance().fy()*IndexMap::FACTOR);

    predictHRBFProgram->setUniform(Uniform("cam", cam));
    predictHRBFProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()*IndexMap::FACTOR));
    predictHRBFProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()*IndexMap::FACTOR));
    predictHRBFProgram->setUniform(Uniform("scale", (float)IndexMap::FACTOR));
    predictHRBFProgram->setUniform(Uniform("predict_minimum_neighbors", GlobalStateParam::get().preictionMinNeighbors));
    predictHRBFProgram->setUniform(Uniform("predict_maximum_neighbors", GlobalStateParam::get().preictionMaxNeighbors));
    predictHRBFProgram->setUniform(Uniform("winMultiply", GlobalStateParam::get().preictionWindowMultiplier));
    predictHRBFProgram->setUniform(Uniform("predict_confidence_threshold", GlobalStateParam::get().preictionConfThreshold));

    predictHRBFProgram->setUniform(Uniform("indexSampler", 0));
    predictHRBFProgram->setUniform(Uniform("vertConfSampler", 1));
    predictHRBFProgram->setUniform(Uniform("colorTimeSampler", 2));
    predictHRBFProgram->setUniform(Uniform("normRadSampler", 3));
    predictHRBFProgram->setUniform(Uniform("curv_maxSampler", 4));
    predictHRBFProgram->setUniform(Uniform("curv_minSampler", 5));

    predictHRBFProgram->setUniform(Uniform("icp_weight_lambda", GlobalStateParam::get().registrationICPCurvWeightImpactControl));


    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, indexTexture.texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, vertConfTexture.texture->tid);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, colorTimeTexture.texture->tid);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, normalRadTexture.texture->tid);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, curv_map_maxTexture.texture->tid);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, curv_map_minTexture.texture->tid);

//    glActiveTexture(GL_TEXTURE6);
//    glBindTexture(GL_TEXTURE_2D, vertexSurfelTexture.texture->tid);

//    glActiveTexture(GL_TEXTURE7);
//    glBindTexture(GL_TEXTURE_2D, normalSurfelTexture.texture->tid);

    glDrawArrays(GL_POINTS, 0, 1);

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);

    if(predictionType == IndexMap::ACTIVE)
    {
        predictHRBFFrameBuffer.Unbind();
    }
    else if(predictionType == IndexMap::INACTIVE)
    {
        oldFrameBufferHRBF.Unbind();
    }
    else
    {
        assert(false);
    }

    predictHRBFProgram->Unbind();
    glPopAttrib();
    glFinish();
}

void IndexMap::synthesizeDepth(const Eigen::Matrix4f & pose,
                               const std::pair<GLuint, GLuint> & model,
                               const float depthCutoff,
                               const float confThreshold,
                               const int time,
                               const int maxTime,
                               const int timeDelta)
{
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    depthFrameBuffer.Bind();

    glPushAttrib(GL_VIEWPORT_BIT);

    glViewport(0, 0, depthRenderBuffer.width, depthRenderBuffer.height);

    glClearColor(0, 0, 0, 0);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    depthProgram->Bind();

    Eigen::Matrix4f t_inv = pose.inverse();

    Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
                  Intrinsics::getInstance().cy(),
                  Intrinsics::getInstance().fx(),
                  Intrinsics::getInstance().fy());

    depthProgram->setUniform(Uniform("t_inv", t_inv));
    depthProgram->setUniform(Uniform("cam", cam));
    depthProgram->setUniform(Uniform("maxDepth", depthCutoff));
    depthProgram->setUniform(Uniform("confThreshold", confThreshold));
    depthProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
    depthProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));
    depthProgram->setUniform(Uniform("time", time));
    depthProgram->setUniform(Uniform("maxTime", maxTime));
    depthProgram->setUniform(Uniform("timeDelta", timeDelta));

    glBindBuffer(GL_ARRAY_BUFFER, model.first);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

    glDrawTransformFeedback(GL_POINTS, model.second);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    depthFrameBuffer.Unbind();

    depthProgram->Unbind();

    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_POINT_SPRITE);

    glPopAttrib();

    glFinish();
}

void IndexMap::downloadTexture(const Eigen::Matrix4f& pose, int frameID){
    Img<Eigen::Matrix<unsigned char, 4, 1>> image(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    Img<Eigen::Vector4f> verticesImag(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    Img<Eigen::Vector4f> verticesImag_surfel(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    Img<Eigen::Vector4f> normalImag(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    Img<Eigen::Vector4f> normalImag_surfel(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    Img<Eigen::Vector4f> curvk1Imag(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    Img<Eigen::Vector4f> curvk2Imag(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    Img<float> icpweight(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    imageTextureHRBF.texture->Download(image.data, GL_RGBA, GL_UNSIGNED_BYTE);
    vertexTextureHRBF.texture->Download(verticesImag.data, GL_RGBA, GL_FLOAT);
    vertexSurfelTexture.texture->Download(verticesImag_surfel.data, GL_RGBA, GL_FLOAT);
    normalTextureHRBF.texture->Download(normalImag.data, GL_RGBA, GL_FLOAT);
    normalSurfelTexture.texture->Download(normalImag_surfel.data, GL_RGBA, GL_FLOAT);
    curvk1TextureHRBF.texture->Download(curvk1Imag.data, GL_RGBA, GL_FLOAT);
    curvk2TextureHRBF.texture->Download(curvk2Imag.data, GL_RGBA, GL_FLOAT);
    icpWeightHRBF.texture->Download(icpweight.data, GL_LUMINANCE, GL_FLOAT);

    cv::Mat imageCV;
    imageCV.create(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_8UC3);
    for (int i = 0; i < imageCV.rows; ++i){
        for (int j = 0; j < imageCV.cols; ++j){
            cv::Vec3b& bgra = imageCV.at<cv::Vec3b>(i, j);
            if(image.at<Eigen::Matrix<unsigned char, 4, 1>>(i,j)(0) == 0 && image.at<Eigen::Matrix<unsigned char, 4, 1>>(i,j)(1) == 0 && image.at<Eigen::Matrix<unsigned char, 4, 1>>(i,j)(2) == 0)
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
    cv::imwrite("indexHRBF_colorMap.png", imageCV, compression_params);
    std::cout << "indexHRBF_colorMap saved successfully" << std::endl;

    //show and down load texture
    //cv::namedWindow("HRBFNormalTexture", cv::WINDOW_AUTOSIZE);
    cv::Mat normalImagCV;
    normalImagCV.create(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_8UC3);
    //create normalImagCV;
    //CV_ASSERT(normalImagCV.channels() == 4);
    for (int i = 0; i < normalImagCV.rows; ++i){
        for (int j = 0; j < normalImagCV.cols; ++j){
            cv::Vec3b& bgra = normalImagCV.at<cv::Vec3b>(i, j);
            if(normalImag.at<Eigen::Vector4f>(i,j)(0) == 0 && normalImag.at<Eigen::Vector4f>(i,j)(1) == 0 && normalImag.at<Eigen::Vector4f>(i,j)(2) == 0)
            {
                bgra[2] = static_cast<unsigned char>(UCHAR_MAX);
                bgra[1] = static_cast<unsigned char>(UCHAR_MAX);
                bgra[0] = static_cast<unsigned char>(UCHAR_MAX);
            }else{
                bgra[2] = static_cast<unsigned char>(UCHAR_MAX * (normalImag.at<Eigen::Vector4f>(i,j)(0) > 0 ? normalImag.at<Eigen::Vector4f>(i,j)(0) : 0)); // Blue
                bgra[1] = static_cast<unsigned char>(UCHAR_MAX * (normalImag.at<Eigen::Vector4f>(i,j)(1) > 0 ? normalImag.at<Eigen::Vector4f>(i,j)(1) : 0)); // Green
                bgra[0] = static_cast<unsigned char>(UCHAR_MAX * (normalImag.at<Eigen::Vector4f>(i,j)(2) > 0 ? normalImag.at<Eigen::Vector4f>(i,j)(2) : 0)); // Red
            }

        }
    }
//    std::vector<int> compression_params;
//    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
//    compression_params.push_back(0);
    cv::imwrite("HRBF_normal.png", normalImagCV, compression_params);
    std::cout << "normal texture saved successfully" << std::endl;

    //output txt
    std::ofstream textureD;
    textureD.open("prediction_hrbf_" + std::to_string(frameID) + ".txt");

    for(int i = 0; i < verticesImag_surfel.rows; i++){
        for(int j = 0; j < verticesImag_surfel.cols; j++)
        {
            if(curvk1Imag.at<Eigen::Vector4f>(i, j)(3) == 0 || std::isnan(curvk1Imag.at<Eigen::Vector4f>(i, j)(3)))
                continue;
            float local_x = verticesImag.at<Eigen::Vector4f>(i, j)(0);
            float local_y = verticesImag.at<Eigen::Vector4f>(i, j)(1);
            float local_z = verticesImag.at<Eigen::Vector4f>(i, j)(2);
            Eigen::Vector4f pos_local(local_x, local_y, local_z, 1.0);
            Eigen::Vector4f pos_global = pose * pos_local;

            float local_nx = normalImag.at<Eigen::Vector4f>(i, j)(0);
            float local_ny = normalImag.at<Eigen::Vector4f>(i, j)(1);
            float local_nz = normalImag.at<Eigen::Vector4f>(i, j)(2);

            Eigen::Matrix3f Rot = pose.topLeftCorner(3, 3);
            Eigen::Vector3f normal_local(local_nx, local_ny, local_nz);
            Eigen::Vector3f normal_global = Rot * normal_local;

            //fitting error
            textureD << pos_global(0)<<" "<<pos_global(1)<<" "<<pos_global(2)<<" "
                     << normal_global(0)<<" "<<normal_global(1)<<" "<<normal_global(2)<<" "
                     <<" "<<icpweight.at<float>(i, j)<<" "
                     //<<" "<<normalImag.at<Eigen::Vector4f>(i, j)(3)
                     << curvk1Imag.at<Eigen::Vector4f>(i, j)(3) <<" "<< curvk2Imag.at<Eigen::Vector4f>(i, j)(3) <<"\n";
                     /*curvk1Imag.at<Eigen::Vector4f>(i, j)(0)<<" "<<curvk1Imag.at<Eigen::Vector4f>(i, j)(1)<<" "<<curvk1Imag.at<Eigen::Vector4f>(i, j)(2)<<" "<<curvk1Imag.at<Eigen::Vector4f>(i, j)(3)<<" "
                     << curvk2Imag.at<Eigen::Vector4f>(i, j)(0)<<" "<<curvk2Imag.at<Eigen::Vector4f>(i, j)(1)<<" "<<curvk2Imag.at<Eigen::Vector4f>(i, j)(2)<<" "<<curvk2Imag.at<Eigen::Vector4f>(i, j)(3)*/
                     //<<"\n";
        }
    }
    textureD.close();

    textureD.open("prediction_surfel_" + std::to_string(frameID) + ".txt");
    for(int i = 0; i < verticesImag.rows; i++){
        for(int j = 0; j < verticesImag.cols; j++)
        {
            if(curvk1Imag.at<Eigen::Vector4f>(i, j)(3) == 0 || std::isnan(curvk1Imag.at<Eigen::Vector4f>(i, j)(3)))
                continue;
            float local_x = verticesImag_surfel.at<Eigen::Vector4f>(i, j)(0);
            float local_y = verticesImag_surfel.at<Eigen::Vector4f>(i, j)(1);
            float local_z = verticesImag_surfel.at<Eigen::Vector4f>(i, j)(2);
            Eigen::Vector4f pos_local(local_x, local_y, local_z, 1.0);
            Eigen::Vector4f pos_global = pose * pos_local;

            float local_nx = normalImag_surfel.at<Eigen::Vector4f>(i, j)(0);
            float local_ny = normalImag_surfel.at<Eigen::Vector4f>(i, j)(1);
            float local_nz = normalImag_surfel.at<Eigen::Vector4f>(i, j)(2);

            Eigen::Matrix3f Rot = pose.topLeftCorner(3, 3);
            Eigen::Vector3f normal_local(local_nx, local_ny, local_nz);
            Eigen::Vector3f normal_global = Rot * normal_local;

            //fitting error
            textureD << pos_global(0)<<" "<<pos_global(1)<<" "<<pos_global(2)<<" "
                     << normal_global(0)<<" "<<normal_global(1)<<" "<<normal_global(2)<<" "
                     //<<" "<<icpweight.at<float>(i, j)<<" "
                     //<<" "<<normalImag_surfel.at<Eigen::Vector4f>(i, j)(3)
                     << curvk1Imag.at<Eigen::Vector4f>(i, j)(3) <<" "<< curvk2Imag.at<Eigen::Vector4f>(i, j)(3) <<"\n";
                     /*curvk1Imag.at<Eigen::Vector4f>(i, j)(0)<<" "<<curvk1Imag.at<Eigen::Vector4f>(i, j)(1)<<" "<<curvk1Imag.at<Eigen::Vector4f>(i, j)(2)<<" "<<curvk1Imag.at<Eigen::Vector4f>(i, j)(3)<<" "
                     << curvk2Imag.at<Eigen::Vector4f>(i, j)(0)<<" "<<curvk2Imag.at<Eigen::Vector4f>(i, j)(1)<<" "<<curvk2Imag.at<Eigen::Vector4f>(i, j)(2)<<" "<<curvk2Imag.at<Eigen::Vector4f>(i, j)(3)*/
                     //<<"\n";
        }
    }
    textureD.close();
    std::cout <<"texture saved successfully"<< std::endl;
}
