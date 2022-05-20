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

#include "GlobalModel.h"

const int GlobalModel::TEXTURE_DIMENSION = 4596;       //3072;
const int GlobalModel::MAX_VERTICES = GlobalModel::TEXTURE_DIMENSION * GlobalModel::TEXTURE_DIMENSION;
const int GlobalModel::NODE_TEXTURE_DIMENSION = 16384;
const int GlobalModel::MAX_NODES = GlobalModel::NODE_TEXTURE_DIMENSION / 16;  //16 floats per node
const int GlobalModel::MAX_SUBMAPS = 3600;
const int GlobalModel::DELTA_TRANS_DIMENSION = 19200;
const int GlobalModel::ACTIVE_KEYFRAME_DIMENSION = 19200;


GlobalModel::GlobalModel()
 : target(0),
   renderSource(1),
   bufferSize(MAX_VERTICES * Vertex::SIZE),
   count(0),
   initTProgram(loadProgramGeomFromFile("init_unstableTex.vert", "init_unstableTex.geom")),
   drawProgram(loadProgramFromFile("draw_feedback.vert", "draw_feedback.frag")),
   drawSurfelProgram(loadProgramFromFile("draw_global_surface.vert", "draw_global_surface.frag", "draw_global_surface.geom")),
   dataProgram(loadProgramFromFile("data.vert", "data.frag", "data.geom")),
   updateProgram(loadProgramFromFile("update.vert")),
   unstableProgram(loadProgramGeomFromFile("copy_unstable.vert", "copy_unstable.geom")),
   updateDeltaTransProgram(loadProgramGeomFromFile("update_delta_trans.vert", "update_delta_trans.geom")),
   renderBuffer(TEXTURE_DIMENSION, TEXTURE_DIMENSION),
   updateMapVertsConfs(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
   updateMapColorsTime(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
   updateMapNormsRadii(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
   updateMapCurvatureK1(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
   updateMapCurvatureK2(TEXTURE_DIMENSION, TEXTURE_DIMENSION, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT),
   DeltaTransVec(DELTA_TRANS_DIMENSION, 1, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT),
   KeyFrameIDMap(ACTIVE_KEYFRAME_DIMENSION, 1, GL_LUMINANCE32F_ARB, GL_LUMINANCE, GL_FLOAT),
   lActiveKFID(0)
{
    vbos = new std::pair<GLuint, GLuint>[2];

    float * vertices = new float[bufferSize];

    // set the first bufferSize bytes of the block of memory by vertices to 0, erase and fill in;
    memset(&vertices[0], 0, bufferSize);

    //generate transform feedback and GlobalModel buffer
    glGenTransformFeedbacks(1, &vbos[0].second);
    glGenBuffers(1, &vbos[0].first);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[0].first);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glGenTransformFeedbacks(1, &vbos[1].second);
    glGenBuffers(1, &vbos[1].first);
    glBindBuffer(GL_ARRAY_BUFFER, vbos[1].first);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete [] vertices;

    vertices = new float[Resolution::getInstance().numPixels() * Vertex::SIZE];

    memset(&vertices[0], 0, Resolution::getInstance().numPixels() * Vertex::SIZE);

    //record newly added unstable points, with no global points found
    glGenTransformFeedbacks(1, &newUnstableFid);
    glGenBuffers(1, &newUnstableVbo);
    glBindBuffer(GL_ARRAY_BUFFER, newUnstableVbo);
    glBufferData(GL_ARRAY_BUFFER, Resolution::getInstance().numPixels() * Vertex::SIZE, &vertices[0], GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    delete [] vertices;

    std::vector<Eigen::Vector2f> uv;

    for(int i = 0; i < Resolution::getInstance().width(); i++)
    {
        for(int j = 0; j < Resolution::getInstance().height(); j++)
        {
            uv.push_back(Eigen::Vector2f(((float)i / (float)Resolution::getInstance().width()) + 1.0 / (2 * (float)Resolution::getInstance().width()),
                                   ((float)j / (float)Resolution::getInstance().height()) + 1.0 / (2 * (float)Resolution::getInstance().height())));
        }
    }

    uvSize = uv.size();     //also Resolution::getInstance().numPixels()

    glGenBuffers(1, &uvo);
    glBindBuffer(GL_ARRAY_BUFFER, uvo);
    glBufferData(GL_ARRAY_BUFFER, uvSize * sizeof(Eigen::Vector2f), &uv[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    frameBuffer.AttachColour(*updateMapVertsConfs.texture);
    frameBuffer.AttachColour(*updateMapColorsTime.texture);
    frameBuffer.AttachColour(*updateMapNormsRadii.texture);
    frameBuffer.AttachColour(*updateMapCurvatureK1.texture);
    frameBuffer.AttachColour(*updateMapCurvatureK2.texture);
    frameBuffer.AttachDepth(renderBuffer);

    updateProgram->Bind();

    int locUpdate[5] =
    {
        glGetVaryingLocationNV(updateProgram->programId(), "vPosition0"),
        glGetVaryingLocationNV(updateProgram->programId(), "vColor0"),
        glGetVaryingLocationNV(updateProgram->programId(), "vNormRad0"),
        glGetVaryingLocationNV(updateProgram->programId(), "curv_map_max0"),
        glGetVaryingLocationNV(updateProgram->programId(), "curv_map_min0"),
    };

    glTransformFeedbackVaryingsNV(updateProgram->programId(), 5, locUpdate, GL_INTERLEAVED_ATTRIBS);

    updateProgram->Unbind();

    dataProgram->Bind(); //set transform feedback attributes, before link the program
    int dataUpdate[5] =
    {
        glGetVaryingLocationNV(dataProgram->programId(), "vPosition0"),
        glGetVaryingLocationNV(dataProgram->programId(), "vColor0"),
        glGetVaryingLocationNV(dataProgram->programId(), "vNormRad0"),
        glGetVaryingLocationNV(dataProgram->programId(), "curv_map_max0"),
        glGetVaryingLocationNV(dataProgram->programId(), "curv_map_min0"),
    };
    glTransformFeedbackVaryingsNV(dataProgram->programId(), 5, dataUpdate, GL_INTERLEAVED_ATTRIBS);
    dataProgram->Unbind();

    unstableProgram->Bind();
    int unstableUpdate[5] =
    {
        glGetVaryingLocationNV(unstableProgram->programId(), "vPosition0"),
        glGetVaryingLocationNV(unstableProgram->programId(), "vColor0"),
        glGetVaryingLocationNV(unstableProgram->programId(), "vNormRad0"),
        glGetVaryingLocationNV(unstableProgram->programId(), "curv_map_max0"),
        glGetVaryingLocationNV(unstableProgram->programId(), "curv_map_min0"),
    };
    glTransformFeedbackVaryingsNV(unstableProgram->programId(), 5, unstableUpdate, GL_INTERLEAVED_ATTRIBS);
    unstableProgram->Unbind();

    //record the output of the geometry shader
    updateDeltaTransProgram->Bind();
    int modelposeUpdate[5] =
    {
        glGetVaryingLocationNV(updateDeltaTransProgram->programId(), "vPosition0"),
        glGetVaryingLocationNV(updateDeltaTransProgram->programId(), "vColor0"),
        glGetVaryingLocationNV(updateDeltaTransProgram->programId(), "vNormRad0"),
        glGetVaryingLocationNV(updateDeltaTransProgram->programId(), "curv_map_max0"),
        glGetVaryingLocationNV(updateDeltaTransProgram->programId(), "curv_map_min0"),
    };
    glTransformFeedbackVaryingsNV(updateDeltaTransProgram->programId(), 5, modelposeUpdate, GL_INTERLEAVED_ATTRIBS);
    updateDeltaTransProgram->Unbind();

    initTProgram->Bind();
    int locinitT[5] =
    {
        glGetVaryingLocationNV(initTProgram->programId(), "vPosition0"),
        glGetVaryingLocationNV(initTProgram->programId(), "vColor0"),
        glGetVaryingLocationNV(initTProgram->programId(), "vNormRad0"),
        glGetVaryingLocationNV(initTProgram->programId(), "curv_map_max0"),
        glGetVaryingLocationNV(initTProgram->programId(), "curv_map_min0"),
    };
    glTransformFeedbackVaryingsNV(initTProgram->programId(), 5, locinitT, GL_INTERLEAVED_ATTRIBS);
    glGenQueries(1, &countQuery); //generate countQuery, just generate query

    //Empty both transform feedbacks
    glEnable(GL_RASTERIZER_DISCARD);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[0].second);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[0].first);  //bind a buffer object to an indexed buffer target
    glBeginTransformFeedback(GL_POINTS); //start tarnsform feedback operation
    glDrawArrays(GL_POINTS, 0, 0);
    glEndTransformFeedback();

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[1].second);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[1].first);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 0);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    initTProgram->Unbind();
}

GlobalModel::~GlobalModel()
{
    glDeleteBuffers(1, &vbos[0].first);
    glDeleteTransformFeedbacks(1, &vbos[0].second);

    glDeleteBuffers(1, &vbos[1].first);
    glDeleteTransformFeedbacks(1, &vbos[1].second);

    glDeleteQueries(1, &countQuery);

    glDeleteBuffers(1, &uvo);

    glDeleteTransformFeedbacks(1, &newUnstableFid);
    glDeleteBuffers(1, &newUnstableVbo);

    delete [] vbos;
}

void GlobalModel::initialise(GPUTexture * vertexMap,
                             GPUTexture * normalMap,
                             GPUTexture * colorMap,
                             GPUTexture * curv1Map,
                             GPUTexture * curv2Map,
                             GPUTexture * gradientMagMap,
                             const Eigen::Matrix4f& init_pose)
{
    initTProgram->Bind();

    initTProgram->setUniform(Uniform("vertexSampler", 0));
    initTProgram->setUniform(Uniform("normalSampler", 1));
    initTProgram->setUniform(Uniform("colorSampler", 2));
    initTProgram->setUniform(Uniform("curv1Sampler", 3));
    initTProgram->setUniform(Uniform("curv2Sampler", 4));
    initTProgram->setUniform(Uniform("gradientMagSampler", 5));

    initTProgram->setUniform(Uniform("cols", static_cast<float>(Resolution::getInstance().cols())));
    initTProgram->setUniform(Uniform("rows", static_cast<float>(Resolution::getInstance().rows())));
    initTProgram->setUniform(Uniform("cam",  Eigen::Vector4f(Intrinsics::getInstance().cx(),
                                       Intrinsics::getInstance().cy(),
                                       1.0 / Intrinsics::getInstance().fx(),
                                       1.0 / Intrinsics::getInstance().fy())));
    initTProgram->setUniform(Uniform("curvature_valid_threshold", static_cast<float>(GlobalStateParam::get().preprocessingCurvValidThreshold)));
    initTProgram->setUniform(Uniform("useConfidenceEvaluation", static_cast<float>(GlobalStateParam::get().preprocessingUseConfEval)));
    initTProgram->setUniform(Uniform("epsilon", GlobalStateParam::get().preprocessingConfEvalEpsilon));

    initTProgram->setUniform(Uniform("init_pose", init_pose));
    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, uvo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glEnable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[target].second);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[target].first);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, vertexMap->texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, normalMap->texture->tid);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, colorMap->texture->tid);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, curv1Map->texture->tid);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, curv2Map->texture->tid);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, gradientMagMap->texture->tid);
    
    glBeginTransformFeedback(GL_POINTS);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

    glDrawArrays(GL_POINTS, 0, uvSize);

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);

    glEndTransformFeedback();
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);
    glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);
    glDisable(GL_RASTERIZER_DISCARD);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    initTProgram->Unbind();
    glFinish();

}

void GlobalModel::renderPointCloud(pangolin::OpenGlMatrix mvp,
                                   const float threshold,
                                   const bool drawUnstable,
                                   const bool drawNormals,
                                   const bool drawColors,
                                   const bool drawPoints,
                                   const bool drawWindow,
                                   const bool drawTimes,
                                   const int time,
                                   const int timeDelta)
{    
    std::shared_ptr<Shader> program = drawPoints ? drawProgram : drawSurfelProgram;

    program->Bind();

    program->setUniform(Uniform("MVP", mvp));

    program->setUniform(Uniform("threshold", threshold));

    program->setUniform(Uniform("colorType", (drawNormals ? 1 : drawColors ? 2 : drawTimes ? 3 : 0)));

    program->setUniform(Uniform("unstable", drawUnstable));

    program->setUniform(Uniform("drawWindow", drawWindow));

    program->setUniform(Uniform("time", time));

    program->setUniform(Uniform("timeDelta", timeDelta));

    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();

    //This is for the vertex shader
    program->setUniform(Uniform("pose", pose));

    glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

    //enable point size;
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    glDrawTransformFeedback(GL_POINTS, vbos[target].second);   //this is used for rendering

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    program->Unbind();
    glFinish();
}

const std::pair<GLuint, GLuint> & GlobalModel::model()
{
    return vbos[target];
}


void GlobalModel::fuse(const Eigen::Matrix4f & pose,
                       const int & time,
                       GPUTexture * rgb,
                       GPUTexture * depthRaw,
                       GPUTexture * depthFiltered,
                       GPUTexture * curv_map_max,
                       GPUTexture * curv_map_min,
                       GPUTexture * confidence,
                       GPUTexture * indexMap,
                       GPUTexture * vertConfMap,
                       GPUTexture * colorTimeMap,
                       GPUTexture * normRadMap,
                       const float depthCutoff,
                       const float confThreshold,
                       const float weighting,
                       const bool insertSubmap,
                       const float indexSubmap)
{
    //This first part does data association and computes the vertex to merge with, storing
    //in an array that sets which vertices to update by index
    frameBuffer.Bind();
    glPushAttrib(GL_VIEWPORT_BIT);
    glViewport(0, 0, renderBuffer.width, renderBuffer.height);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    dataProgram->Bind();
    dataProgram->setUniform(Uniform("cSampler", 0));
    dataProgram->setUniform(Uniform("drSampler", 1));
    dataProgram->setUniform(Uniform("drfSampler", 2));
    dataProgram->setUniform(Uniform("new_curv1_Samp", 3));
    dataProgram->setUniform(Uniform("new_curv2_Samp", 4));
    dataProgram->setUniform(Uniform("new_conf_Samp", 5));

    dataProgram->setUniform(Uniform("indexSampler", 6));
    dataProgram->setUniform(Uniform("vertConfSampler", 7));
    dataProgram->setUniform(Uniform("colorTimeSampler", 8));
    dataProgram->setUniform(Uniform("normRadSampler", 9));
    dataProgram->setUniform(Uniform("time", static_cast<float>(time)));
    dataProgram->setUniform(Uniform("weighting", weighting));
    dataProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(),
                                                     Intrinsics::getInstance().cy(),
                                                     1.0 / Intrinsics::getInstance().fx(),
                                                     1.0 / Intrinsics::getInstance().fy())));
    dataProgram->setUniform(Uniform("cols", static_cast<float>(Resolution::getInstance().cols())));
    dataProgram->setUniform(Uniform("rows", static_cast<float>(Resolution::getInstance().rows())));
    dataProgram->setUniform(Uniform("scale", static_cast<float>(IndexMap::FACTOR)));
    dataProgram->setUniform(Uniform("texDim", static_cast<float>(TEXTURE_DIMENSION)));
    dataProgram->setUniform(Uniform("pose", pose));
    dataProgram->setUniform(Uniform("maxDepth", depthCutoff));
    dataProgram->setUniform(Uniform("indexSubmap",  indexSubmap));
    dataProgram->setUniform(Uniform("insertSubmap", insertSubmap));
    dataProgram->setUniform(Uniform("RadiusMultiplier", GlobalStateParam::get().preprocessingInitRadiusMultiplier));
    dataProgram->setUniform(Uniform("PCAforNormalEstimation", GlobalStateParam::get().preprocessingNormalEstimationPCA));

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, uvo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, newUnstableFid);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, newUnstableVbo);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, rgb->texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, depthRaw->texture->tid);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, depthFiltered->texture->tid);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, curv_map_max->texture->tid);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, curv_map_min->texture->tid);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, confidence->texture->tid);

    glActiveTexture(GL_TEXTURE6);
    glBindTexture(GL_TEXTURE_2D, indexMap->texture->tid);

    glActiveTexture(GL_TEXTURE7);
    glBindTexture(GL_TEXTURE_2D, vertConfMap->texture->tid);

    glActiveTexture(GL_TEXTURE8);
    glBindTexture(GL_TEXTURE_2D, colorTimeMap->texture->tid);

    glActiveTexture(GL_TEXTURE9);
    glBindTexture(GL_TEXTURE_2D, normRadMap->texture->tid);

    glBeginTransformFeedback(GL_POINTS);

    glDrawArrays(GL_POINTS, 0, uvSize);

    glEndTransformFeedback();

    frameBuffer.Unbind();

    glBindTexture(GL_TEXTURE_2D, 0);

    glActiveTexture(GL_TEXTURE0);

    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    dataProgram->Unbind();

    glPopAttrib();
    glFinish();
  
    //Next we update the vertices at the indexes stored in the update textures
    //Using a transform feedback conditional on a texture sample
    updateProgram->Bind();

    updateProgram->setUniform(Uniform("vertSamp", 0));
    updateProgram->setUniform(Uniform("colorSamp", 1));
    updateProgram->setUniform(Uniform("normSamp", 2));
    updateProgram->setUniform(Uniform("curv_map_maxSamp", 3));
    updateProgram->setUniform(Uniform("curv_map_minSamp", 4));

    updateProgram->setUniform(Uniform("texDim", static_cast<float>(TEXTURE_DIMENSION)));

    updateProgram->setUniform(Uniform("time", time));
    updateProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(),Intrinsics::getInstance().cy(),
                                                     1.0 / Intrinsics::getInstance().fx(), 1.0 / Intrinsics::getInstance().fy())));
    updateProgram->setUniform(Uniform("pose", pose));
    Eigen::Matrix4f pose_inv = pose.inverse();
    updateProgram->setUniform(Uniform("pose_inv", pose_inv));
    updateProgram->setUniform(Uniform("cols", static_cast<float>(Resolution::getInstance().cols())));
    updateProgram->setUniform(Uniform("rows", static_cast<float>(Resolution::getInstance().rows())));

    glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

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

    glEnable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].second);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].first);

    glBeginTransformFeedback(GL_POINTS);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, updateMapVertsConfs.texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, updateMapColorsTime.texture->tid);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, updateMapNormsRadii.texture->tid);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, updateMapCurvatureK1.texture->tid);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, updateMapCurvatureK2.texture->tid);

    glDrawTransformFeedback(GL_POINTS, vbos[target].second);

    glEndTransformFeedback();

    glDisable(GL_RASTERIZER_DISCARD);

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    updateProgram->Unbind();
    std::swap(target, renderSource);
    glFinish();
}

void GlobalModel::clean(const Eigen::Matrix4f & pose,
                        const int & time,
                        GPUTexture * indexMap,
                        GPUTexture * vertConfMap,
                        GPUTexture * colorTimeMap,
                        GPUTexture * normRadMap,
                        GPUTexture * depthMap,
                        const float confThreshold,
                        const float maxDepth)
{
    //add new vertices and clean the long-term unstable vertices
    unstableProgram->Bind();
    unstableProgram->setUniform(Uniform("time", time));
    unstableProgram->setUniform(Uniform("confThreshold", confThreshold));
    unstableProgram->setUniform(Uniform("scale", (float)IndexMap::FACTOR));
    unstableProgram->setUniform(Uniform("indexSampler", 0));
    unstableProgram->setUniform(Uniform("vertConfSampler", 1));
    unstableProgram->setUniform(Uniform("colorTimeSampler", 2));
    unstableProgram->setUniform(Uniform("normRadSampler", 3));
    unstableProgram->setUniform(Uniform("depthSampler", 4));
    unstableProgram->setUniform(Uniform("maxDepth", maxDepth));
    unstableProgram->setUniform(Uniform("window_multiplier", GlobalStateParam::get().fusionCleanWindowMultiplier));
    unstableProgram->setUniform(Uniform("curvature_valid_threshold", GlobalStateParam::get().preprocessingCurvValidThreshold));

    Eigen::Matrix4f t_inv = pose.inverse();
    unstableProgram->setUniform(Uniform("t_inv", t_inv));
    unstableProgram->setUniform(Uniform("pose", pose));

    unstableProgram->setUniform(Uniform("cam", Eigen::Vector4f(Intrinsics::getInstance().cx(),
                                                         Intrinsics::getInstance().cy(),
                                                         Intrinsics::getInstance().fx(),
                                                         Intrinsics::getInstance().fy())));
    unstableProgram->setUniform(Uniform("cols", (float)Resolution::getInstance().cols()));
    unstableProgram->setUniform(Uniform("rows", (float)Resolution::getInstance().rows()));

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
    unstableProgram->setUniform(Uniform("KeyFrameIDMap", 5));
    unstableProgram->setUniform(Uniform("KeyFrameIDDimen", static_cast<float>(ACTIVE_KEYFRAME_DIMENSION)));

    glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

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

    glEnable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].second);

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].first);

    glBeginTransformFeedback(GL_POINTS);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, indexMap->texture->tid);

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, vertConfMap->texture->tid);

    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, colorTimeMap->texture->tid);

    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, normRadMap->texture->tid);

    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, depthMap->texture->tid);

    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, KeyFrameIDMap.texture->tid);

    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);

    glDrawTransformFeedback(GL_POINTS, vbos[target].second);

    glBindBuffer(GL_ARRAY_BUFFER, newUnstableVbo);

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

    glDrawTransformFeedback(GL_POINTS, newUnstableFid);

    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

    glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);

    glEndTransformFeedback();

    glDisable(GL_RASTERIZER_DISCARD);

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    unstableProgram->Unbind();
    std::swap(target, renderSource);
    glFinish();
}

void GlobalModel::updateModel()
{
    //std::unique_lock<mutex> lock(UpdateGobalModel);
    //upload transformation matrix to texture
    std::vector<float> DTFK;
    DTFK.reserve(DeltaTransformKF.size() * 16);
    int count_N = 0;
    for(int i = 0; i < DeltaTransformKF.size(); i++){
        Eigen::Matrix4f m = DeltaTransformKF[i];
        for(int j = 0; j < 4; j++){
            for(int k = 0; k < 4; k++){
                DTFK.push_back(m(k, j));
                count_N++;
            }
        }
    }

    glBindTexture(GL_TEXTURE_2D, DeltaTransVec.texture->tid);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DTFK.size(), 1, GL_LUMINANCE, GL_FLOAT, DTFK.data());

    //update global model verteices in GPU storage
    updateDeltaTransProgram->Bind();
    updateDeltaTransProgram->setUniform(Uniform("DeltaTransformKF", 0));
    updateDeltaTransProgram->setUniform(Uniform("DeltaTransDimen", static_cast<float>(DELTA_TRANS_DIMENSION)));

    glBindBuffer(GL_ARRAY_BUFFER, vbos[target].first);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0);

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f)));

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2));

    //curvature attributes
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 3));

    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 4));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, DeltaTransVec.texture->tid);

    glEnable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, vbos[renderSource].second);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbos[renderSource].first);

    glBeginTransformFeedback(GL_POINTS);
    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);
    glDrawTransformFeedback(GL_POINTS, vbos[target].second);
    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);

    glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glDisableVertexAttribArray(3);
    glDisableVertexAttribArray(4);

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);
    updateDeltaTransProgram->Unbind();

    //set vetor pose to uniform for the usage;
    std::swap(target, renderSource);  //update target with rendersource
    glFinish(); 
//    delete[] DTFK;
}


unsigned int GlobalModel::lastCount()
{
    return count;
}

Eigen::Vector4f * GlobalModel::downloadMap()
{

    glFinish();

    Eigen::Vector4f * vertices = new Eigen::Vector4f[count * 5];

    memset(&vertices[0], 0, count * Vertex::SIZE);

    GLuint downloadVbo;

    glGenBuffers(1, &downloadVbo);
    glBindBuffer(GL_ARRAY_BUFFER, downloadVbo);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    glBindBuffer(GL_COPY_READ_BUFFER, vbos[renderSource].first);
    glBindBuffer(GL_COPY_WRITE_BUFFER, downloadVbo);

    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, count * Vertex::SIZE);
    glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, count * Vertex::SIZE, vertices);

    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);
    glDeleteBuffers(1, &downloadVbo);

    glFinish();

    return vertices;
}
