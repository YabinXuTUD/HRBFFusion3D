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

#include "FeedbackBuffer.h"

const std::string FeedbackBuffer::RAW = "RAW";
const std::string FeedbackBuffer::FILTERED = "FILTERED";
const std::string FeedbackBuffer::OUTLIER_REMOVAL = "OUTLIER_REMOVAL";

FeedbackBuffer::FeedbackBuffer(std::shared_ptr<Shader> program) //this is for the computation usage
 : program(program),
   drawProgram(loadProgramFromFile("draw_feedback.vert", "draw_feedback.frag")),
   bufferSize(Resolution::getInstance().numPixels() * Vertex::SIZE),
   count(0)
{
    float * vertices = new float[bufferSize];

    memset(&vertices[0], 0, bufferSize);

    glGenTransformFeedbacks(1, &fid);  //generate transform feedback id;
    glGenBuffers(1, &vbo);             //generate vertex buffer object
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, &vertices[0], GL_STREAM_DRAW);   //allocate buffer momery
    glBindBuffer(GL_ARRAY_BUFFER, 0);  //unbind buffer

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

    glGenBuffers(1, &uvo); //create uv object and bind with it
    glBindBuffer(GL_ARRAY_BUFFER, uvo);
    glBufferData(GL_ARRAY_BUFFER, uv.size() * sizeof(Eigen::Vector2f), &uv[0], GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    program->Bind();  //set the transform feedback attributes use shader program

    int loc[5] =
    {
        glGetVaryingLocationNV(program->programId(), "vPosition0"),
        glGetVaryingLocationNV(program->programId(), "vColor0"),
        glGetVaryingLocationNV(program->programId(), "vNormRad0"),
        glGetVaryingLocationNV(program->programId(), "curv_map_max0"),
        glGetVaryingLocationNV(program->programId(), "curv_map_min0"),
    };

    glTransformFeedbackVaryingsNV(program->programId(), 5, loc, GL_INTERLEAVED_ATTRIBS); //in geometry shader

    program->Unbind();

    glGenQueries(1, &countQuery);
}

FeedbackBuffer::~FeedbackBuffer()
{
    glDeleteTransformFeedbacks(1, &fid);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &uvo);
    glDeleteQueries(1, &countQuery);
}

void FeedbackBuffer::compute(pangolin::GlTexture * color,
                             pangolin::GlTexture * depth,
                             const int & time,
                             const float depthCutoff)
{
    program->Bind();

    Eigen::Vector4f cam(Intrinsics::getInstance().cx(),
                  Intrinsics::getInstance().cy(),
                  1.0f / Intrinsics::getInstance().fx(),
                  1.0f / Intrinsics::getInstance().fy());

    program->setUniform(Uniform("cam", cam));
    program->setUniform(Uniform("threshold", 0.0f));
    program->setUniform(Uniform("cols", static_cast<float>(Resolution::getInstance().cols())));
    program->setUniform(Uniform("rows", static_cast<float>(Resolution::getInstance().rows())));
    program->setUniform(Uniform("time", time));
    program->setUniform(Uniform("gSampler", 0));
    program->setUniform(Uniform("cSampler", 1));
    program->setUniform(Uniform("maxDepth", depthCutoff));
    program->setUniform(Uniform("windowMultiply", GlobalStateParam::get().preprocessingCurvEstimationWindow));
    program->setUniform(Uniform("radius_multiplier", GlobalStateParam::get().preprocessingInitRadiusMultiplier));

    glEnableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, uvo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glEnable(GL_RASTERIZER_DISCARD);

    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, fid); //the transform feedback records the output of the geometry shader, NOT the vertex shader NOR fragment shader!!

    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, vbo);

    glBeginTransformFeedback(GL_POINTS);

    glBeginQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN, countQuery);//

    glActiveTexture(GL_TEXTURE0);   //activate texture
    glBindTexture(GL_TEXTURE_2D, depth->tid); //create texture with tid, bind with target

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, color->tid);

    glDrawArrays(GL_POINTS, 0, Resolution::getInstance().numPixels());

    glBindTexture(GL_TEXTURE_2D, 0); //unbind texture

    glActiveTexture(GL_TEXTURE0);

    glEndTransformFeedback();

    glEndQuery(GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN);     //

    glGetQueryObjectuiv(countQuery, GL_QUERY_RESULT, &count); //

    glDisable(GL_RASTERIZER_DISCARD);

    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTransformFeedback(GL_TRANSFORM_FEEDBACK, 0);

    program->Unbind();

    glFinish();
}

void FeedbackBuffer::render(pangolin::OpenGlMatrix mvp,
                            const Eigen::Matrix4f & pose,
                            const bool drawNormals,
                            const bool drawColors)
{
    drawProgram->Bind();

    drawProgram->setUniform(Uniform("MVP", mvp));
    drawProgram->setUniform(Uniform("pose", pose));
    drawProgram->setUniform(Uniform("colorType", (drawNormals ? 1 : drawColors ? 2 : 0)));

    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glEnableVertexAttribArray(0); //the index of the accurate attributes(position/normal/color) in your buffer
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, 0); //Position + confidence

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 1)); //color + timestamp

    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, Vertex::SIZE, reinterpret_cast<GLvoid*>(sizeof(Eigen::Vector4f) * 2)); //normal + radii

    //enable point size
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);

    glDrawTransformFeedback(GL_POINTS, fid);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    drawProgram->Unbind();

    glDisable(GL_PROGRAM_POINT_SIZE);
    glDisable(GL_POINT_SPRITE);
}

Eigen::Vector4f* FeedbackBuffer::genFeedbackPointCloud(){

    Eigen::Vector4f * feedbackVertices = new Eigen::Vector4f[count * 5];
    memset(&feedbackVertices[0], 0, count * Vertex::SIZE);

    GLuint genfeedbackPointCloud;
    glGenBuffers(1, &genfeedbackPointCloud);
    glBindBuffer(GL_ARRAY_BUFFER, genfeedbackPointCloud);
    glBufferData(GL_ARRAY_BUFFER, bufferSize, 0, GL_STREAM_DRAW);

    glBindBuffer(GL_COPY_READ_BUFFER, vbo); //bind the copy vertex object
    glBindBuffer(GL_COPY_WRITE_BUFFER, genfeedbackPointCloud); //bind the write vertex object

    glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, count * Vertex::SIZE); //proceed the copy object
    glGetBufferSubData(GL_COPY_WRITE_BUFFER, 0, count * Vertex::SIZE, feedbackVertices);  //copy the generated vertex buffer to the feedback vertices.

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_COPY_READ_BUFFER, 0);
    glBindBuffer(GL_COPY_WRITE_BUFFER, 0);

    //glDeleteBuffers(1,&globalPointFeedback);
    glDeleteBuffers(1,&genfeedbackPointCloud);

    glFinish();
    return feedbackVertices;
}

unsigned int FeedbackBuffer::getcount(){
    return count;
}
