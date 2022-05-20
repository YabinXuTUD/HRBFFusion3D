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

#ifndef GUI_H_
#define GUI_H_

#include <pangolin/pangolin.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>
#include <map>
#include <GPUTexture.h>
#include <Utils/Intrinsics.h>
#include <Utils/Resolution.h>
#include <Shaders/Shaders.h>
#include <Shaders/Vertex.h>

#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049

class GUI
{
    public:
        GUI(bool liveCap, bool showcaseMode)
         : showcaseMode(showcaseMode)
        {
            width = 1580;
            height = 1080;
            panel = 180;          //the left panel on the interface

            width += panel;

            pangolin::Params windowParams;

            windowParams.Set("SAMPLE_BUFFERS", 0);
            windowParams.Set("SAMPLES", 0);

            pangolin::CreateWindowAndBind("HRBFFusion", width, height, windowParams);

            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);  //This affect the glReadPixels, affect how data return to the client memory
            glPixelStorei(GL_PACK_ALIGNMENT, 1);    //One value affects the packing of pixel data into memory

            //Internally render at 3840x2160
            renderBuffer = new pangolin::GlRenderBuffer(3840, 2160),
            colorTexture = new GPUTexture(renderBuffer->width, renderBuffer->height, GL_RGBA32F, GL_LUMINANCE, GL_FLOAT, true);

            colorFrameBuffer = new pangolin::GlFramebuffer;
            colorFrameBuffer->AttachColour(*colorTexture->texture);
            colorFrameBuffer->AttachDepth(*renderBuffer);

            colorProgram = std::shared_ptr<Shader>(loadProgramFromFile("draw_global_surface.vert", "draw_global_surface_phong.frag", "draw_global_surface.geom"));
            fxaaProgram = std::shared_ptr<Shader>(loadProgramFromFile("empty.vert", "fxaa.frag", "quad.geom"));

            pangolin::SetFullscreen(showcaseMode);

            glEnable(GL_DEPTH_TEST);
            glDepthMask(GL_TRUE);
            glDepthFunc(GL_LESS);

            s_cam = pangolin::OpenGlRenderState(pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
                                                pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 1, pangolin::AxisNegY));

            pangolin::Display("cam").SetBounds(0, 1.0f, 0, 1.0f, -640 / 480.0)
                                    .SetHandler(new pangolin::Handler3D(s_cam));

            pangolin::Display(GPUTexture::RGB).SetAspect(640.0f / 480.0f);

            pangolin::Display(GPUTexture::DEPTH_NORM).SetAspect(640.0f / 480.0f);

            pangolin::Display("ModelImg").SetAspect(640.0f / 480.0f);

            pangolin::Display("Model").SetAspect(640.0f / 480.0f);

            std::vector<std::string> labels;
            labels.push_back(std::string("residual"));
            labels.push_back(std::string("threshold"));
            resLog.SetLabels(labels);

            resPlot = new pangolin::Plotter(&resLog, 0, 600, 0, 0.00005, 30, 0.000005);
            resPlot->Track("$i");

            std::vector<std::string> labels2;
            labels2.push_back(std::string("inliers"));
            labels2.push_back(std::string("threshold"));
            inLog.SetLabels(labels2);

            inPlot = new pangolin::Plotter(&inLog, 0, 600, 0, 190000, 30, 1000);
            inPlot->Track("$i");

            if(!showcaseMode)
            {
                  pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(panel)); //from 0-1.0
                  pangolin::Display("multi").SetBounds(pangolin::Attach::Pix(0), 1 / 4.0f, showcaseMode ? 0 : pangolin::Attach::Pix(180), 1.0)
                                          .SetLayout(pangolin::LayoutEqualHorizontal)
                                          .AddDisplay(pangolin::Display(GPUTexture::RGB))
                                          .AddDisplay(pangolin::Display(GPUTexture::DEPTH_NORM))
                                          .AddDisplay(pangolin::Display("ModelImg"))
                                          .AddDisplay(pangolin::Display("Model"))
                                          .AddDisplay(*resPlot)
                                          .AddDisplay(*inPlot);
            }

            start = new pangolin::Var<bool>("ui.Start/Pause", false, false);
            step = new pangolin::Var<bool>("ui.perFrameProcess", false, false);
            save = new pangolin::Var<bool>("ui.save_GlobalModel", false, false);
            saveTexture = new pangolin::Var<bool>("ui.save_Texture", false, false);
            saveCurrentFrames = new pangolin::Var<bool>("ui.save_CurrentFrames", false, false);
            reset = new pangolin::Var<bool>("ui.Reset", false, false);
            flipColors = new pangolin::Var<bool>("ui.Flip RGB", false, true);

            if(liveCap)
            {
                autoSettings = new pangolin::Var<bool>("ui.Auto Settings", true, true);
            }
            else
            {
                autoSettings = 0;
            }

            pyramid = new pangolin::Var<bool>("ui.Pyramid", true, true);
            so3 = new pangolin::Var<bool>("ui.SO(3)", true, true);
            frameToFrameRGB = new pangolin::Var<bool>("ui.FtoFRGB", false, true);
            fastOdom = new pangolin::Var<bool>("ui.FastOd", false, true);
            rgbOnly = new pangolin::Var<bool>("ui.RGBonly", false, true);
            confidenceThreshold = new pangolin::Var<float>("ui.ConfTh", 10.0, 0.0, 24.0);
            depthCutoff = new pangolin::Var<float>("ui.Depthcut", 3.0, 0.0, 12.0);
            icpWeight = new pangolin::Var<float>("ui.ICPweight", 10.0, 0.0, 100.0);

            followPose = new pangolin::Var<bool>("ui.Follow pose", true, true);
//            followGivenPose = new pangolin::Var<bool>("ui.FollowGpose", false, true);
            drawRawCloud = new pangolin::Var<bool>("ui.DrawR", false, true);
            drawFilteredCloud = new pangolin::Var<bool>("ui.DrawF", false, true);
            drawGlobalModel = new pangolin::Var<bool>("ui.DrawGM", true, true);
            drawUnstable = new pangolin::Var<bool>("ui.DrawUP", false, true);
            drawPoints = new pangolin::Var<bool>("ui.DrawP", false, true);
            drawColors = new pangolin::Var<bool>("ui.DrawC", showcaseMode, true);
            drawFxaa = new pangolin::Var<bool>("ui.DrawFxaa", showcaseMode, true);
            drawNormals = new pangolin::Var<bool>("ui.DrawN", false, true);
            drawSparseMapPoints = new pangolin::Var<bool>("ui.DrawSMP", false, true);
            drawKeyFrames = new pangolin::Var<bool>("ui.DrawKF", false, true);
            drawCovisiGrap = new pangolin::Var<bool>("ui.DrawCovisiGrap", false, true);
            drawEssentialGrap = new pangolin::Var<bool>("ui.DrawEssentialGrap", false, true);
            drawSpanningTree = new pangolin::Var<bool>("ui.DrawSpanningTree", false, true);
            drawLoopEdge = new pangolin::Var<bool>("ui.drawLoopEdge", false, true);
            draw_line = new pangolin::Var<bool>("ui.DrawLine", false, true);
            draw_line_global = new pangolin::Var<bool>("ui.DrawLineGlob", false, true);
            draw_prediction = new pangolin::Var<bool>("ui.DrawPrediction", false, true);
            draw_groundTruthTraj = new pangolin::Var<bool>("ui.DrawGroundTruth", false, true);


            gpuMem = new pangolin::Var<int>("ui.GPU memory free", 0);

            totalPoints = new pangolin::Var<std::string>("ui.Total points", "0");
            trackInliers = new pangolin::Var<std::string>("ui.Inliers", "0");
            trackRes = new pangolin::Var<std::string>("ui.Residual", "0");
            logProgress = new pangolin::Var<std::string>("ui.Log", "0");

            if(showcaseMode)
            {
                pangolin::RegisterKeyPressCallback(' ', pangolin::SetVarFunctor<bool>("ui.Reset", true));
            }
        }

        virtual ~GUI()
        {
            delete start;
            delete reset;
            delete inPlot;
            delete resPlot;

            if(autoSettings)
            {
                delete autoSettings;

            }
            delete step;
            delete save;
            delete saveTexture;
            delete saveCurrentFrames;
            delete trackInliers;
            delete trackRes;
            delete confidenceThreshold;
            delete so3;
            delete depthCutoff;
            delete logProgress;
            delete drawFxaa;
            delete fastOdom;
            delete icpWeight;
            delete pyramid;
            delete rgbOnly;
            delete followPose;
            delete drawRawCloud;
            delete totalPoints;
            delete frameToFrameRGB;
            delete flipColors;
            delete drawFilteredCloud;
            delete drawNormals;
            delete drawColors;
            delete drawGlobalModel;
            delete drawUnstable;
            delete drawPoints;
            delete gpuMem;
            delete renderBuffer;
            delete colorFrameBuffer;
            delete colorTexture;
        }

        void preCall()
        {
            //glClearColor(0.05 * !showcaseMode, 0.05 * !showcaseMode, 0.3 * !showcaseMode, 0.0f);
            glClearColor(1.0, 1.0, 1.0, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            width = pangolin::DisplayBase().v.w;
            height = pangolin::DisplayBase().v.h;

            pangolin::Display("cam").Activate(s_cam);
        }

        inline void drawFrustum(const Eigen::Matrix4f & pose)
        {
            if(showcaseMode)
                return;

            Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
            K(0, 0) = Intrinsics::getInstance().fx();
            K(1, 1) = Intrinsics::getInstance().fy();
            K(0, 2) = Intrinsics::getInstance().cx();
            K(1, 2) = Intrinsics::getInstance().cy();

            Eigen::Matrix3f Kinv = K.inverse();

            pangolin::glDrawFrustum  (Kinv,
                                     Resolution::getInstance().width(),
                                     Resolution::getInstance().height(),
                                     pose,
                                     0.1f);
        }

        void displayImg(const std::string & id, GPUTexture * img)
        {
            if(showcaseMode)
                return;

            glDisable(GL_DEPTH_TEST);

            pangolin::Display(id).Activate();
            img->texture->RenderToViewport(true);

            glEnable(GL_DEPTH_TEST);
        }

        void postCall()
        {
            GLint cur_avail_mem_kb = 0;
            glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

            int memFree = cur_avail_mem_kb / 1024;

            gpuMem->operator=(memFree);

            pangolin::FinishFrame();

            glFinish();
        }

        void drawFXAA(pangolin::OpenGlMatrix mvp,
                      pangolin::OpenGlMatrix mv,
                      const std::pair<GLuint, GLuint> & model,
                      const float threshold,
                      const int time,
                      const int timeDelta,
                      const bool invertNormals)
        {
            //First pass computes positions, colors and normals per pixel
            colorFrameBuffer->Bind();

            glPushAttrib(GL_VIEWPORT_BIT);

            glViewport(0, 0, renderBuffer->width, renderBuffer->height);

            glClearColor(0.05 * !showcaseMode, 0.05 * !showcaseMode, 0.3 * !showcaseMode, 0);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            colorProgram->Bind();

            colorProgram->setUniform(Uniform("MVP", mvp));

            colorProgram->setUniform(Uniform("threshold", threshold));

            colorProgram->setUniform(Uniform("time", time));

            colorProgram->setUniform(Uniform("timeDelta", timeDelta));

            colorProgram->setUniform(Uniform("signMult", invertNormals ? 1.0f : -1.0f));

            colorProgram->setUniform(Uniform("colorType", (drawNormals->Get() ? 1 : drawColors->Get() ? 2 : false ? 3 : 0)));

            colorProgram->setUniform(Uniform("unstable", drawUnstable->Get()));

            colorProgram->setUniform(Uniform("drawWindow", false));

            Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
            //This is for the point shader
            colorProgram->setUniform(Uniform("pose", pose));

            Eigen::Matrix4f modelView = mv;

            Eigen::Vector3f lightpos = modelView.topRightCorner(3, 1);

            colorProgram->setUniform(Uniform("lightpos", lightpos));

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

            colorFrameBuffer->Unbind();

            colorProgram->Unbind();

            glPopAttrib();

            fxaaProgram->Bind();

            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, colorTexture->texture->tid);

            Eigen::Vector2f resolution(renderBuffer->width, renderBuffer->height);

            fxaaProgram->setUniform(Uniform("tex", 0));
            fxaaProgram->setUniform(Uniform("resolution", resolution));

            glDrawArrays(GL_POINTS, 0, 1);

            fxaaProgram->Unbind();

            glBindFramebuffer(GL_READ_FRAMEBUFFER, colorFrameBuffer->fbid);

            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

            glBlitFramebuffer(0, 0, renderBuffer->width, renderBuffer->height, 0, 0, width, height, GL_DEPTH_BUFFER_BIT, GL_NEAREST);

            glBindTexture(GL_TEXTURE_2D, 0);

            glFinish();
        }

        bool showcaseMode;
        int width;
        int height;
        int panel;

        pangolin::Var<bool> * start,
                            * step,
                            * save,
                            * saveTexture,
                            * saveCurrentFrames,
                            * reset,
                            * flipColors,
                            * rgbOnly,
                            * pyramid,
                            * so3,
                            * frameToFrameRGB,
                            * fastOdom,
                            * followPose,
//                            * followGivenPose,
                            * drawRawCloud,
                            * drawFilteredCloud,
                            * drawNormals,
                            * autoSettings,
                            * drawColors,
                            * drawFxaa,
                            * drawGlobalModel,
                            * drawUnstable,
                            * drawPoints,
                            * drawSparseMapPoints,
                            * drawKeyFrames,
                            * drawCovisiGrap,
                            * drawEssentialGrap,
                            * drawSpanningTree,
                            * drawLoopEdge,
                            * draw_line,
                            * draw_line_global,
                            * draw_prediction,
                            * draw_groundTruthTraj;
        pangolin::Var<int> * gpuMem;
        pangolin::Var<std::string> * totalPoints,                                                                   
                                   * trackInliers,
                                   * trackRes,
                                   * logProgress;
        pangolin::Var<float> * confidenceThreshold,
                             * depthCutoff,
                             * icpWeight;

        pangolin::DataLog resLog, inLog;
        pangolin::Plotter * resPlot,
                          * inPlot;

        pangolin::OpenGlRenderState s_cam;

        pangolin::GlRenderBuffer * renderBuffer;
        pangolin::GlFramebuffer * colorFrameBuffer;
        GPUTexture * colorTexture;
        std::shared_ptr<Shader> colorProgram;
        std::shared_ptr<Shader> fxaaProgram;
};


#endif /* GUI_H_ */
