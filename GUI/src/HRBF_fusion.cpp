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

#include "HRBF_fusion.h"

MainController::MainController(int argc, char * argv[])
 : good(true),
   aStep(false),
   hrbfFusion(nullptr),
   gui(nullptr),
   groundTruthOdometry(nullptr),
   logReader(nullptr),
   framesToSkip(0),
   resetButton(false),
   resizeStream(nullptr)
{
    //*********************parameter initilization******************************//
    std::string empty;
    //load application parameter from file
    ParameterFile pf("../GlobalStateParam.txt");

    //set parameter to global variable
    GlobalStateParam::getInstance().readMembers(pf);
    //GlobalStateParam::getInstance().print();

    //set current working directory
    int ret = chdir(GlobalStateParam::get().currentWorkingDirectory.c_str());

    //Load camera parameters from open CV settings file and set global variables
    cv::FileStorage fSettings(GlobalStateParam::get().parameterFileCvFormat, cv::FileStorage::READ);
    //load camera intrinsics and resolution
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];
    int width = fSettings["Camera.width"];
    int height = fSettings["Camera.height"];
    Resolution::getInstance(width, height);
    Intrinsics::getInstance(fx, fy, cx, cy);

//    //load default (groundtruth) trajectory (for visualization)
//    if(0)
//        load_trajectory();

    //input datatype: Live camera
    if(GlobalStateParam::get().sensorType == 1){
        //set up the camera, waiting for frames to come, if kinect v2 is not ok then use realsense
        bool flipColors = Parse::get().arg(argc,argv,"-f",empty) > -1;
        logReader = new LiveLogReader(logFile, flipColors, LiveLogReader::CameraType::OpenNI2);
        good = ((LiveLogReader *)logReader)->cam->ok();
#ifdef WITH_REALSENSE
        if(!good){
          delete logReader;
          logReader = new LiveLogReader(logFile, flipColors, LiveLogReader::CameraType::RealSense);
          good = ((LiveLogReader *)logReader)->cam->ok();
        }
#endif
    }
    //input datatype: from compressed *.klg file
    if(GlobalStateParam::get().sensorType == 2) {
        std::cout << "Using local klg file!" << std::endl;
        logReader = new RawLogReader(GlobalStateParam::get().klgFileName, Parse::get().arg(argc, argv, "-f", empty) > -1);
    }

    //input datatype: from raw rgb/depth images
    if(GlobalStateParam::get().sensorType == 3){
        std::cout << "Using raw images!" << std::endl;
        logReader = new RawImageLogReader(GlobalStateParam::get().AssociationFile, Parse::get().arg(argc, argv, "-f", empty) > -1);
    }

    //default parameter settings
    confidence = GlobalStateParam::get().globalConfidenceThreshold;
    depth = GlobalStateParam::get().globalDepthCutoff;
    icp = GlobalStateParam::get().registrationJointICPWeight;
    icpErrThresh = GlobalStateParam::get().registrationICPErrorThreshold;
    covThresh = GlobalStateParam::get().registrationICPCovarianceThreshold;
    photoThresh = GlobalStateParam::get().registrationColorPhotoThreshold;
    framesToSkip = GlobalStateParam::get().globalFrameToSkip;
    timeDelta = 200;
    icpCountThresh = 40000;
    start = GlobalStateParam::get().globalStartFrame;
    so3 = GlobalStateParam::get().registrationPreAlignSO3;
    end = std::numeric_limits<unsigned short>::max();     //Funny bound, since we predict times in this format really!
    if(GlobalStateParam::get().globalEndFrame > 0)
        end = GlobalStateParam::get().globalEndFrame;

    //user input parameter
    fastOdom = Parse::get().arg(argc, argv, "-fo", empty) > -1;  //fast odometry
    rewind = Parse::get().arg(argc, argv, "-r", empty) > -1;
    frameToFrameRGB = Parse::get().arg(argc, argv, "-ftf", empty) > -1;
    draw_unstable = true;

    //setup the gui
    gui = new GUI(logFile.length() == 0, Parse::get().arg(argc, argv, "-sc", empty) > -1);
    gui->flipColors->Ref().Set(logReader->flipColors);
    gui->rgbOnly->Ref().Set(false);
    gui->pyramid->Ref().Set(true);
    gui->fastOdom->Ref().Set(fastOdom);
    gui->confidenceThreshold->Ref().Set(confidence);
    gui->depthCutoff->Ref().Set(depth);
    gui->icpWeight->Ref().Set(icp);
    gui->so3->Ref().Set(so3);
    gui->frameToFrameRGB->Ref().Set(frameToFrameRGB);
    gui->drawUnstable->Ref().Set(draw_unstable);

    resizeStream = new Resize(Resolution::getInstance().width(),
                              Resolution::getInstance().height(),
                              Resolution::getInstance().width() / 2,
                              Resolution::getInstance().height() / 2);
}

MainController::~MainController()
{
    if(hrbfFusion)
    {
        delete hrbfFusion;
    }

    if(gui)
    {
        delete gui;
    }

    if(groundTruthOdometry)
    {
        delete groundTruthOdometry;
    }

    if(logReader)
    {
        delete logReader;
    }

    if(resizeStream)
    {
        delete resizeStream;
    }
}


void MainController::launch()
{
    while(good)
    {
        if(hrbfFusion)
        {
            run();
        }

        if(hrbfFusion == nullptr || resetButton)
        {
            resetButton = false;

            if(hrbfFusion)
            {
                delete hrbfFusion;
            }
            logReader->rewind();
            hrbfFusion = new HRBFFusion(icpCountThresh,
                                        icpErrThresh,
                                        confidence,
                                        depth,
                                        icp,
                                        fastOdom,
                                        so3,
                                        frameToFrameRGB);
        }
        else
        {
            break;
        }
    }
}

void MainController::run()
{
    while(!pangolin::ShouldQuit() && !((!logReader->hasMore()) && quiet) && !(hrbfFusion->getTick() == end && quiet))
    {
        if(gui->start->Get() || pangolin::Pushed(*gui->step) || gui->showcaseMode)
        {
            if((logReader->hasMore() || rewind) && hrbfFusion->getTick() < end)
            {
                if(rewind)
                {
                    if(!logReader->hasMore())
                    {
                        logReader->getBack();
                    }
                    else
                    {
                        logReader->getNext();
                    }

                    if(logReader->rewound())
                    {
                        logReader->currentFrame = 0;
                    }
                }
                else
                {
                    logReader->getNext();
                }

                if(hrbfFusion->getTick() < start)
                {
                    hrbfFusion->setTick(start);
                    logReader->fastForward(start);
                    logReader->getNext();
                }

                float weightMultiplier = framesToSkip + 1;

                if(framesToSkip > 0)
                {
                    hrbfFusion->setTick(hrbfFusion->getTick() + framesToSkip);
                    logReader->fastForward(logReader->currentFrame + framesToSkip);
                    framesToSkip = 0;
                }

                hrbfFusion->processFrame(logReader->rgb, logReader->depth, logReader->timestamp, weightMultiplier);

                if(framesToSkip && Stopwatch::getInstance().getTimings().at("Run") > 1000.f / 30.f)
                {
                    framesToSkip = int(Stopwatch::getInstance().getTimings().at("Run") / (1000.f / 30.f));
                }

                aStep = true;
            }
        }

        if(gui->followPose->Get())
        {
            pangolin::OpenGlMatrix mv;

            Eigen::Matrix4f currPose = hrbfFusion->getCurrPose();
            Eigen::Matrix3f currRot = currPose.topLeftCorner(3, 3);

            Eigen::Quaternionf currQuat(currRot);
            Eigen::Vector3f forwardVector(0, 0, 1);
            Eigen::Vector3f upVector(0, GlobalStateParam::get().globalInputICLNUIMDataset ? 1 : -1, 0);

            Eigen::Vector3f forward = (currQuat * forwardVector).normalized();
            Eigen::Vector3f up = (currQuat * upVector).normalized();

            Eigen::Vector3f eye(currPose(0, 3), currPose(1, 3), currPose(2, 3));

            eye -= forward;

            Eigen::Vector3f at = eye + forward;

            Eigen::Vector3f z = (eye - at).normalized();     // Forward
            Eigen::Vector3f x = up.cross(z).normalized();    // Right
            Eigen::Vector3f y = z.cross(x);

            Eigen::Matrix4d m;
            m << x(0),  x(1),  x(2),  -(x.dot(eye)),
                 y(0),  y(1),  y(2),  -(y.dot(eye)),
                 z(0),  z(1),  z(2),  -(z.dot(eye)),
                    0,     0,     0,              1;

            memcpy(&mv.m[0], m.data(), sizeof(Eigen::Matrix4d));

            gui->s_cam.SetModelViewMatrix(mv);
        }

        gui->preCall();

        //Tracking inliers in histgram
        std::stringstream stri;
        stri << hrbfFusion->getFrameToModel().lastICPCount;
        gui->trackInliers->Ref().Set(stri.str());
        //Tracking ICP error in histgram.
        std::stringstream stre;

        stre << (std::isnan(hrbfFusion->getFrameToModel().lastICPError) ? 0 : hrbfFusion->getFrameToModel().lastICPError);
        gui->trackRes->Ref().Set(stre.str());

        if(gui->start->Get()|| aStep) {
            gui->resLog.Log((std::isnan(hrbfFusion->getFrameToModel().lastICPError) ? std::numeric_limits<float>::max() : hrbfFusion->getFrameToModel().lastICPError), icpErrThresh);
            gui->inLog.Log(hrbfFusion->getFrameToModel().lastICPCount, icpCountThresh);
            aStep = false;         
        }
        //render feedback point
        Eigen::Matrix4f pose = hrbfFusion->getCurrPose();
        if(gui->drawRawCloud->Get() || gui->drawFilteredCloud->Get())
            hrbfFusion->computeFeedbackBuffers();
        if(gui->drawRawCloud->Get())
            hrbfFusion->getFeedbackBuffers().at(FeedbackBuffer::RAW)->render(gui->s_cam.GetProjectionModelViewMatrix(), pose, gui->drawNormals->Get(), gui->drawColors->Get());
        if(gui->drawFilteredCloud->Get())
            hrbfFusion->getFeedbackBuffers().at(FeedbackBuffer::FILTERED)->render(gui->s_cam.GetProjectionModelViewMatrix(), pose, gui->drawNormals->Get(), gui->drawColors->Get());
        if(gui->drawGlobalModel->Get()){
            glFinish();
            if(gui->drawFxaa->Get()){
                gui->drawFXAA(gui->s_cam.GetProjectionModelViewMatrix(),
                              gui->s_cam.GetModelViewMatrix(),
                              hrbfFusion->getGlobalModel().model(),
                              hrbfFusion->getConfidenceThreshold(),
                              hrbfFusion->getTick(),
                              200,
                              GlobalStateParam::get().globalInputICLNUIMDataset);
            }
            else{
                hrbfFusion->getGlobalModel().renderPointCloud(gui->s_cam.GetProjectionModelViewMatrix(),
                                                           hrbfFusion->getConfidenceThreshold(),
                                                           gui->drawUnstable->Get(),
                                                           gui->drawNormals->Get(),
                                                           gui->drawColors->Get(),
                                                           gui->drawPoints->Get(),
                                                           false,
                                                           false,
                                                           hrbfFusion->getTick(),
                                                           200);
            }
            glFinish();
        }

        if(gui->drawSparseMapPoints->Get())
        {
            //draw global sparse map points
            hrbfFusion->getSparseMapDrawer()->DrawMapPoints(gui->s_cam.GetProjectionModelViewMatrix());
            glFinish();
        }

        if(gui->drawKeyFrames->Get())
        {
            hrbfFusion->getSparseMapDrawer()->DrawKeyFrames(true, false, false, false, false);
        }

        if(gui->drawCovisiGrap->Get())
        {
            hrbfFusion->getSparseMapDrawer()->DrawKeyFrames(true, true, false, false, false);
        }

        if(gui->drawEssentialGrap->Get())
        {
            hrbfFusion->getSparseMapDrawer()->DrawKeyFrames(true, false, true, false, false);
        }

        if(gui->drawSpanningTree->Get())
        {
            hrbfFusion->getSparseMapDrawer()->DrawKeyFrames(true, false, false, true, false);
        }

        if(gui->drawLoopEdge->Get())
        {
            hrbfFusion->getSparseMapDrawer()->DrawKeyFrames(true, false, false, true, true);
        }


        if(gui->draw_prediction->Get())
        {
           hrbfFusion->getIndexMap().renderHRBFPrediction(gui->s_cam.GetProjectionModelViewMatrix(), pose);
            //hrbfFusion->getIndexMap().renderSurfelPrediction(gui->s_cam.GetProjectionModelViewMatrix(), pose);
        }

        if(gui->draw_groundTruthTraj->Get())
        {
            //draw ground truth trajectory(in green)
            glLineWidth(1.5);
            glColor4f(0.0f,1.0f,0.0f,0.6f);
            glBegin(GL_LINES);
            for(int i = 0; i < poses.size() - 1; i++)
            {
                glVertex3f(poses[i](0, 3), poses[i](1, 3), poses[i](2, 3));
                glVertex3f(poses[i + 1](0, 3), poses[i + 1](1, 3), poses[i + 1](2, 3));
            }
            glEnd();

            //draw current camera trajectory(in blue)
            glLineWidth(1.5);
            glColor4f(0.0f,0.0f,1.0f,0.6f);
            glBegin(GL_LINES);
            if(hrbfFusion->trajectory_manager->poses.size() > 0)
            {
                for(int i = 0; i < hrbfFusion->trajectory_manager->poses.size() - 1; i++)
                {
                    Eigen::Matrix4f pose_1 = /*poses[0] **/ hrbfFusion->trajectory_manager->poses[i];
                    Eigen::Matrix4f pose_2 = /*poses[0] **/ hrbfFusion->trajectory_manager->poses[i + 1];
                    glVertex3f(pose_1(0, 3), pose_1(1, 3), pose_1(2, 3));
                    glVertex3f(pose_2(0, 3), pose_2(1, 3), pose_2(2, 3));
                }
            }
            glEnd();
        }

        if(hrbfFusion->getLost()){
            glColor3f(1, 1, 0);
        }
        else{
            glColor3f(1, 0, 1);
        }
        gui->drawFrustum(pose);
        glColor3f(1, 1, 1);

        const std::vector<PoseMatch> & poseMatches = hrbfFusion->getPoseMatches();
        int maxDiff = 0;
        for(size_t i = 0; i < poseMatches.size(); i++){
            if(poseMatches.at(i).secondId - poseMatches.at(i).firstId > maxDiff)
            {
                maxDiff = poseMatches.at(i).secondId - poseMatches.at(i).firstId;
            }
        }

        //for visualization
        hrbfFusion->normaliseDepth(0.3f, gui->depthCutoff->Get());

        for(std::map<std::string, GPUTexture*>::const_iterator it = hrbfFusion->getTextures().begin(); it != hrbfFusion->getTextures().end(); ++it)
        {
            //display first two images
            if(it->second->draw)
            {
                gui->displayImg(it->first, it->second);
            }
        }

        gui->displayImg("ModelImg",
                        hrbfFusion->getIndexMap().normalTexHRBF()
                         );
        gui->displayImg("Model",
                        hrbfFusion->getTextures()[GPUTexture::NORMAL]
                        );

        std::stringstream strs;
        strs << hrbfFusion->getGlobalModel().lastCount();
        gui->totalPoints->operator=(strs.str());

        std::stringstream strs5;
        strs5 << hrbfFusion->getTick() << "/" << logReader->getNumFrames();
        gui->logProgress->operator=(strs5.str());

        gui->postCall();

        logReader->flipColors = gui->flipColors->Get();
        hrbfFusion->setRgbOnly(gui->rgbOnly->Get());
        hrbfFusion->setPyramid(gui->pyramid->Get());
        hrbfFusion->setFastOdom(gui->fastOdom->Get());
        hrbfFusion->setConfidenceThreshold(gui->confidenceThreshold->Get());
        hrbfFusion->setDepthCutoff(gui->depthCutoff->Get());
        hrbfFusion->setIcpWeight(gui->icpWeight->Get());
        hrbfFusion->setSo3(gui->so3->Get());
        hrbfFusion->setFrameToFrameRGB(gui->frameToFrameRGB->Get());
        resetButton = pangolin::Pushed(*gui->reset);

        if(gui->autoSettings)
        {
            static bool last = gui->autoSettings->Get();

            if(gui->autoSettings->Get() != last)
            {
                last = gui->autoSettings->Get();
                static_cast<LiveLogReader *>(logReader)->setAuto(last);
            }
        }
        Stopwatch::getInstance().sendAll();

        //current frame start from 1
        if(resetButton){
            break;
        }

        if(pangolin::Pushed(*gui->save))
        {
            hrbfFusion->savePly("hrbf_globalModel.ply");
            if(GlobalStateParam::get().optimizationUseLocalBA || GlobalStateParam::get().get().optimizationUseGlobalBA)
                hrbfFusion->getSparseMap()->DownloadSparseMapPoints();
        }

        if(pangolin::Pushed(*gui->saveTexture))
        {
            hrbfFusion->getIndexMap().downloadTexture(hrbfFusion->getCurrPose(), hrbfFusion->getTick() - 1);
            hrbfFusion->getFillIn().downloadtexture(Eigen::Matrix4f::Identity(), hrbfFusion->getTick() - 2, false, " ");
            hrbfFusion->downloadTextures();
            hrbfFusion->getFrameToModel().DownloadGPUMaps();
        }
        if(pangolin::Pushed(*gui->saveCurrentFrames))
        {
            hrbfFusion->saveFeedbackBufferPly("raw","");
            hrbfFusion->saveFeedbackBufferPly("filtered", "");
            std::cout << "save current feedback and rendered model"<<std::endl;
        }
    }
}


//void MainController::load_trajectory()
//{
//    std::ifstream file;
//    std::string line;
//    file.open(GlobalStateParam::get().globalInputTrajectoryFile.c_str());
//    if(!file.is_open())
//    {
//        std::cerr << "Open ground truth trajectory (TUM) failed" << std::endl;
//        exit(0);
//    }

//    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
//    int count = 0;
//    Eigen::Matrix4f pose_init = Eigen::Matrix4f::Identity();
//    while(!file.eof())
//    {
//       unsigned long long int utime;                //timestamp
//       float x, y, z, qx, qy, qz, qw;

//       std::getline(file, line);
//       if(line.empty() || line[0] == '#') continue;
//       size_t first_spacePos = line.find_first_of(" ");
//       std::remove(line.begin(), line.begin() + first_spacePos, '.');
//       int n = sscanf(line.c_str(), "%llu %f %f %f %f %f %f %f", &utime, &x, &y, &z, &qx, &qy, &qz, &qw);

//       if(file.eof())
//           break;
//       assert(n==8);

//       Eigen::Quaternionf q(qw, qx, qy, qz);
//       Eigen::Vector3f t(x, y, z);

//       Eigen::Isometry3f T;
//       T.setIdentity();
//       T.pretranslate(t).rotate(q);
//       pose = T.matrix();
//       poses.push_back(pose);
//       count++;
//    }
//    file.close();
//    //!!!important: remember to align the timestamp between the groundtruth trajectory and the estimated trajectory.
////    Eigen::Matrix4f M;
////    M <<  -1,  0, 0, 0,
////          0,  1, 0, 0,
////          0,  0, 1, 0,
////          0,  0, 0, 1;
////    for (int i = 0; i < poses.size(); i++){
////        poses[i] = poses[i].inverse().eval();
////    }
//    //Change Global coodinate frame to Local coodinate frame
////    Eigen::Matrix4f M;(for TUM fr1/desk trajectory)
////    M << 0.888423, -0.331777, 0.317219, 0.764334,
////    0.310507, 0.943342, 0.117008, -0.067391,
////    -0.338067, -0.005454, 0.941106, 0.416784,
////    0.000000, 0.000000, 0.000000, 1.000000;
////    Eigen::Matrix4f M_inv = M.inverse();
//    Eigen::Matrix4f init_pose = poses[0];
//    for (int i = 0; i < poses.size(); i++){
//        poses[i] = init_pose.inverse() * poses[i];
//    }

////    for (int i = 0; i < poses.size(); i++){
////        poses[i] = M_inv * poses[i];
////    }
//}

