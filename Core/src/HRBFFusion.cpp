
#include "HRBFFusion.h"

//-------- global variables --------
std::condition_variable condVar;
std::condition_variable condVarGlobalBA;

HRBFFusion::HRBFFusion(const int countThresh,
                       const float errThresh,
                       const float confidence,
                       const float depthCut,
                       const float icpThresh,
                       const bool fastOdom,
                       const bool so3,
                       const bool frameToFrameRGB)
 : frameToModel(Resolution::getInstance().width(),
                Resolution::getInstance().height(),
                Intrinsics::getInstance().cx(),
                Intrinsics::getInstance().cy(),
                Intrinsics::getInstance().fx(),
                Intrinsics::getInstance().fy()), 
   currPose(Eigen::Matrix4f::Identity()),
   lastKeyFramePose(Eigen::Matrix4f::Identity()),
   tick(1),
   icpCountThresh(countThresh),
   icpErrThresh(errThresh),
   consSample(20),
   resize(Resolution::getInstance().width(),
          Resolution::getInstance().height(),
          Resolution::getInstance().width() / consSample,
          Resolution::getInstance().height() / consSample),
   imageBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
   verticesBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
   timesBuff(Resolution::getInstance().rows() / consSample, Resolution::getInstance().cols() / consSample),
   lost(false),
   lastFrameRecovery(false),
   maxDepthProcessed(20.0f),
   rgbOnly(false),
   icpWeight(icpThresh),
   pyramid(true),
   fastOdom(fastOdom),
   confidenceThreshold(confidence),
   so3(so3),
   frameToFrameRGB(frameToFrameRGB),
   depthCutoff(depthCut),
   insertAsubmap(false),
   indexSubmap(0),
   mbStopRequested(0),
   mbPoseGragh(false),
   mbGlobalBA(false)
{
    LoadCameraParaAndInitORBExtractor();
    createTextures();
    createCompute();
    createFeedbackBuffers();
    trajectory_manager = new TrajectoryManager();
    if(GlobalStateParam::get().globalInputLoadTrajectory)
    {
       trajectory_manager->LoadFromFile();
       currPose = trajectory_manager->poses[0];
    }

    Stopwatch::getInstance().setCustomSignature(12431231);

    weighting = 1.0;

    globalModel = new GlobalModel();

    //Retrieve paths to images from association file
    if(GlobalStateParam::get().sensorType == 3)
    {
        using namespace std;
        string strAssociationFilename = string(GlobalStateParam::get().AssociationFile);
        LoadImagesAssociationFile(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

        int nImages = vstrImageFilenamesRGB.size();
        if(vstrImageFilenamesRGB.empty())
        {
            cerr << endl << "No images found in provided path." << endl;
            exit(0);
        }
        else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
        {
           cerr << endl << "Different number of images for rgb and depth." << endl;
           exit(0);
        }
    }

    //Initialize BA optimization
    if(GlobalStateParam::get().optimizationUseLocalBA || GlobalStateParam::get().optimizationUseGlobalBA)
    {
        std::cout << "Loading vocabulary......." << std::endl;
        mpORBVocabulary = new ORBVocabulary();
        bool bVocLoad = mpORBVocabulary->loadFromTextFile(GlobalStateParam::get().optimizationVocabularyFile);
        if(!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << GlobalStateParam::get().optimizationVocabularyFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;

        //Create KeyFrame Database, for loop closure detection
        mpKeyFrameDB = new KeyFrameDatabase(*mpORBVocabulary);

        //Create Sparse Map, for local and global BA
        spMap = new Map();
        cout << "Create the Sparse Map Working" << endl;

        //draw current frame
        mpFrameDrawer = new FrameDrawer(spMap);
        cout << "Create FrameDrawer Working" << endl;

        //draw sparse map
        mpMapDrawer = new MapDrawer(spMap, GlobalStateParam::get().parameterFileCvFormat);
        cout << "Create MapDrawer Working" << endl;

        //Initialize the Local Mapping thread
        mpLocalMapper = new LocalMapping(spMap, false);
        mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run, mpLocalMapper);
        cout << "Create mpLocalMapper thread" << endl;

        //Initialize the Loop Closing thread
        mpLoopCloser = new LoopClosing(spMap, mpKeyFrameDB, mpORBVocabulary, true);
        mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);
        cout << "Create Loop Closing thread" << endl;

        mpLocalMapper->SetHRBFFusion(this);
        mpLocalMapper->SetLoopCloser(mpLoopCloser);

        mpLoopCloser->SetHRBFFusion(this);
        mpLoopCloser->SetLocalMapper(mpLocalMapper);
    }
}

HRBFFusion::~HRBFFusion()
{
    if(GlobalStateParam::get().globalInputICLNUIMDataset)
    {
        savePly("hrbf_globalModel.ply");
    }

    if(GlobalStateParam::get().globalOutputSaveTrjectoryFile){
        trajectory_manager->SaveTrajectoryToFile();
    }

    if(GlobalStateParam::get().optimizationUseLocalBA || GlobalStateParam::get().optimizationUseGlobalBA)
    {
       SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    }

    if(GlobalStateParam::get().globalOutputCalculateMeanDistWithGroundTruth)
    {
        //output mean_pose_error of each frame;
        std::ofstream f_m;
        f_m.open("mean_error_pose.txt");
        for(int i = 0; i < MeanErrorSet.size(); i++)
        {
            f_m << i << " " << MeanErrorSet[i] << "\n";
        }
        f_m.close();

        std::ofstream inlier_num_out;
        inlier_num_out.open("inliers_num.txt");
        for(int i = 0; i < inliers_num.size(); i++)
        {
            inlier_num_out << inliers_num[i] << "\n";
        }
        inlier_num_out.close();
    }

    if(GlobalStateParam::get().globalOutputsaveTimings)
    {
        //save timings
        std::ofstream f_t;
        f_t.open("sequence_processing_time.txt");
        for(int i = 0; i < processingTimes.size(); i++)
        {
            f_t << i << " ";
            for(std::map<std::string, float>::const_iterator it = processingTimes[i].begin(); it != processingTimes[i].end(); it++)
            {
                f_t << it->second << " ";
            }
            f_t << "\n";
        }
        f_t.close();
    }

    for(std::map<std::string, GPUTexture*>::iterator it = textures.begin(); it != textures.end(); ++it)
    {
        delete it->second;
    }

    textures.clear();
    for(std::map<std::string, ComputePack*>::iterator it = computePacks.begin(); it != computePacks.end(); ++it)
    {
        delete it->second;
    }

    computePacks.clear();

    for(std::map<std::string, FeedbackBuffer*>::iterator it = feedbackBuffers.begin(); it != feedbackBuffers.end(); ++it)
    {
        delete it->second;
    }
    feedbackBuffers.clear();

    delete trajectory_manager;
    delete globalModel;
}

void HRBFFusion::LoadImagesAssociationFile(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps)
{
    std::ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());

    while(!fAssociation.eof())
    {
       std::string s;
       getline(fAssociation,s);
       if(!s.empty())
       {
           std::stringstream ss;
           ss << s;
           double t;
           std::string sD, sRGB;
           ss >> t;
           vTimestamps.push_back(t);
           ss >> sD;
           vstrImageFilenamesD.push_back(sD);
           ss >> t;
           //vTimestamps.push_back(t);
           ss >> sRGB;
           vstrImageFilenamesRGB.push_back(sRGB);
       }
    }
}

void HRBFFusion::inputFrame()
{
    //Read image and depthmap from association.txt file
    imRGB = cv::imread(GlobalStateParam::get().currentWorkingDirectory + "/" + vstrImageFilenamesRGB[tick -1], cv::IMREAD_UNCHANGED);
    imD = cv::imread(GlobalStateParam::get().currentWorkingDirectory + "/" + vstrImageFilenamesD[tick - 1], cv::IMREAD_UNCHANGED);

    double tframe = vTimestamps[tick -1];
    if(imRGB.empty())
    {
        cerr << endl << "Failed to load image at: "
             << string(GlobalStateParam::get().currentWorkingDirectory) << "/" << vstrImageFilenamesRGB[tick - 1] << endl;
        exit(0);
    }

    mImGray = imRGB;
    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }
    cvtColor(imRGB, imRGB, CV_RGB2BGR);
}

void HRBFFusion::CheckReplacedInLastFrame()
{
    for(int i =0; i<mLastFrame.N; i++)
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i];

        if(pMP)
        {
            MapPoint* pRep = pMP->GetReplaced();
            if(pRep)
            {
                mLastFrame.mvpMapPoints[i] = pRep;
            }
        }
    }
}

bool HRBFFusion::NeedNewSubMap()
{
    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;
    const int nKFs = spMap->KeyFramesInMap();

    if(mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
        return false;

    //Local Mapping accept keyframes
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

    //every ten frames
    insertAsubmap = ((tick - 1) % 10 == 0);
    //adaptive add New submaps
//    Eigen::Matrix4f diff = currPose.inverse() * lastKeyFramePose;
//    Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
//    Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);

//    std::cout << "translation: " << diffTrans.norm() << " rotation: " << rodrigues2(diffRot).norm() << "\n";
//    if(diffTrans.norm() > 0.1 || rodrigues2(diffRot).norm() > (20 * 3.14159265f / 180))
//    {
//        insertAsubmap = true;
//    }else
//    {
//        insertAsubmap = false;
//    }
    if(insertAsubmap)
    {
        //if local mapping is idle, then insert current frame as KF
        if(bLocalMappingIdle)
        {
            return true;
        }else {
            //if local mapping is doing the BA, then interrupt it
            mpLocalMapper->InterruptBA();
            //if number of keyframes in queue is less than 3, insert a new keyframe;
            if(mpLocalMapper->KeyframesInQueue() < 3)
                return true;
            else {
                return false;
            }
        }
    }else
        return false;
}

void HRBFFusion::ConstructSubmaps()
{
    if(!mpLocalMapper->SetNotStop(true))
        return;
    //update the pose of the last Keyframe after local BA
    unsigned long frameID = mLastFrame.mnId;
    cv::Mat poseTwc_last =  Converter::toCvMat(trajectory_manager->poses[frameID]);
    cv::Mat poseTcw_last =  Converter::toCvMatInvertTranformation(poseTwc_last);
    mLastFrame.SetPose(poseTcw_last);

    //Update current Frame Pose
    cv::Mat poseTwc =  Converter::toCvMat(currPose);
    cv::Mat poseTcw =  Converter::toCvMatInvertTranformation(poseTwc);
    //set current frame pose;
    mCurrentFrame.SetPose(poseTcw);

    //find matches from last KeyFrame
    fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint*>(nullptr));
    ORBmatcher matcher(0.9);
    int th = 7.0;
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame, th, false);    
    std::cout << "find " << nmatches << " matches before local search" << endl;
    //If few matches, uses a wider window search
    if(nmatches < 20)
    {
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(nullptr));
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,false);
        std::cout << "no enough local map points found with motion model, enlarge the search radius\n";
    }

//    std::cout << "find " << nmatches << " matches" << endl;
    //find matches in local keyframes with covisibility
    UpdateLocalKeyFrames();
    UpdateLocalPoints();
    SearchLocalPoints();

    std::cout << "find " << nmatches << " matches after local search" << endl;
    //search all previous Key features
//    SearchAllPoints();

    //Keyframe initilization
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,spMap,mpKeyFrameDB);

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.s
    vector<pair<float,int> > vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        float z = mCurrentFrame.mvDepth[i];
        if(z>0)
        {
            vDepthIdx.push_back(make_pair(z,i));
        }
    }

    if(!vDepthIdx.empty())
    {
        sort(vDepthIdx.begin(),vDepthIdx.end());

        int nPoints = 0;

        for(size_t j=0; j<vDepthIdx.size(); j++)
        {
            //gain index, map points with depth less than depthThreshold(tipically 3m) will be created
            int i = vDepthIdx[j].second;
            bool bCreateNew = false;

            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            //if no associated MapPoints found in global map, add a new one
            if(!pMP)
                bCreateNew = true;
            else if(pMP->Observations()<1)
            {
                bCreateNew = true;
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(nullptr);
            }

            //if there is no global map point associated with the key points in the current frame
            //add the landmark as new
            if(bCreateNew)
            {
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKF,spMap);
                pNewMP->AddObservation(pKF,i);
                pKF->AddMapPoint(pNewMP,i);
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                spMap->AddMapPoint(pNewMP);
                mCurrentFrame.mvpMapPoints[i]=pNewMP;
                nPoints++;
            }else {
                nPoints++;
            }
            if(vDepthIdx[j].first>mThDepth && nPoints>100)
                break;
        }
    }
    mpLocalMapper->InsertKeyFrame(pKF);
    mpLocalMapper->SetNotStop(false);

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;

    mLastFrame = ORB_SLAM2::Frame(mCurrentFrame);
}

void HRBFFusion::UpdateDenseGlobalModel()
{
    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(spMap->mMutexMapUpdate);

    //obtain all keyframes in sparse map, sort them
    vector<KeyFrame*> all_KF = spMap->GetAllKeyFrames();
    sort(all_KF.begin(),all_KF.end(),KeyFrame::lId);

    //Delta T for Global model points
    globalModel->DeltaTransformKF.clear();
    globalModel->DeltaTransformKF.reserve(all_KF.size());

    //Update camera trajectory (till second last keyframe)
    for (int i = 0; i < all_KF.size() - 1; i++) {
        unsigned long int frameId = all_KF[i]->mnFrameId;
//        std::cout << "frameId: " << frameId << std::endl;
        cv::Mat Twc = all_KF[i]->GetPoseInverse();
        Eigen::Matrix4f poseKF = Converter::toMatrix4f(Twc);
        Eigen::Matrix4f pose_relative = poseKF * trajectory_manager->poses[frameId].inverse();
        globalModel->DeltaTransformKF.push_back(pose_relative);

        //update all frame poses after keyframe (e.g. 1, 2, 3, 4...9)
        unsigned long int frameIdNext = all_KF[i + 1]->mnFrameId;
        for (int j = frameId + 1; j < frameIdNext; j++){
            //camera pose update
//            std::cout << "Up data traj frameId: " << j << std::endl;
            Eigen::Matrix4f poseToKey =  trajectory_manager->poses[j] * trajectory_manager->poses[frameId].inverse();
            trajectory_manager->poses[j] = poseToKey * poseKF;
        }
        //update KeyFrame poses (e.g. 0, 10...) !!!should be updated here!!!
        trajectory_manager->poses[frameId] = poseKF;
    }

    //Update last KeyFrame (last key frame pose)
    int lastKeyframeIndex = all_KF.size() - 1;
    cv::Mat Twc = all_KF[lastKeyframeIndex]->GetPoseInverse();
    Eigen::Matrix4f poseKF = Converter::toMatrix4f(Twc);
    unsigned long int lastKeyframeID = all_KF[lastKeyframeIndex]->mnFrameId;
    Eigen::Matrix4f pose_relative = poseKF * trajectory_manager->poses[lastKeyframeID].inverse();
    globalModel->DeltaTransformKF.push_back(pose_relative);

    std::cout << "lastKeyframeID frameId: " << lastKeyframeID << std::endl;
    std::cout << "trajectory_manager poses size: " << trajectory_manager->poses.size() << std::endl;
    std::cout << "tick: " << tick << std::endl;
    for (int i = lastKeyframeID + 1; i < tick; i++) {
        Eigen::Matrix4f poseToKey =  trajectory_manager->poses[i] * trajectory_manager->poses[lastKeyframeID].inverse();
        trajectory_manager->poses[i] = poseToKey * poseKF;
    }
    trajectory_manager->poses[lastKeyframeID] = poseKF;
    //update global model
    globalModel->updateModel();

    //update last keyframe pose,
    lastKeyFramePose = trajectory_manager->poses[lastKeyframeID];

    //update current pose, pose relative (if Global BA the pose shoud be tick - 2)
    currPose = trajectory_manager->poses[tick - 1];
}


void HRBFFusion::SparseMapRGBDInitilization()
{
    //process a frame first, extract key points and undistorted key points
    PreProcessFrame(tick - 1);

    //detect more than 500 key points, validate initialization
    if(mCurrentFrame.N > 500)
    {
        // Set first Frame pose to the origin,  identity matrix
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

        //Create KeyFrame
        //construct a new keyframe for submap
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame, spMap, mpKeyFrameDB);

        // Insert KeyFrame in the map
        spMap->AddKeyFrame(pKFini);

        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++)
        {              
            float z = mCurrentFrame.mvDepth[i];
            if(z>0)
            {
                //3D coordinates(position)
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,spMap);
                //each map point has the observation, mark it
                pNewMP->AddObservation(pKFini,i);
                //associate map point to the keyframe
                pKFini->AddMapPoint(pNewMP,i);
                //to find the best descriptor for fast marching
                pNewMP->ComputeDistinctiveDescriptors();
                pNewMP->UpdateNormalAndDepth();
                //add map to the sparse feature map
                spMap->AddMapPoint(pNewMP);
                mCurrentFrame.mvpMapPoints[i]=pNewMP;
            }
        }

        //init local map
        mpLocalMapper->InsertKeyFrame(pKFini);

        mLastFrame = ORB_SLAM2::Frame(mCurrentFrame);
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFini;

        //local KeyFrames for tracking
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = spMap->GetAllMapPoints();
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        spMap->SetReferenceMapPoints(mvpLocalMapPoints);
        spMap->mvpKeyFrameOrigins.push_back(pKFini);
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
    }
}

void HRBFFusion::PreProcessFrame(int id)
{
    cv::Mat imDepth = imD;
    if((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
        imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);
    tframe = vTimestamps[id];

    //preprocessing for an input frame, extract ORB features
    mCurrentFrame = ORB_SLAM2::Frame(mImGray, imDepth, tframe, mpORBextractorLeft, mpORBVocabulary, mK, mDistCoef, mbf, mThDepth);

    //!!USE filtered depth map
    Img<float> image_DepthF(Resolution::getInstance().rows(), Resolution::getInstance().cols());
    textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture->Download(image_DepthF.data, GL_LUMINANCE, GL_FLOAT);
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        //find the depth i filtered depth map;
        int index_x = static_cast<int>(mCurrentFrame.mvKeysUn[i].pt.x);  //cols
        int index_y = static_cast<int>(mCurrentFrame.mvKeysUn[i].pt.y);  //rows
        //discard index exceeds the original limit (after undistortion);
        if(index_x < 0 || index_y < 0 || index_x > Resolution::getInstance().cols() || index_y > Resolution::getInstance().rows())
            continue;
        float d = image_DepthF.at<float>(index_y, index_x);              //rows and cols
        if(d > 0.3)
            mCurrentFrame.mvDepth[i] = d;
        else
            mCurrentFrame.mvDepth[i] = -1;
    }

    //compute covariance matrix / information matrix
    mCurrentFrame.info.resize(mCurrentFrame.N);
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvDepth[i] < 0)
        {
            mCurrentFrame.info[i] = Eigen::Matrix3d::Zero();
                continue;
        }
        //find the depth i filtered depth map;
        int index_x = static_cast<int>(mCurrentFrame.mvKeys[i].pt.x);
        int index_y = static_cast<int>(mCurrentFrame.mvKeys[i].pt.y);

        int R = 6;
        int D = R * 2 + 1;

        int tx = min(index_x - D / 2 + D, Resolution::getInstance().width());
        int ty = min(index_y - D / 2 + D, Resolution::getInstance().height());

        int count = 0;
        double sum_x = 0, sum_y = 0, sum_z = 0;
        for(int cx = max(index_x - D / 2, 0); cx < tx; ++cx)
        {
            for(int cy = max(index_y - D / 2, 0); cy < ty; ++cy)
            {
                float depth = 1000 * image_DepthF.at<float>(cy, cx);
                if(depth > 300)
                {
                    const double x = (cx - mCurrentFrame.cx) * depth * mCurrentFrame.invfx;
                    const double y = (cy - mCurrentFrame.cy) * depth * mCurrentFrame.invfy;
                    const double z = depth;

                    sum_x+=x;
                    sum_y+=y;
                    sum_z+=z;

                    count++;
                }

            }
        }

        if(count == 0)
        {
            mCurrentFrame.info[i] = Eigen::Matrix3d::Zero();
              continue;
        }
        //compute average
        double average_x = sum_x / static_cast<double>(count);
        double average_y = sum_y / static_cast<double>(count);
        double average_z = sum_z / static_cast<double>(count);

        double Cosum_x = 0, Cosum_y = 0, Cosum_z = 0;
        //compute Covariance Matrix
        for(int cx = max(index_x - D / 2, 0); cx < tx; ++cx)
        {
            for(int cy = max(index_y - D / 2, 0); cy < ty; ++cy)
            {
                float depth = 1000 * image_DepthF.at<float>(cy, cx);
                if(depth > 300)
                {
                    const double x = (cx - mCurrentFrame.cx) * depth * mCurrentFrame.invfx;
                    const double y = (cy - mCurrentFrame.cy) * depth * mCurrentFrame.invfy;
                    const double z = depth;

                    Cosum_x += (x - average_x) * (x - average_x);
                    Cosum_y += (y - average_y) * (y - average_y);
                    Cosum_z += (z - average_z) * (z - average_z);
                }

            }
        }

        double cov_x = Cosum_x / static_cast<double>(count);
        double cov_y = Cosum_y / static_cast<double>(count);
        double cov_z = Cosum_z / static_cast<double>(count);

        mCurrentFrame.info[i] = Eigen::Matrix3d::Identity();
        mCurrentFrame.info[i](0, 0) = 1.0 / cov_x;
        mCurrentFrame.info[i](1, 1) = 1.0 / cov_y;
        mCurrentFrame.info[i](2, 2) = 1.0 / cov_z;
    }
    //set the current frame ID
    mCurrentFrame.mnId = id;
}

void HRBFFusion::LoadCameraParaAndInitORBExtractor()
{
    // Load camera parameters from opencv settings file
    cv::FileStorage fSettings(GlobalStateParam::get().parameterFileCvFormat, cv::FileStorage::READ);

    //load camera intrinsics
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    //camera intrinsic
    cv::Mat K1 = cv::Mat::eye(3, 3, CV_32F);

    K1.at<float>(0,0) = fx;
    K1.at<float>(1,1) = fy;
    K1.at<float>(0,2) = cx;
    K1.at<float>(1,2) = cy;

    K1.copyTo(mK);

    //distortion coefficients, [k1, k2, p1, p2, k3], k3 for large distortion
    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    //camera baseline in pixels
    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;

    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters
    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    //initialize ORB extractor for keyframe feature extraction
    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;

    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;

    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    //according to the camera baseline
    mThDepth = mbf*(float)fSettings["ThDepth"]/fx;

    //factor for depth value
    mDepthMapFactor = fSettings["DepthMapFactor"];

    std::cout << endl << "- Current mDepthMapFactor: " << mDepthMapFactor << std::endl;

    if(fabs(mDepthMapFactor)<1e-5)
        mDepthMapFactor=1;
    else
        //inverse for further usage
        mDepthMapFactor = 1.0f/mDepthMapFactor;
}

void HRBFFusion::createTextures()
{
    textures[GPUTexture::RGB] = new GPUTexture(Resolution::getInstance().width(),
                                               Resolution::getInstance().height(),
                                               GL_RGBA,
                                               GL_RGB,
                                               GL_UNSIGNED_BYTE,
                                               true,
                                               true);

    textures[GPUTexture::DEPTH_RAW] = new GPUTexture(Resolution::getInstance().width(),
                                                     Resolution::getInstance().height(),
                                                     GL_LUMINANCE16UI_EXT,
                                                     GL_LUMINANCE_INTEGER_EXT,
                                                     GL_UNSIGNED_SHORT);

    textures[GPUTexture::DEPTH_FILTERED] = new GPUTexture(Resolution::getInstance().width(),
                                                          Resolution::getInstance().height(),
                                                          GL_LUMINANCE32F_ARB,
                                                          GL_LUMINANCE,
                                                          GL_FLOAT,
                                                          false,
                                                          true);

    textures[GPUTexture::DEPTH_METRIC] = new GPUTexture(Resolution::getInstance().width(),
                                                        Resolution::getInstance().height(),
                                                        GL_LUMINANCE32F_ARB,
                                                        GL_LUMINANCE,
                                                        GL_FLOAT);

    textures[GPUTexture::DEPTH_METRIC_FILTERED] = new GPUTexture(Resolution::getInstance().width(),
                                                                 Resolution::getInstance().height(),
                                                                 GL_LUMINANCE32F_ARB,
                                                                 GL_LUMINANCE,
                                                                 GL_FLOAT);

    textures[GPUTexture::DEPTH_NORM] = new GPUTexture(Resolution::getInstance().width(),
                                                      Resolution::getInstance().height(),
                                                      GL_LUMINANCE,
                                                      GL_LUMINANCE,
                                                      GL_FLOAT,
                                                      true);
     textures[GPUTexture::VERTEX_RAW] = new GPUTexture(Resolution::getInstance().width(),
                                                     Resolution::getInstance().height(),
                                                     GL_RGBA32F,
                                                     GL_LUMINANCE,
                                                     GL_FLOAT,
                                                     false,
                                                     true);

    textures[GPUTexture::VERTEX_FILTERED] = new GPUTexture(Resolution::getInstance().width(),
                                                     Resolution::getInstance().height(),
                                                     GL_RGBA32F,
                                                     GL_LUMINANCE,
                                                     GL_FLOAT,
                                                     false,
                                                     true);

    textures[GPUTexture::NORMAL] = new GPUTexture(Resolution::getInstance().width(),
                                                     Resolution::getInstance().height(),
                                                     GL_RGBA32F,
                                                     GL_LUMINANCE,
                                                     GL_FLOAT,
                                                     false,
                                                     true);
    //Debug. for radius change
    textures[GPUTexture::NORMAL_OPT] = new GPUTexture(Resolution::getInstance().width(),
                                                     Resolution::getInstance().height(),
                                                     GL_RGBA32F,
                                                     GL_LUMINANCE,
                                                     GL_FLOAT,
                                                     false,
                                                     true);

    textures[GPUTexture::PRINCIPAL_CURV1] = new GPUTexture(Resolution::getInstance().width(),
                                                     Resolution::getInstance().height(),
                                                     GL_RGBA32F,
                                                     GL_LUMINANCE,
                                                     GL_FLOAT,
                                                     false,
                                                     true);
    textures[GPUTexture::PRINCIPAL_CURV2] = new GPUTexture(Resolution::getInstance().width(),
                                                     Resolution::getInstance().height(),
                                                     GL_RGBA32F,
                                                     GL_LUMINANCE,
                                                     GL_FLOAT,
                                                     false,
                                                     true);
    textures[GPUTexture::GRADIENT_MAG] = new GPUTexture(Resolution::getInstance().width(),
                                                      Resolution::getInstance().height(),
                                                      GL_LUMINANCE32F_ARB,
                                                      GL_LUMINANCE,
                                                      GL_FLOAT,
                                                      false,
                                                      true);

    textures[GPUTexture::RADIUS] = new GPUTexture(Resolution::getInstance().width(),
                                                     Resolution::getInstance().height(),
                                                      GL_LUMINANCE32F_ARB,
                                                      GL_LUMINANCE,
                                                      GL_FLOAT,
                                                      false,
                                                      true);

    textures[GPUTexture::CONFIDENCE] = new GPUTexture(Resolution::getInstance().width(),
                                                     Resolution::getInstance().height(),
                                                      GL_LUMINANCE32F_ARB,
                                                      GL_LUMINANCE,
                                                      GL_FLOAT,
                                                      false,
                                                      true);
}

void HRBFFusion::createCompute()
{
    computePacks[ComputePack::NORM] = new ComputePack(loadProgramFromFile("empty.vert", "depth_norm.frag", "quad.geom"),
                                                      textures[GPUTexture::DEPTH_NORM]->texture);

    if(GlobalStateParam::get().preprocessingUsebilateralFilter)
    {
        computePacks[ComputePack::FILTER] = new ComputePack(loadProgramFromFile("empty.vert", "depth_bilateral.frag", "quad.geom"),
                                                        textures[GPUTexture::DEPTH_FILTERED]->texture);
        std::cout << "using bilateral filtering" << std::endl;
    }
    else{
        computePacks[ComputePack::FILTER] = new ComputePack(loadProgramFromFile("empty.vert", "depth_guass.frag", "quad.geom"),
                                                            textures[GPUTexture::DEPTH_FILTERED]->texture);
        std::cout << "using guass filtering" << std::endl;
    }
    computePacks[ComputePack::METRIC] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric_raw.frag", "quad.geom"),
                                                        textures[GPUTexture::DEPTH_METRIC]->texture);

    computePacks[ComputePack::METRIC_FILTERED] = new ComputePack(loadProgramFromFile("empty.vert", "depth_metric_filtered.frag", "quad.geom"),
                                                                 textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture);

    computePacks[ComputePack::VERTEX_NORMAL_RADIUS] = new ComputePack(loadProgramFromFile("empty.vert", "depth_vertex_normal_radius.frag", "quad.geom"),
                                                           textures[GPUTexture::VERTEX_RAW]->texture,
                                                           textures[GPUTexture::VERTEX_FILTERED]->texture,
                                                           textures[GPUTexture::NORMAL]->texture,
                                                           textures[GPUTexture::RADIUS]->texture);

    computePacks[ComputePack::CURVATURE] = new ComputePack(loadProgramFromFile("empty.vert", "depth_curvature_gradient.frag", "quad.geom"),
                                                           textures[GPUTexture::PRINCIPAL_CURV1]->texture, textures[GPUTexture::PRINCIPAL_CURV2]->texture,
                                                           textures[GPUTexture::GRADIENT_MAG]->texture, textures[GPUTexture::NORMAL_OPT]->texture);

    computePacks[ComputePack::UPDATE_NORMALRAD] = new ComputePack(loadProgramFromFile("empty.vert", "depth_update_normalrad.frag", "quad.geom"),
                                                           textures[GPUTexture::NORMAL]->texture);

    computePacks[ComputePack::CONFIDENCE_EVALUATION] = new ComputePack(loadProgramFromFile("empty.vert", "depth_confidence_evaluation.frag", "quad.geom"),
                                                           textures[GPUTexture::CONFIDENCE]->texture);
}

void HRBFFusion::createFeedbackBuffers()
{
    feedbackBuffers[FeedbackBuffer::RAW] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback_raw.vert", "vertex_feedback_raw.geom"));
    feedbackBuffers[FeedbackBuffer::FILTERED] = new FeedbackBuffer(loadProgramGeomFromFile("vertex_feedback_filtered.vert", "vertex_feedback_filtered.geom"));
}

void HRBFFusion::computeFeedbackBuffers()
{
    //To vertex buffers position confidence + normal radius + color time
    feedbackBuffers[FeedbackBuffer::RAW]->compute(textures[GPUTexture::RGB]->texture,
                                                  textures[GPUTexture::DEPTH_METRIC]->texture,
                                                  tick,
                                                  maxDepthProcessed);

    feedbackBuffers[FeedbackBuffer::FILTERED]->compute(textures[GPUTexture::RGB]->texture,
                                                       textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture,
                                                       tick,
                                                       maxDepthProcessed);
}

bool HRBFFusion::denseEnough(const Img<Eigen::Matrix<unsigned char, 3, 1>> & img)
{
    int sum = 0;

    for(int i = 0; i < img.rows; i++)
    {
        for(int j = 0; j < img.cols; j++)
        {
            sum += img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(0) > 0 &&
                   img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(1) > 0 &&
                   img.at<Eigen::Matrix<unsigned char, 3, 1>>(i, j)(2) > 0;
        }
    }

    float per = float(sum) / float(img.rows * img.cols);
    //std::cout << "valid pixel percentage: " << per << std::endl;
    return per > GlobalStateParam::get().globalDenseEnoughThresh;
}

bool HRBFFusion::denseEnough(const Img<Eigen::Vector4f> & vertices)
{
    int sum = 0;

    for(int i = 0; i < vertices.rows; i++)
    {
        for(int j = 0; j < vertices.cols; j++)
        {
            sum += vertices.at<Eigen::Vector4f>(i, j)(2) > 0;
        }
    }

    float per = float(sum) / float(vertices.rows * vertices.cols);
    return per > GlobalStateParam::get().globalDenseEnoughThresh;
}


void HRBFFusion::processFrame(const unsigned char * rgb,
                              const unsigned short * depth,
                              const int64_t & timestamp,
                              const float weightMultiplier)
{

    //load images
    if(GlobalStateParam::get().sensorType == 3)
    {
        if(tick > vstrImageFilenamesD.size())
        {
            tick++;
            return;
        }
        inputFrame();
        textures[GPUTexture::DEPTH_RAW]->texture->Upload(imD.data, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
        textures[GPUTexture::RGB]->texture->Upload(imRGB.data, GL_RGB, GL_UNSIGNED_BYTE);
    }else if(GlobalStateParam::get().sensorType == 2){
        textures[GPUTexture::DEPTH_RAW]->texture->Upload(depth, GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
        textures[GPUTexture::RGB]->texture->Upload(rgb, GL_RGB, GL_UNSIGNED_BYTE);
    }else {
        std::cout << "please input the specified data format" << std::endl;
    }

    //Preprocessing:
    TICK("Initialization");
    filterDepth(mDepthMapFactor);
    metriciseDepth(mDepthMapFactor);
    computeVertexNormalRadius();
    computeCurvatureGradient();
    updateNormalRad();
    TOCK("Initialization");

    //initial first frame weighting
    Eigen::Matrix4f lastPose;
    insertSubmap = false;

    //RGB-D odometry
    //initilization for Global dense/sparse map
    if(tick == 1)
    {
        //sparse map initialization
        if(GlobalStateParam::get().optimizationUseLocalBA || GlobalStateParam::get().optimizationUseGlobalBA)
        {
            //Get Map Mutex -> Map cannot be changed
            unique_lock<mutex> lock(spMap->mMutexMapUpdate);
            //sparse map initilization
            SparseMapRGBDInitilization();            
            mpFrameDrawer->Update(this);
        }

        //global dense model initialization
        globalModel->initialise(textures[GPUTexture::VERTEX_RAW],
                                textures[GPUTexture::NORMAL],
                                textures[GPUTexture::RGB],
                                textures[GPUTexture::PRINCIPAL_CURV1],
                                textures[GPUTexture::PRINCIPAL_CURV2],
                                textures[GPUTexture::GRADIENT_MAG],
                                currPose);

        //tracking initilization
        frameToModel.initFirstRGB(textures[GPUTexture::RGB]);

        //initialize first submap ID
        indexMap.lActiveKFID.push_back(indexSubmap);

        globalModel->lActiveKFID.push_back(indexSubmap);

        //initialize first pose of the trajectory
        trajectory_manager->poses.push_back(Eigen::Matrix4f::Identity());
    }else
    {
        TICK("Registration");
        lastPose = currPose;
        bool trackingOk = true;
        if(!GlobalStateParam::get().globalInputLoadTrajectory)
        {
            //see if the predicted texture is not dense enough
            resize.vertex(indexMap.vertexTexHRBF(), verticesBuff);
            bool shouldFillIn = !denseEnough(verticesBuff);

            //WARNING initICP* must be called before initRGB*
            frameToModel.initICPModel(shouldFillIn ? &fillIn.vertexTexture : indexMap.vertexTexHRBF(),
                                      shouldFillIn ? &fillIn.normalTexture : indexMap.normalTexHRBF(),
                                      maxDepthProcessed, currPose);
            frameToModel.initRGBModel((shouldFillIn || frameToFrameRGB) ? &fillIn.imageTexture : indexMap.imageTexHRBF());
            frameToModel.initCurvatureModel(shouldFillIn ? &fillIn.curvk1Texture : indexMap.curvk1TexHRBF(),
                                            shouldFillIn ? &fillIn.curvk2Texture : indexMap.curvk2TexHRBF(),
                                            currPose);
//            frameToModel.initICP(textures[GPUTexture::DEPTH_FILTERED], maxDepthProcessed, mDepthMapFactor);
            frameToModel.initICP(textures[GPUTexture::VERTEX_FILTERED], textures[GPUTexture::NORMAL], maxDepthProcessed);
            frameToModel.initRGB(textures[GPUTexture::RGB]);
            frameToModel.initCurvature(textures[GPUTexture::PRINCIPAL_CURV1], textures[GPUTexture::PRINCIPAL_CURV2]);

            //icpweight map
            frameToModel.initICPweight(shouldFillIn ? &fillIn.icpweightTexture : indexMap.icpweightTexHRBF());

            Eigen::Vector3f trans = currPose.topRightCorner(3, 1);
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> rot = currPose.topLeftCorner(3, 3);

            //frame-to-model odometry
            frameToModel.getIncrementalTransformation(trans,
                                                      rot,
                                                      rgbOnly,
                                                      icpWeight,
                                                      pyramid,
                                                      fastOdom,
                                                      so3,
                                                      GlobalStateParam::get().registrationICPUseWeightedICP,
                                                      tick - 1);

            currPose.topRightCorner(3, 1) = trans;
            currPose.topLeftCorner(3, 3) = rot;
        }
        else
        {
            currPose = trajectory_manager->poses[tick - 1];
        }
        TOCK("Registration");

        //Weight by velocity
        Eigen::Matrix4f diff = currPose.inverse() * lastPose;
        Eigen::Vector3f diffTrans = diff.topRightCorner(3, 1);
        Eigen::Matrix3f diffRot = diff.topLeftCorner(3, 3);
        weighting = std::max(diffTrans.norm(), rodrigues2(diffRot).norm());

        float largest = 0.01;
        float minWeight = 0.5;
        if(weighting > largest)
        {
            weighting = largest;
        }
        weighting = std::max(1.0f - (weighting / largest), minWeight) * weightMultiplier;

        //confidence evaluation
        VertexConfidence(weighting);

        //save current pose to trajecotry
        if(!GlobalStateParam::get().globalInputLoadTrajectory)
        {
            trajectory_manager->poses.push_back(currPose);
            trajectory_manager->timstamp.push_back(timestamp);
        }

        //Local BA, KeyFrame Insertion
        if(GlobalStateParam::get().optimizationUseLocalBA || GlobalStateParam::get().optimizationUseGlobalBA)
        {

            if(NeedNewSubMap())
            {
                //preprocess the current frame,
                //(feature extraction, map undistortion, assign feature to grid)
                {
                    std::cout << "constructing new submap..." << std::endl;
                    unique_lock<mutex> lock(spMap->mMutexMapUpdate);
                    CheckReplacedInLastFrame();
                    PreProcessFrame(tick - 1);
                    ConstructSubmaps();
                }
                indexSubmap++;
                insertSubmap = true;

                //wait local BA to process
                std::unique_lock <std::mutex> lck(updateModel);
                while (condVar.wait_for(lck,std::chrono::seconds(1))==std::cv_status::timeout) {
                  std::cout << "waiting for Local BA over 1s..." << std::endl;
                }

                //obtain current active submaps to conduct tracking and fusing
                std::vector<int> LKID;
                LKID.reserve(1000);
                for(int i=0; i< mvpLocalKeyFrames.size(); i++)
                {
                    LKID.push_back(mvpLocalKeyFrames[i]->mnId);
                }
                indexMap.lActiveKFID = LKID;
                indexMap.lActiveKFID.push_back(indexSubmap);

                globalModel->lActiveKFID = LKID;
                globalModel->lActiveKFID.push_back(indexSubmap);

                //update drawer
                mpFrameDrawer->Update(this);

                //update dense global model and camera trajectory
                UpdateDenseGlobalModel();
                std::cout << "Update global map done\n" << '\n';
            }

            //if loop is detected and optimized
            if(checkPoseGraphOptimization())
            {
                std::cout << "loop detected waiting for Global BA" << std::endl;
                mpFrameDrawer->Update(this);
                //update dense global model and camera trajectory
                UpdateDenseGlobalModel();
                mbPoseGragh = false;
            }
        }

        //Depth Fusion
        if(!rgbOnly && trackingOk && !lost)
        {
            //std::unique_lock<mutex> lock1(UpdateGlobalModel);         
            indexMap.predictIndices(currPose, tick, tick, globalModel->model(), maxDepthProcessed, insertSubmap, indexSubmap);
            TICK("Integration");
            globalModel->fuse(currPose,
                             tick,
                             textures[GPUTexture::RGB],
                             textures[GPUTexture::DEPTH_METRIC],
                             textures[GPUTexture::DEPTH_METRIC_FILTERED],
                             textures[GPUTexture::PRINCIPAL_CURV1],
                             textures[GPUTexture::PRINCIPAL_CURV2],
                             textures[GPUTexture::CONFIDENCE],
                             indexMap.indexTex(),
                             indexMap.vertConfTex(),
                             indexMap.colorTimeTex(),
                             indexMap.normalRadTex(),
                             maxDepthProcessed,
                             confidenceThreshold,
                             weighting,
                             insertSubmap,
                             indexSubmap);
            TOCK("Integration");
            indexMap.predictIndices(currPose, tick, tick, globalModel->model(), maxDepthProcessed, insertSubmap, indexSubmap);
            //free space violation may not be valid when we obtain the canbinet-like objects
            //where there will be missing regions
            globalModel->clean(currPose,
                              tick,
                              indexMap.indexTex(),
                              indexMap.vertConfTex(),
                              indexMap.colorTimeTex(),
                              indexMap.normalRadTex(),
                              indexMap.depthTex(),
                              confidenceThreshold,
                              maxDepthProcessed);
        }
    }

    poseGraph.push_back(std::pair<unsigned long long int, Eigen::Matrix4f>(tick, currPose));
    poseLogTimes.push_back(timestamp);

    predict(weighting);

    if(!lost)
    {
       tick++;
    }

    processingTimes.push_back(Stopwatch::getInstance().getTimings());
}


void HRBFFusion::predict(float weighting)
{
    //HRBF-based prediction
    indexMap.predictIndices(currPose, tick, tick, globalModel->model(), maxDepthProcessed, insertSubmap, indexSubmap);
    TICK("Prediction");
    indexMap.predictHRBF(IndexMap::ACTIVE);
    TOCK("Prediction");

    //Fill in vertex maps
    fillIn.vertex(indexMap.vertexTexHRBF(), textures[GPUTexture::VERTEX_FILTERED],indexMap.icpweightTexHRBF(),
                  textures[GPUTexture::PRINCIPAL_CURV1], textures[GPUTexture::PRINCIPAL_CURV2],
                  textures[GPUTexture::CONFIDENCE], tick, weighting, lost);
    fillIn.normal(indexMap.normalTexHRBF(), textures[GPUTexture::NORMAL], lost);
    fillIn.curvature(indexMap.curvk1TexHRBF(), indexMap.curvk2TexHRBF(),
                     textures[GPUTexture::PRINCIPAL_CURV1], textures[GPUTexture::PRINCIPAL_CURV2], lost);
    fillIn.image(indexMap.imageTexHRBF(), textures[GPUTexture::RGB], lost || frameToFrameRGB);
}


void HRBFFusion::metriciseDepth(float depthFactor)
{
    std::vector<Uniform> uniforms;
    uniforms.push_back(Uniform("maxD", depthCutoff));
    uniforms.push_back(Uniform("depthFactor", depthFactor));
    computePacks[ComputePack::METRIC]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
    computePacks[ComputePack::METRIC_FILTERED]->compute(textures[GPUTexture::DEPTH_FILTERED]->texture, &uniforms);
}

void HRBFFusion::filterDepth(float depthFactor)
{
    std::vector<Uniform> uniforms;
    uniforms.push_back(Uniform("cols", static_cast<float>(Resolution::getInstance().cols())));
    uniforms.push_back(Uniform("rows", static_cast<float>(Resolution::getInstance().rows())));
    uniforms.push_back(Uniform("maxD", depthCutoff));
    uniforms.push_back(Uniform("depthFactor", depthFactor));
    computePacks[ComputePack::FILTER]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
}

void HRBFFusion::computeCurvatureGradient()
{
    std::vector<Uniform> uniforms;
    uniforms.push_back(Uniform("cols", static_cast<float>(Resolution::getInstance().cols())));
    uniforms.push_back(Uniform("rows", static_cast<float>(Resolution::getInstance().rows())));
    uniforms.push_back(Uniform("cam",  Eigen::Vector4f(Intrinsics::getInstance().cx(),
                                                      Intrinsics::getInstance().cy(),
                                                      1.0 / Intrinsics::getInstance().fx(),
                                                      1.0 / Intrinsics::getInstance().fy())));
    //in metric-Depth, we have utilized the maxD to filter the value outside the range
    uniforms.push_back(Uniform("maxD", depthCutoff));
    uniforms.push_back(Uniform("winMultiply", GlobalStateParam::get().preprocessingCurvEstimationWindow));

    uniforms.push_back(Uniform("VertexFiltered", 0));
    uniforms.push_back(Uniform("NormalRadSampler", 1));

    computePacks[ComputePack::CURVATURE]->compute_2input(textures[GPUTexture::VERTEX_FILTERED]->texture,
                                                         textures[GPUTexture::NORMAL]->texture, &uniforms);
}
void HRBFFusion::updateNormalRad()
{
    std::vector<Uniform> uniforms;
    uniforms.push_back(Uniform("cols", static_cast<float>(Resolution::getInstance().cols())));
    uniforms.push_back(Uniform("rows", static_cast<float>(Resolution::getInstance().rows())));
    uniforms.push_back(Uniform("NormalOptSampler", 0));
    uniforms.push_back(Uniform("VertexSampler", 1));
    computePacks[ComputePack::UPDATE_NORMALRAD]->compute_2input(textures[GPUTexture::NORMAL_OPT]->texture,
                                                                textures[GPUTexture::VERTEX_FILTERED]->texture, &uniforms);
}
void HRBFFusion::VertexConfidence(float weighting)
{
    std::vector<Uniform> uniforms;
    uniforms.push_back(Uniform("cols", static_cast<float>(Resolution::getInstance().cols())));
    uniforms.push_back(Uniform("rows", static_cast<float>(Resolution::getInstance().rows())));
    uniforms.push_back(Uniform("cam",  Eigen::Vector4f(Intrinsics::getInstance().cx(),
                                                       Intrinsics::getInstance().cy(),
                                                       1.0 / Intrinsics::getInstance().fx(),
                                                       1.0 / Intrinsics::getInstance().fy())));
    uniforms.push_back(Uniform("useConfidenceEvaluation", static_cast<float>(GlobalStateParam::get().preprocessingUseConfEval)));
    uniforms.push_back(Uniform("epsilon", static_cast<float>(GlobalStateParam::get().preprocessingConfEvalEpsilon)));
    uniforms.push_back(Uniform("gradient_mag", 0));
    uniforms.push_back(Uniform("depthSampler", 1));
    uniforms.push_back(Uniform("weighting", weighting));
    computePacks[ComputePack::CONFIDENCE_EVALUATION]->compute_2input(textures[GPUTexture::GRADIENT_MAG]->texture,
                                                                     textures[GPUTexture::DEPTH_METRIC]->texture, &uniforms);
};

void HRBFFusion::computeVertexNormalRadius()
{
     std::vector<Uniform> uniforms;
     uniforms.push_back(Uniform("cols", static_cast<float>(Resolution::getInstance().cols())));
     uniforms.push_back(Uniform("rows", static_cast<float>(Resolution::getInstance().rows())));
     uniforms.push_back(Uniform("cam",  Eigen::Vector4f(Intrinsics::getInstance().cx(),
                                        Intrinsics::getInstance().cy(),
                                        1.0 / Intrinsics::getInstance().fx(),
                                        1.0 / Intrinsics::getInstance().fy())));
     uniforms.push_back(Uniform("depthRawSampler", 0));                           //for the filtered map;
     uniforms.push_back(Uniform("depthFilteredSampler", 1));
     uniforms.push_back(Uniform("radius_multiplier", GlobalStateParam::get().preprocessingInitRadiusMultiplier));
     uniforms.push_back(Uniform("PCAforNormalEstimation", GlobalStateParam::get().preprocessingNormalEstimationPCA));
     computePacks[ComputePack::VERTEX_NORMAL_RADIUS]->compute_2input(textures[GPUTexture::DEPTH_METRIC]->texture,
                                                                  textures[GPUTexture::DEPTH_METRIC_FILTERED]->texture,
                                                                  &uniforms);
}

void HRBFFusion::outliersRemoval()
{
//    std::vector<Uniform> uniforms;
//    uniforms.push_back(Uniform("cols", static_cast<float>(Resolution::getInstance().cols())));
//    uniforms.push_back(Uniform("rows", static_cast<float>(Resolution::getInstance().rows())));
//    uniforms.push_back(Uniform("maxD", depthCutoff));
//    computePacks[ComputePack::FILTERED_BY_MODEL_POINTS]->compute_filterByModel(textures[GPUTexture::DEPTH_METRIC]->texture,
//            indexMap.vertexTexHRBF()->texture, &uniforms);
}

void HRBFFusion::normaliseDepth(const float & minVal, const float & maxVal)
{
    std::vector<Uniform> uniforms;
    uniforms.push_back(Uniform("maxVal", maxVal / mDepthMapFactor));
    uniforms.push_back(Uniform("minVal", minVal / mDepthMapFactor));
    computePacks[ComputePack::NORM]->compute(textures[GPUTexture::DEPTH_RAW]->texture, &uniforms);
}

void HRBFFusion::UpdateLocalKeyFrames()
{
    //Each map point vote for the keyframes in which it has been observed
    //Keyframe + map point vote
    map<KeyFrame*,int> keyframeCounter;
    for(int i=0; i<mCurrentFrame.N; i++)
    {
        if(mCurrentFrame.mvpMapPoints[i])
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(!pMP->isBad())
            {
                //search all the previous Keyframe that have observe the current map point
                const map<KeyFrame*,size_t> observations = pMP->GetObservations();
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++;
            }
            else
            {
                mCurrentFrame.mvpMapPoints[i]=nullptr;
            }
        }
    }
    if(keyframeCounter.empty())
        return;

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(nullptr);

    mvpLocalKeyFrames.clear();                          //update local Key Frames
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size());

    // All keyframes that observe a map point are included in the local map.
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first;

        if(pKF->isBad())   //when use culling method in local mapping procedure
            continue;

        if(it->second>max) //select keyframe shares most points
        {
            max=it->second;
            pKFmax=pKF;
        }

        mvpLocalKeyFrames.push_back(it->first);
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId; //avoid duplication
    }

    // Include also some not-already-included keyframes that are Covisible to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80)
            break;

        KeyFrame* pKF = *itKF;

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF;
            if(!pNeighKF->isBad())
            {
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)   //avoid duplication
                {
                    mvpLocalKeyFrames.push_back(pNeighKF);
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds();
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit;
            if(!pChildKF->isBad())
            {
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId)   //avoid duplication
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break;
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent();
        if(pParent)
        {
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId)    //avoid duplication
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                break;
            }
        }

    }

    //set the reference KF as the one who share the most map points with current keyframe
    if(pKFmax)
    {
        mpReferenceKF = pKFmax;
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

void HRBFFusion::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear();

    //search local map points for each KeyFrames
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF;
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP;
            if(!pMP)
                continue;
            //skip already matched one
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId)
            {
                continue;
            }
            //for map point culling
            if(!pMP->isBad())
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId;
            }
        }
    }
}
void HRBFFusion::SearchLocalPoints()
{
    // Do not search map points already matched
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(nullptr);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;   //mark
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;  //This means the number of newly added matches

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        //already associated;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        //Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId + 2)
            th=5;
        int matches1 = matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
//        std::cout << "newly added matches: " << matches1 << std::endl;
    }
}

void HRBFFusion::SearchAllPoints()
{
    // Get Map Mutex -> Map cannot be changed
    //unique_lock<mutex> lock(spMap->mMutexMapUpdate);
    std::vector<MapPoint*> allMapPoints = spMap->GetAllMapPoints();

    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        if(pMP)
        {
            if(pMP->isBad())
            {
                *vit = static_cast<MapPoint*>(nullptr);
            }
            else
            {
                pMP->IncreaseVisible();
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;  //mark
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;  //This means the number of newly added matches
    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=allMapPoints.begin(), vend=allMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit;
        //already associated;
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId)
            continue;
        if(pMP->isBad())
            continue;
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5))
        {
            pMP->IncreaseVisible();
            nToMatch++;
        }
    }

    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 3;
        // If the camera has been relocalised recently, perform a coarser search
        if(mCurrentFrame.mnId<mnLastRelocFrameId + 2)
            th=5;
         matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }

}

void HRBFFusion::downloadTextures()
{
    Img<Eigen::Matrix<unsigned char, 4, 1>> image(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    textures[GPUTexture::RGB]->texture->Download(image.data, GL_RGBA, GL_UNSIGNED_BYTE);

    Img<Eigen::Vector4f> vertices(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    textures[GPUTexture::VERTEX_RAW]->texture->Download(vertices.data, GL_RGBA, GL_FLOAT);

    Img<Eigen::Vector4f> normal(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    textures[GPUTexture::NORMAL]->texture->Download(normal.data, GL_RGBA, GL_FLOAT);

    Img<Eigen::Vector4f> normal_opt(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    textures[GPUTexture::NORMAL_OPT]->texture->Download(normal_opt.data, GL_RGBA, GL_FLOAT);

    Img<float> confidence(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    textures[GPUTexture::CONFIDENCE]->texture->Download(confidence.data, GL_LUMINANCE, GL_FLOAT);

    Img<float> radius(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    textures[GPUTexture::RADIUS]->texture->Download(radius.data, GL_LUMINANCE, GL_FLOAT);

    Img<float> gradient_mag(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    textures[GPUTexture::GRADIENT_MAG]->texture->Download(gradient_mag.data, GL_LUMINANCE, GL_FLOAT);

    // Img<float> radius_opt(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    // textures[GPUTexture::RADIUS_OPTIMIZED]->texture->Download(radius_opt.data, GL_LUMINANCE, GL_FLOAT);

    Img<Eigen::Vector4f> cuvatureCuvrk1(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    textures[GPUTexture::PRINCIPAL_CURV1]->texture->Download(cuvatureCuvrk1.data, GL_RGBA, GL_FLOAT);

    Img<Eigen::Vector4f> cuvatureCuvrk2(Resolution::getInstance().rows(),Resolution::getInstance().cols());
    textures[GPUTexture::PRINCIPAL_CURV2]->texture->Download(cuvatureCuvrk2.data, GL_RGBA, GL_FLOAT);

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
    cv::imwrite("live_colorMap.png", imageCV, compression_params);
    std::cout << "image saved successfully" << std::endl;

    //show and down load texture
    //cv::namedWindow("HRBFNormalTexture", cv::WINDOW_AUTOSIZE);
    cv::Mat normalImagCV;
    normalImagCV.create(Resolution::getInstance().rows(), Resolution::getInstance().cols(), CV_8UC3);
    //create normalImagCV;
    //CV_ASSERT(normalImagCV.channels() == 4);
    for (int i = 0; i < normalImagCV.rows; ++i){
        for (int j = 0; j < normalImagCV.cols; ++j){
            cv::Vec3b& bgra = normalImagCV.at<cv::Vec3b>(i, j);
            if(normal.at<Eigen::Vector4f>(i,j)(0) == 0 && 
               normal.at<Eigen::Vector4f>(i,j)(1) == 0 && 
               normal.at<Eigen::Vector4f>(i,j)(2) == 0)
            {
                bgra[2] = static_cast<unsigned char>(UCHAR_MAX);
                bgra[1] = static_cast<unsigned char>(UCHAR_MAX);
                bgra[0] = static_cast<unsigned char>(UCHAR_MAX);
            }else{
                bgra[2] = static_cast<unsigned char>(UCHAR_MAX * (normal.at<Eigen::Vector4f>(i,j)(0) > 0 ? normal.at<Eigen::Vector4f>(i,j)(0) : 0)); // Blue
                bgra[1] = static_cast<unsigned char>(UCHAR_MAX * (normal.at<Eigen::Vector4f>(i,j)(1) > 0 ? normal.at<Eigen::Vector4f>(i,j)(1) : 0)); // Green
                bgra[0] = static_cast<unsigned char>(UCHAR_MAX * (normal.at<Eigen::Vector4f>(i,j)(2) > 0 ? normal.at<Eigen::Vector4f>(i,j)(2) : 0)); // Red
            }

        }
    }
//    std::vector<int> compression_params;
//    compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
//    compression_params.push_back(0);
    cv::imwrite("live_NormalMap.png", normalImagCV, compression_params);
    std::cout << "live_NormalMap saved successfully" << std::endl;

    //export mean square error
    float sum_error = 0.0;
    int count_error = 0;

    std::ofstream textureD;
    textureD.open("rawMap_attributes.txt");
    for(int i = 0; i < Resolution::getInstance().rows(); i++){
        for(int j = 0; j < Resolution::getInstance().cols(); j++)
        {
            if(vertices.at<Eigen::Vector4f>(i, j)(2) == 0 || normal.at<Eigen::Vector4f>(i, j)(0) == 0 /*isnan(cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(3))*/ /*|| cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(3) > 500 || cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(3) < -500*/)
                continue;
            textureD << vertices.at<Eigen::Vector4f>(i, j)(0)<<" " << vertices.at<Eigen::Vector4f>(i, j)(1) << " " << vertices.at<Eigen::Vector4f>(i, j)(2)<<" " /*<< vertices.at<Eigen::Vector4f>(i, j)(3) <<" "*/
                     //<< cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(0)<<" " << cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(1) << " " << cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(2)<<" "
                     << normal.at<Eigen::Vector4f>(i, j)(0)<<" " << normal.at<Eigen::Vector4f>(i, j)(1) << " " << normal.at<Eigen::Vector4f>(i, j)(2) <<" " //<< normal.at<Eigen::Vector4f>(i, j)(3)<<" "
                     /*<< cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(0)<<" "<<cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(1)<<" "<<cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(2)<<" "*/
                     /*<< cuvatureCuvrk2.at<Eigen::Vector4f>(i, j)(0)<<" "<<cuvatureCuvrk2.at<Eigen::Vector4f>(i, j)(1)<<" "<<cuvatureCuvrk2.at<Eigen::Vector4f>(i, j)(2)<<" "*/
//                     << vertices.at<Eigen::Vector4f>(i, j)(3) << " "
//                     << cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(3) << " " << cuvatureCuvrk2.at<Eigen::Vector4f>(i, j)(3) << " "
                     << sqrt(gradient_mag.at<float>(i, j)) << " "<< confidence.at<float>(i, j) //<< " "
                     //<< radius.at<float>(i, j) << " " << normal_opt.at<Eigen::Vector4f>(i, j)(3) << " "
                     << "\n";

            if(gradient_mag.at<float>(i, j) > 0)
            {
                sum_error =+ gradient_mag.at<float>(i, j);
                count_error++;
            }
//              if(std::isnan(cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(3)) || cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(3) == 0)
//                    textureD << j << ", " << i <<", "<< 0 <<", "<< 0 << "\n";
//              else
//                    textureD << j << ", " << i <<", "<< cuvatureCuvrk1.at<Eigen::Vector4f>(i, j)(3) << ", "<< cuvatureCuvrk2.at<Eigen::Vector4f>(i, j)(3) << "\n";
        }
    }
    float RMSE = sum_error / count_error;
    std::cout << "the RMSE of the hrbf recontruction is: " << RMSE << std::endl;

    textureD.close();
    std::cout <<"......textureCurr saved....."<< std::endl;
}

void HRBFFusion::savePly(std::string filename)
{

    // Open file
    std::ofstream fs;
    fs.open (filename.c_str ());

    Eigen::Vector4f * mapData = globalModel->downloadMap();
    //unsigned int validCount = globalModel.lastCount();
    int validCount = 0;

    std::cout << "global model last count: " << globalModel->lastCount() << std::endl;

    for(unsigned int i = 0; i < globalModel->lastCount(); i++)
    {
        Eigen::Vector4f pos = mapData[(i * 5) + 0];
        //std::cout << pos[3] << " ";
        if(pos[3] > GlobalStateParam::get().globalOutputSavePointCloudConfThreshold)
        {
            validCount++;
        }
    }

    // Write header
    fs << "ply";
    fs << "\nformat " << "binary_little_endian" << " 1.0";

    // Vertices
    fs << "\nelement vertex "<< validCount;
    fs << "\nproperty float x"
          "\nproperty float y"
          "\nproperty float z";

    fs << "\nproperty uchar red"
          "\nproperty uchar green"
          "\nproperty uchar blue";

    fs << "\nproperty float nx"
          "\nproperty float ny"
          "\nproperty float nz";

    fs << "\nproperty float curvature_max"
          "\nproperty float curvature_min";

    fs << "\nproperty float radius";

    fs << "\nproperty float submapIndex";

    fs << "\nend_header\n";

    // Close the file
    fs.close ();

    // Open file in binary appendable
    std::ofstream fpout(filename.c_str (), std::ios::app | std::ios::binary);
    for(unsigned int i = 0; i < globalModel->lastCount(); i++)
    {
        Eigen::Vector4f pos = mapData[(i * 5) + 0];

        //out put all points
        if(pos[3] > GlobalStateParam::get().globalOutputSavePointCloudConfThreshold)
        {
            Eigen::Vector4f col = mapData[(i * 5) + 1];
            Eigen::Vector4f nor = mapData[(i * 5) + 2];

            Eigen::Vector4f c_max = mapData[(i * 5) + 3];
            Eigen::Vector4f c_min = mapData[(i * 5) + 4];

            nor[0] *= -1;
            nor[1] *= -1;
            nor[2] *= -1;

            float value;
            memcpy (&value, &pos[0], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &pos[1], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &pos[2], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            //decoding of the current color
            unsigned char r = int(col[0]) >> 16 & 0xFF;
            unsigned char g = int(col[0]) >> 8 & 0xFF;
            unsigned char b = int(col[0]) & 0xFF;

            fpout.write (reinterpret_cast<const char*> (&r), sizeof (unsigned char));
            fpout.write (reinterpret_cast<const char*> (&g), sizeof (unsigned char));
            fpout.write (reinterpret_cast<const char*> (&b), sizeof (unsigned char));

            memcpy (&value, &nor[0], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[1], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[2], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &c_max[3], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &c_min[3], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &nor[3], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));

            memcpy (&value, &col[1], sizeof (float));
            fpout.write (reinterpret_cast<const char*> (&value), sizeof (float));
        }
    }
    // Close file
    fs.close ();
    delete [] mapData;
}

void HRBFFusion::saveFragments(int index_fragments){
    char fragment_template[128];
    strcpy(fragment_template, "fragments/fragment_");
    char suffix[64];
    snprintf(suffix, sizeof(suffix), "%03d.ply", index_fragments);
    strcat(fragment_template, suffix);
    savePly(fragment_template);

}
void HRBFFusion::saveFeedbackBufferPly(std::string type, std::string pose_type){

    Eigen::Matrix4f pose;
    std::cout <<"start saving;"<<std::endl;
    computeFeedbackBuffers();
    std::string filename;

    Eigen::Vector4f * frameData;
    unsigned int vertex_N;

    std::string feedbackSaveDir = "FeedBackPointCloud";
    struct stat st;
    if(stat(feedbackSaveDir.c_str(), &st) != 0)
    {
        std::cerr << feedbackSaveDir << " does not exist! make a clean folder\n";
        boost::filesystem::path feedbackFolder = feedbackSaveDir;
        boost::filesystem::create_directory(feedbackFolder);
    }

    if(type == "raw")
    {
        filename = feedbackSaveDir + "/raw_feedback_";
        filename += std::to_string(tick - 1);
        frameData = feedbackBuffers[FeedbackBuffer::RAW]->genFeedbackPointCloud();
        vertex_N = feedbackBuffers[FeedbackBuffer::RAW]->getcount();
    }
    if(type == "filtered")
    {
        filename = feedbackSaveDir + "/filtered_feedback_";
        filename += std::to_string(tick - 1);
        frameData = feedbackBuffers[FeedbackBuffer::FILTERED]->genFeedbackPointCloud();
        vertex_N = feedbackBuffers[FeedbackBuffer::FILTERED]->getcount();
    }
    filename.append(".ply");
    //Open file
    std::ofstream fs;
    fs.open(filename.c_str());
    std::cout<<"We got "<<vertex_N<<" points here"<<std::endl;
    //Write header
    fs<<"ply\n";
    fs<<"format ascii 1.0\n";
    fs<<"comment Created by Robin\n";

    //Vertices
    fs<<"element vertex "<<vertex_N<<"\n";
    fs<<"property float x\n"
        "property float y\n"
        "property float z\n";

    fs<<"property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n";

    fs<<"property float nx\n"
        "property float ny\n"
        "property float nz\n";

    fs<<"property float curvature1\n"
        "property float curvature2\n";
    //fs<<"property float scalar\n"; //output the index of each frame for each point
    fs<<"end_header\n";

    if(pose_type == "hrbf")
    {
        pose = currPose;
        std::cout << "hrbf pose: " <<std::endl;
        std::cout << pose << std::endl;
    }else if(pose_type == "groundtruth")
    {
        pose = trajectory_manager->poses[tick - 1];
        std::cout << "groundtruth pose: " <<std::endl;
        std::cout << pose << std::endl;
    }else {
        pose = Eigen::Matrix4f::Identity();
    }

    for(int i = 0; i < vertex_N;i++)
    {
        unsigned char r = int(frameData[5 * i + 1](0)) >> 16 & 0xFF;
        unsigned char g = int(frameData[5 * i + 1](0)) >> 8 & 0xFF;
        unsigned char b = int(frameData[5 * i + 1](0)) & 0xFF;

        if(r == 0 && g == 0 && b == 0)
        {
            r = 255;
            g = 255;
            b = 255;
        }

        Eigen::Vector4f vPoint;
        //this is the homogeneous coordinates for transformation
        vPoint << frameData[5 * i](0) , frameData[5 * i](1) , frameData[5 * i](2), 1.0;

        //original normal point to the camera viewing direction, this transfer points and normals to the global coordinate
        Eigen::Vector3f vNorm;
        vNorm << -frameData[5 * i + 2](0), -frameData[5 * i + 2](1), -frameData[5 * i + 2](2);

        float curv_max, curv_min;
        curv_max = frameData[5 * i + 3](3);
        curv_min = frameData[5 * i + 4](3);

        Eigen::Matrix3f rot = pose.topLeftCorner(3, 3);
        Eigen::Vector4f vPoint_g = pose * vPoint;
        Eigen::Vector3f vNorm_g = rot * vNorm;


        fs << vPoint_g(0) <<" "<< vPoint_g(1) <<" "<< vPoint_g(2)<<" "
           << static_cast<int> (r) <<" "<<static_cast<int> (g) <<" "<<static_cast<int> (b) << " "
           << vNorm_g(0) <<" "<< vNorm_g(1) <<" "<< vNorm_g(2) << " " << curv_max << " " << curv_min /*<<" "*/
           /*<< tick - 1*/ << "\n";
    }
    // Close the file
    fs.close ();
    delete [] frameData;
}

void HRBFFusion::mean_dist_diff_pose()
{
    computeFeedbackBuffers();
    Eigen::Vector4f* frameData;
    frameData = feedbackBuffers[FeedbackBuffer::FILTERED]->genFeedbackPointCloud();     //no use "currPose"
    unsigned int vertex_N = feedbackBuffers[FeedbackBuffer::FILTERED]->getcount();
    //
    float sum_dist = 0;
    for(int i = 0; i < vertex_N; i++)
    {
        Eigen::Vector4f vPoint, vPoint_hrbf, vPoint_groundtruth;
        vPoint << frameData[3 * i](0) , frameData[3 * i](1) , frameData[3 * i](2), 1.0;
        vPoint_hrbf = currPose * vPoint;
        vPoint_groundtruth = trajectory_manager->poses[tick - 1] * vPoint;
        sum_dist += sqrt((vPoint_hrbf(0) - vPoint_groundtruth(0)) * (vPoint_hrbf(0) - vPoint_groundtruth(0)) +
                         (vPoint_hrbf(1) - vPoint_groundtruth(1)) * (vPoint_hrbf(1) - vPoint_groundtruth(1)) +
                         (vPoint_hrbf(2) - vPoint_groundtruth(2)) * (vPoint_hrbf(2) - vPoint_groundtruth(2)));
    }
    float mean_dist_error = sum_dist / vertex_N;
    MeanErrorSet.push_back(mean_dist_error);
    std::cout <<"the mean_dist error of " << tick - 1 <<"th frame is: "<< mean_dist_error << std::endl;
    delete [] frameData;
}

Eigen::Vector3f HRBFFusion::rodrigues2(const Eigen::Matrix3f& matrix)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

    double rx = R(2, 1) - R(1, 2);
    double ry = R(0, 2) - R(2, 0);
    double rz = R(1, 0) - R(0, 1);

    double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
    double c = (R.trace() - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;

    double theta = acos(c);

    //not easy to understand if rotation matrix is a symmetric matrix?!!!
    if( s < 1e-5 )
    {
        double t;

        if( c > 0 )
            rx = ry = rz = 0;
        else
        {
            t = (R(0, 0) + 1)*0.5;
            rx = sqrt( std::max(t, 0.0) );
            t = (R(1, 1) + 1)*0.5;
            ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1)*0.5;
            rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
                rz = -rz;
            theta /= sqrt(rx*rx + ry*ry + rz*rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
        }
    }
    else
    {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
    }
    return Eigen::Vector3d(rx, ry, rz).cast<float>();
}

void HRBFFusion::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = spMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void HRBFFusion::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = true;
}

bool HRBFFusion::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopRequested;
}

void HRBFFusion::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopRequested = false;
}

void HRBFFusion::setPoseGraphOptimization()
{
    unique_lock<mutex> lock(mMutexPoseGraph);
    mbPoseGragh = true;
}

bool HRBFFusion::checkPoseGraphOptimization()
{
    unique_lock<mutex> lock(mMutexPoseGraph);
    return mbPoseGragh;
}

void HRBFFusion::setGlobalBA()
{
    unique_lock<mutex> lock(mMutexGlobalBA);
    mbGlobalBA = true;
}

bool HRBFFusion::checkGlobalBA()
{
    unique_lock<mutex> lock(mMutexGlobalBA);
    return mbGlobalBA;
}

//Sad times ahead
Map* HRBFFusion::getSparseMap()
{
    return spMap;
}

MapDrawer* HRBFFusion::getSparseMapDrawer()
{
    return mpMapDrawer;
}

FrameDrawer* HRBFFusion::getFrameDrawer()
{
    return mpFrameDrawer;
}

IndexMap & HRBFFusion::getIndexMap()
{
    return indexMap;
}

GlobalModel & HRBFFusion::getGlobalModel()
{
    return *globalModel;
}

std::map<std::string, GPUTexture*> & HRBFFusion::getTextures()
{
    return textures;
}

const std::vector<PoseMatch> & HRBFFusion::getPoseMatches()
{
    return poseMatches;
}

RGBDOdometry & HRBFFusion::getFrameToModel()
{
    return frameToModel;
}

const float & HRBFFusion::getConfidenceThreshold()
{
    return confidenceThreshold;
}

void HRBFFusion::setRgbOnly(const bool & val)
{
    rgbOnly = val;
}

void HRBFFusion::setIcpWeight(const float & val)
{
    icpWeight = val;
}

void HRBFFusion::setPyramid(const bool & val)
{
    pyramid = val;
}

void HRBFFusion::setFastOdom(const bool & val)
{
    fastOdom = val;
}

void HRBFFusion::setSo3(const bool & val)
{
    so3 = val;
}

void HRBFFusion::setFrameToFrameRGB(const bool & val)
{
    frameToFrameRGB = val;
}

void HRBFFusion::setConfidenceThreshold(const float & val)
{
    confidenceThreshold = val;
}

void HRBFFusion::setDepthCutoff(const float & val)
{
    depthCutoff = val;
}

const bool & HRBFFusion::getLost() //lel
{
    return lost;
}

const int & HRBFFusion::getTick()
{
    return tick;
}

void HRBFFusion::setTick(const int & val)
{
    tick = val;
}

const float & HRBFFusion::getMaxDepthProcessed()
{
    return maxDepthProcessed;
}

const Eigen::Matrix4f & HRBFFusion::getCurrPose()
{
    return currPose;
}

unsigned long int HRBFFusion::getSubmapNumber()
{
    return indexSubmap;
}

FillIn & HRBFFusion::getFillIn()
{
    return fillIn;
}

std::map<std::string, FeedbackBuffer*> & HRBFFusion::getFeedbackBuffers()
{
    return feedbackBuffers;
}

void HRBFFusion::InsertprocessedBA(int i)
{
    unique_lock<mutex> lock(newProcessed);
    processedBA.push_back(i);
}
bool HRBFFusion::CheckprocessedBA()
{
    unique_lock<mutex> lock(newProcessed);
    return(!processedBA.empty());
}

void HRBFFusion::notifyBADone()
{
    condVar.notify_all();
}

void HRBFFusion::notifyGlobalBADone()
{
    condVarGlobalBA.notify_all();
}
