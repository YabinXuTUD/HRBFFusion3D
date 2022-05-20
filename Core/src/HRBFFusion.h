
#pragma once

#ifndef HRBFFusion_H_
#define HRBFFusion_H_

#include "Utils/RGBDOdometry.h"
#include "Utils/Resolution.h"
#include "Utils/Intrinsics.h"
#include "Utils/Stopwatch.h"
#include "Utils/GlobalStateParams.h"
#include "Utils/TrajectoryManager.h"
#include "Shaders/Shaders.h"
#include "Shaders/ComputePack.h"
#include "Shaders/FeedbackBuffer.h"
#include "Shaders/FillIn.h"
#include "GlobalModel.h"
#include "IndexMap.h"
#include "Ferns.h"
#include "PoseMatch.h"
#include "Defines.h"
#include "../../GUI/src/Tools/ThreadMutexObject.h"
#include "../../GUI/src/HRBF_fusion.h"

#include "PlaneExtraction.h"
#include <stdio.h>
#include <iomanip>
#include <cmath>
#include <math.h>
#include <pangolin/gl/glcuda.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <condition_variable>
#include <sys/stat.h>
#include <sys/types.h>
#include <boost/filesystem.hpp>

#include<Eigen/StdVector>

#include "opencv2/opencv.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>

//#include <Core.h>

//ORB_SLAM2 Tracking system
#include "System.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBextractor.h"
#include "ORBVocabulary.h"
#include "Viewer.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

extern std::condition_variable condVar;
//typedef ORB_SLAM2::System ORB_SLAM_SYSTEM;
//using namespace ORB_SLAM2;
//class ORB_SLAM_SYSTEM;
//forward decralation with a namespace
#include "../../Core/src/Line/lineslam.h"
#include "../../Core/src/Line/global_line_constructor.h"

namespace ORB_SLAM2
{
    class System;
    class Map;
    class Frame;
    class LocalMapping;
    class LoopClosing;
    class ORBextractor;
    class Optimizer;
    class FrameDrawer;
}

class HRBFFusion
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EFUSION_API HRBFFusion(const int countThresh = 35000,
                              const float errThresh = 5e-05,
                              const float confidence = 10,
                              const float depthCut = 3,
                              const float icpThresh = 10,
                              const bool fastOdom = false,
                              const bool so3 = true,
                              const bool frameToFrameRGB = false);

        virtual ~HRBFFusion();

        void LoadImagesAssociationFile(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenamesRGB,
                        std::vector<std::string> &vstrImageFilenamesD, std::vector<double> &vTimestamps);

        /**
         * Process an rgb/depth map pair
         * @param rgb unsigned char row major order
         * @param depth unsigned short z-depth in millimeters, invalid depths are 0
         * @param timestamp nanoseconds (actually only used for the output poses, not important otherwise)
         * @param inPose optional input SE3 pose (if provided, we don't attempt to perform tracking)
         * @param weightMultiplier optional full frame fusion weight
         * @param bootstrap if true, use inPose as a pose guess rather than replacement
         */
        EFUSION_API void processFrame(const unsigned char * rgb,
                          const unsigned short * depth,
                          const int64_t & timestamp,
                          const float weightMultiplier = 1.f);

        /**
         * Load new input frame
         */
        EFUSION_API void inputFrame();
        /**
         * input the ith key frame, extract orb features, and converted to submap
         */
        EFUSION_API void PreProcessFrame(int i);
        /**
         *  Local Mapping might have changed some MapPoints tracked in last frame
         */
        EFUSION_API void CheckReplacedInLastFrame();
        /**
         * Check out new submap should be inserted (every 10 frames)
         */
        EFUSION_API bool NeedNewSubMap();
        /**
         * Comstruct new Submap,
         */
        EFUSION_API void ConstructSubmaps();
        /**
         * HRBF dense fusion is stopped when global loop is correcting
        */
        EFUSION_API void RequestStop();

        EFUSION_API bool stopRequested();

        EFUSION_API void setPoseGraphOptimization();

        EFUSION_API void setGlobalBA();

        EFUSION_API bool checkPoseGraphOptimization();

        EFUSION_API bool checkGlobalBA();
        /**
         * Update global model After BA
         */

        EFUSION_API void Release();


        EFUSION_API void UpdateDenseGlobalModel();
        /**
         * Sparse map initilization for ORB based Bundle Adjustment
         */
        EFUSION_API void SparseMapRGBDInitilization();
        /**
         * Update local key frames for the current inserted frame
         */
        EFUSION_API void UpdateLocalKeyFrames();
        /**
         * Update all local map points for the current key frame
         */
        EFUSION_API void UpdateLocalPoints();
        /**
         * Match all local map points for the current key frame, For connecttivity and Bundle adjustment
         */
        EFUSION_API void SearchLocalPoints();
        /**
         * Match all map points for the current key frame, For connecttivity and Bundle adjustment
         */
        EFUSION_API void SearchAllPoints();
        /*
         * Predicts the current view of the scene, updates the [vertex/normal/image]Tex() members
         * of the indexMap class
         */
        EFUSION_API void predict(float weighting);

        EFUSION_API Map* getSparseMap();

        EFUSION_API MapDrawer* getSparseMapDrawer();

        EFUSION_API FrameDrawer* getFrameDrawer();

        /**
         * This class contains all of the predicted renders
         * @return reference
         */
        EFUSION_API IndexMap & getIndexMap();

        /**
         * This class contains the surfel map
         * @return
         */
        EFUSION_API GlobalModel & getGlobalModel();

        /**
         * This is the map of raw input textures (you can display these)
         * @return
         */
        EFUSION_API std::map<std::string, GPUTexture*> & getTextures();

        /**
         * This is the list of deformation constraints
         * @return
         */
        EFUSION_API const std::vector<PoseMatch> & getPoseMatches();

        /**
         * This is the tracking class(frame to model), if you want access
         * @return
         */
        EFUSION_API RGBDOdometry & getFrameToModel();

        /**
         * The point fusion confidence threshold
         * @return
         */
        EFUSION_API const float & getConfidenceThreshold();

        /**
         * If you set this to true we just do 2.5D RGB-only Lucas–Kanade tracking (no fusion)
         * @param val
         */
        EFUSION_API void setRgbOnly(const bool & val);

        /**
         * Weight for ICP in tracking
         * @param val if 100, only use depth for tracking, if 0, only use RGB. Best value is 10
         */
        EFUSION_API void setIcpWeight(const float & val);

        /**
         * Whether or not to use a pyramid for tracking
         * @param val default is true
         */
        EFUSION_API void setPyramid(const bool & val);

        /**
         * Controls the number of tracking iterations
         * @param val default is false
         */
        EFUSION_API void setFastOdom(const bool & val);

        /**
         * Turns on or off SO(3) alignment bootstrapping
         * @param val
         */
        EFUSION_API void setSo3(const bool & val);

        /**
         * Turns on or off frame to frame tracking for RGB
         * @param val
         */
        EFUSION_API void setFrameToFrameRGB(const bool & val);

        /**
         * Raw data fusion confidence threshold
         * @param val default value is 10, but you can play around with this
         */
        EFUSION_API void setConfidenceThreshold(const float & val);

        /**
         * Cut raw depth input off at this point
         * @param val default is 3 meters
         */
        EFUSION_API void setDepthCutoff(const float & val);

        /**
         * Returns whether or not the camera is lost, if relocalisation mode is on
         * @return
         */
        EFUSION_API const bool & getLost();

        /**
         * Get the internal clock value of the fusion process
         * @return monotonically increasing integer value (not real-world time)
         */
        EFUSION_API const int & getTick();

        /**
         * Cheat the clock, only useful for multisession/log fast forwarding
         * @param val control time itself!
         */
        EFUSION_API void setTick(const int & val);

        /**
         * Internal maximum depth processed, this is defaulted to 20 (for rescaling depth buffers)
         * @return
         */
        EFUSION_API const float & getMaxDepthProcessed();

        /**
         * The current global camera pose estimate
         * @return SE3 pose
         */
        EFUSION_API const Eigen::Matrix4f & getCurrPose();

        EFUSION_API unsigned long int getSubmapNumber();

        EFUSION_API FillIn & getFillIn();

        /**
         * These are the vertex buffers computed from the raw input data
         * @return can be rendered
         */
        EFUSION_API std::map<std::string, FeedbackBuffer*> & getFeedbackBuffers();

        /**
         * Return the Global Kernels from the Map
         * @return can be rendered
         */

        /**
         * Calculate the above for the current frame (only done on the first frame normally)
         */
        EFUSION_API void computeFeedbackBuffers();


        EFUSION_API void downloadTextures();

        /**
         * Saves out a .ply mesh file of the current model
         */
        EFUSION_API void savePly(std::string filename);

        EFUSION_API void saveFragments(int index_fragments);

        /**
          *Saves out a .ply mesh file of the current frame,(with global coordinates)
          */

        EFUSION_API void saveFeedbackBufferPly(std::string type, std::string pose_type);

        EFUSION_API void mean_dist_diff_pose();

        EFUSION_API void save_inliers_number();

        /**
         * Renders a normalised view of the input raw depth for displaying as an OpenGL texture
         * (this is stored under textures[GPUTexture::DEPTH_NORM]
         * @param minVal minimum depth value to render
         * @param maxVal maximum depth value to render
         */
        EFUSION_API void normaliseDepth(const float & minVal, const float & maxVal);

        EFUSION_API void SaveKeyFrameTrajectoryTUM(const string &filename);

        std::mutex UpdateTrajectory;
        //store test data below;
        std::vector<float> MeanErrorSet;
        std::vector<float> inliers_num;

        //input
        cv::Mat imRGB, imD;
        double tframe;
        //bool updateModel_BA;
        //std::mutex UpdateGlobalModel;
        bool insertAsubmap;
        bool insertSubmap;

        std::mutex updateModel;

        std::mutex updateModelGlobalBA;

        bool ready_to_update;
        void notifyBADone();
        void notifyGlobalBADone();

        std::mutex newProcessed;
        void InsertprocessedBA(int pKF);
        bool CheckprocessedBA();

        //current frame;
        ORB_SLAM2::Frame mCurrentFrame;
        cv::Mat mImGray;

        //***************orb_slam2*************//
        GlobalModel* globalModel;   //stored in GPU;
        TrajectoryManager* trajectory_manager;  //can be accessed by the local mapping thread

        bool mbStopped;
        bool mbStopRequested;
        bool mbNotStop;
        std::mutex mMutexStop;

        bool mbPoseGragh;
        std::mutex mMutexPoseGraph;
        bool mbGlobalBA;
        std::mutex mMutexGlobalBA;

protected:
        std::list<int> processedBA;
        //****orb_slam2*****//
        //Other Thread Pointers
        LocalMapping* mpLocalMapper;
        LoopClosing* mpLoopCloser;

        // System threads: Local Mapping, Loop Closing, Viewer.
        // The Tracking thread "lives" in the main execution thread that creates the System object.
        std::thread* mptLocalMapping;
        std::thread* mptLoopClosing;
        std::thread* mptViewer;
        //ORB
        ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
        ORBextractor* mpIniORBextractor;

        //Calibration matrix
        cv::Mat mK;
        cv::Mat mDistCoef;
        float mbf;         //camera baseline

        //New KeyFrame rules (according to fps)
        int mMinFrames;
        int mMaxFrames;

        // Threshold close/far points
        // Points seen as close by the stereo/RGBD sensor are considered reliable
        // and inserted from just one frame. Far points requiere a match in two keyframes.
        float mThDepth;

        //the scale factor is inversed,i.e. 1 / 5000.0
        float mDepthMapFactor;

        //Last Frame, KeyFrame and Relocalisation Info
        KeyFrame* mpLastKeyFrame;
        ORB_SLAM2::Frame mLastFrame;
        unsigned int mnLastKeyFrameId;    //the previous newly added frame ID
        unsigned int mnLastRelocFrameId;  //last relocolization frame ID

        //Color order (true RGB, false BGR, ignored if grayscale)
        bool mbRGB;

        std::vector<string> vstrImageFilenamesRGB;
        std::vector<string> vstrImageFilenamesD;
        std::vector<double> vTimestamps;

        //BoW
        ORBVocabulary* mpORBVocabulary;
        KeyFrameDatabase* mpKeyFrameDB;

        //Local Map
        KeyFrame* mpReferenceKF;                  //for relocalization
        std::vector<KeyFrame*> mvpLocalKeyFrames;
        std::vector<MapPoint*> mvpLocalMapPoints;

        //for sparse feature map point and key frames
        Map* spMap;

        //set to the viewer and draw
        FrameDrawer* mpFrameDrawer;
        MapDrawer* mpMapDrawer;

        //submap index in the global model, not frame ID, index in Key frame vector
        int indexSubmap;

        //Here be dragons
private:
        IndexMap indexMap;          //projected index of the vertices, CPU or GPU?
        RGBDOdometry frameToModel;  //registration/RGBD odometry

        FillIn fillIn;    //fill in the image

        std::map<std::string, GPUTexture*> textures;  //for textures,RGB/Depth, the raw data.
        std::map<std::string, ComputePack*> computePacks;  //for all the computational usage in Textures.
        std::map<std::string, FeedbackBuffer*> feedbackBuffers;  //last two buffers reflect on screen...

        void LoadCameraParaAndInitORBExtractor();
        void createTextures();
        void createCompute();
        void createFeedbackBuffers();

        //bilateral filtering/guass filtering
        void filterDepth(float depthFactor);
        //metricise depth map from origin to meters with a scaleFactor
        void metriciseDepth(float depthFactor);
        //compute pricinpal curvatures, gradient, updated normal radius
        void computeCurvatureGradient();

        void updateNormalRad();
        void VertexConfidence(float weighting);
        //compute raw vertex map, filtered vertex map, normal, and radius
        void computeVertexNormalRadius();
        //outlier remove by model points
        void outliersRemoval();
        
        bool denseEnough(const Img<Eigen::Matrix<unsigned char, 3, 1>> & img);
        bool denseEnough(const Img<Eigen::Vector4f> & vertices);

        Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);

        Eigen::Matrix4f initPose;
        Eigen::Matrix4f currPose;
        Eigen::Matrix4f lastKeyFramePose;

        int tick;
        const int icpCountThresh;
        const float icpErrThresh;

        const int consSample;
        Resize resize; //

        std::vector<PoseMatch> poseMatches;

        std::vector<std::pair<unsigned long long int, Eigen::Matrix4f>> poseGraph;
        std::vector<unsigned long long int> poseLogTimes;

        Img<Eigen::Matrix<unsigned char, 3, 1>> imageBuff;
        Img<Eigen::Vector4f> verticesBuff;
        Img<unsigned short> timesBuff;

        bool lost;
        bool lastFrameRecovery;
        const float maxDepthProcessed;

        bool rgbOnly;
        float icpWeight;
        bool pyramid;
        bool fastOdom;
        float confidenceThreshold;

        bool so3;
        bool frameToFrameRGB;
        float depthCutoff;

        float weighting;

        std::vector<std::map<std::string, float>> processingTimes;

};

#endif /* HRBFFusion_H_ */
