#pragma once
#include <vector>
#include "Frame.h"
#include "KeyFrame.h"

#include <Eigen/Core>


using namespace ORB_SLAM2;

class SubMap
{
public:
    SubMap();
    ~SubMap();

    void SetKeyFrame(KeyFrame* spkeyframe_);
    void SetPose(Eigen::Matrix4f inPose);

private:
    //index in the Submap vector, which will be shown in the global model vertex buffer(color.y)
    int index;

    //the ID of the first Keyframe
    long unsigned int submapID;

    //Key frame pose
    Eigen::Matrix4f Pose;

    KeyFrame* spkeyframe;

};
