
#include "SubMap.h"


SubMap::SubMap()
{

}

SubMap::~SubMap()
{

}

void SubMap::SetKeyFrame(KeyFrame* spkeyframe_){
    spkeyframe = spkeyframe_;
}

void SubMap::SetPose(Eigen::Matrix4f inPose)
{
    Pose = inPose;
}
