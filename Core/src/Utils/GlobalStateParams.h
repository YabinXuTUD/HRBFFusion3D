//for global parameter control

#pragma once

#include <cassert>
#include <vector>
#include <string>
#include <list>

#include "parameterFile.h"

#define X_GLOBAL_PARAM_FIELDS \
    X(std::string, currentWorkingDirectory) \
    X(int, sensorType)\
    X(std::string, klgFileName) \
    X(std::string, AssociationFile)\
    X(std::string, parameterFileCvFormat)\
    X(bool, optimizationUseLocalBA)\
    X(bool, optimizationUseGlobalBA)\
    X(std::string, optimizationVocabularyFile)\
    X(bool, preprocessingUsebilateralFilter)\
    X(float, preprocessingInitRadiusMultiplier)\
    X(float, preprocessingCurvEstimationWindow)\
    X(float, preprocessingCurvValidThreshold)\
    X(int, preprocessingUseConfEval)\
    X(float, preprocessingConfEvalEpsilon)\
    X(bool, registrationPreAlignSO3)\
    X(float, registrationJointICPWeight)\
    X(bool, registrationICPUseSparseICP)\
    X(bool, registrationUsePlaneConstraint)\
    X(bool, registrationICPUseCoorespondenceSearch)\
    X(int, registrationICPNeighborSearchRadius)\
    X(bool, registrationICPUseWeightedICP)\
    X(float, registrationICPCurvWeightImpactControl)\
    X(float, registrationICPErrorThreshold)\
    X(float, registrationICPCovarianceThreshold)\
    X(bool, registrationColorUseRGBGrad)\
    X(float, registrationColorPhotoThreshold)\
    X(float, preictionWindowMultiplier)\
    X(int, preictionMinNeighbors)\
    X(int, preictionMaxNeighbors)\
    X(float, preictionConfThreshold)\
    X(float, fusionMergeWindowMultiplier)\
    X(float, fusionCleanWindowMultiplier)\
    X(float, globalConfidenceThreshold)\
    X(float, globalDenseEnoughThresh)\
    X(float, globalDepthCutoff)\
    X(bool, globalInputICLNUIMDataset)\
    X(bool, globalInputLoadTrajectory)\
    X(std::string, globalInputTrajectoryFormat) \
    X(std::string, globalInputTrajectoryFile) \
    X(bool, globalOutputSaveTrjectoryFile)\
    X(std::string, globalOutputSaveTrjectoryFileType)\ 
    X(bool, globalOutputCalculateMeanDistWithGroundTruth)\
    X(float, globalOutputSavePointCloudConfThreshold)\
    X(bool, globalOutputsaveTimings)\
    X(int, globalStartFrame)\
    X(int, globalEndFrame)\
    X(int, globalFrameToSkip)\
    X(bool, globalExportFramePeriod)\
    X(int, globalExportFrameStart)\
    X(int, globalExportFrameEnd)\
    X(float, preprocessingNormalEstimationPCA)

class GlobalStateParam
{
public:
    static const GlobalStateParam& getParameters();

#define X(type, name) type name;
    X_GLOBAL_PARAM_FIELDS
#undef X

    // set the parameter file and reads
    void readMembers(const ParameterFile& parameterFile){
        m_ParameterFile = parameterFile;
        readMembers();
    };

    //reads all the members from the given parameter file (could be called for reloading
    void readMembers(){
#define X(type, name) \
    if (!m_ParameterFile.readParameter(std::string(#name), name)) {std::cout << "skip param name" << name << std::endl;}
        X_GLOBAL_PARAM_FIELDS
#undef X
    IsGlobParamInitialized = true;
    }

    void print() const {
#define X(type, name) \
      std::cout << #name " = " << name << std::endl;
        X_GLOBAL_PARAM_FIELDS
#undef X
    }

    static GlobalStateParam& getInstance(){
        static GlobalStateParam s;
        return s;
    }

    static GlobalStateParam& get() {
        return getInstance();
    }

    //! constructor
    GlobalStateParam() {
        IsGlobParamInitialized = false;
        //m_pQuery = NULL;
    }

    //! destructor
    ~GlobalStateParam() {
    }


private:

    bool IsGlobParamInitialized;
    ParameterFile m_ParameterFile;

};
