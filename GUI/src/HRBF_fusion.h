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

#include <map>
#include <HRBFFusion.h>
#include <Utils/Parse.h>
#include <thread>

#include <sys/stat.h>
#include <sys/types.h>
#include <boost/filesystem.hpp>

#include "Tools/GUI.h"
#include "Tools/GroundTruthOdometry.h"
#include "Tools/RawLogReader.h"
#include "Tools/LiveLogReader.h"
#include "Tools/RawImageReader.h"

#include "Utils/parameterFile.h"
#include "Utils/GlobalStateParams.h"
//#include "../../Core/src/Line/lineslam.h"

#include <Eigen/Core>

#ifndef MAINCONTROLLER_H_
#define MAINCONTROLLER_H_

extern std::condition_variable condVar;
class HRBFFusion;

class MainController
{
    public:
        MainController(int argc, char * argv[]);
        virtual ~MainController();

        void launch();

        //void load_trajectory();

    private:
        void run();

        bool good;
        bool aStep;

        //HRBF running on the main thread
        HRBFFusion * hrbfFusion;

        GUI * gui;
        GroundTruthOdometry * groundTruthOdometry;
        LogReader * logReader;

        std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> poses;

        std::string logFile;
        std::string poseFile;

        std::vector<string> vstrImageFilenamesRGB;
        std::vector<string> vstrImageFilenamesD;
        std::vector<double> vTimestamps;

        float confidence,
              depth,
              icp,
              icpErrThresh,
              covThresh,
              photoThresh;


        int timeDelta,
            icpCountThresh,
            start,
            end;

        bool fillIn,
             openLoop,
             reloc,
             quiet,
             fastOdom,
             so3,
             draw_unstable,
             rewind,
             frameToFrameRGB;

        int framesToSkip;
        bool streaming;
        bool resetButton;

        Resize * resizeStream;
};

#endif /* MAINCONTROLLER_H_ */
