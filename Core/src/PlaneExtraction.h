#ifndef PLANEEXTRACTION_H
#define PLANEEXTRACTION_H

#include <map>
#include <vector>
#include <fstream>
#include <math.h>
#include <algorithm>

#include "GPUTexture.h"
#include "Utils/Img.h"
#include "Utils/Resolution.h"
#include "Utils/Intrinsics.h"

//#include <pcl/point_types.h>
//#include <pcl/io/ply_io.h>
//#include <pcl/common/transforms.h>
//#include "./peac/AHCPlaneFitter.hpp"

//#include "opencv2/opencv.hpp"
//#include <opencv2/core/eigen.hpp>

//for ransac plane extraction
#include "RansacShapeDetector.h"
#include "PlanePrimitiveShapeConstructor.h"
#include "CylinderPrimitiveShapeConstructor.h"
#include "SpherePrimitiveShapeConstructor.h"
#include "ConePrimitiveShapeConstructor.h"
#include "TorusPrimitiveShapeConstructor.h"

#include "PointCloud.h"

//template<class PointT>
//struct OrganizedImage3D{
//    const pcl::PointCloud<PointT>& cloud;
//    //note: ahc::PlaneFitter assumes mm as unit!!!
//    const double unitScaleFactor;

//    OrganizedImage3D(const pcl::PointCloud<PointT>& c) : cloud(c), unitScaleFactor(1) {}
//    OrganizedImage3D(const OrganizedImage3D& other) : cloud(other.cloud), unitScaleFactor(other.unitScaleFactor) {}

//    inline int width() const { return cloud.width; }
//    inline int height() const { return cloud.height; }
//    inline bool get(const int row, const int col, double& x, double& y, double& z) const {
//            const PointT& pt=cloud.at(col,row);
//            x=pt.x*unitScaleFactor; y=pt.y*unitScaleFactor; z=pt.z*unitScaleFactor; //TODO: will this slowdown the speed?
//            return pcl_isnan(z)==0; //return false if current depth is NaN
//    }
//};

//typedef OrganizedImage3D<pcl::PointXYZ> ImageXYZ;
//typedef ahc::PlaneFitter< ImageXYZ > PlaneFitter;
//typedef pcl::PointCloud<pcl::PointXYZRGB> CloudXYZRGB;

struct planePara{
    float pos[3];
    float normal[3];
    float mean_dist;
};

struct shape2pointAssoc{
    int start;
    int interval;
};


class PlaneExtraction
{

public:
    template<class T>
    T iniGet(std::string key, T default_value) {
            std::map<std::string, std::string>::const_iterator itr=initpara.find(key);
            if(itr!=initpara.end()) {
                    std::stringstream ss;
                    ss<<itr->second;
                    T ret;
                    ss>>ret;
                    return ret;
            }
            return default_value;
    }

public:

    PlaneExtraction();
    ~PlaneExtraction();

    std::vector<std::vector<int>> p_member_ship;

    std::vector<shape2pointAssoc> shape2pointAssocSet;

    //the scale is in mm
//    pcl::PointCloud<pcl::PointXYZ>* cloud;
    PointCloud pc;
    std::vector<planePara> planeParaSet;

    enum InputType{
        DEPTH,
        VERTICES
    };

    bool initParameterLoad(std::string iniFileName);

    void SetParametersPF(std::string iniFileName);

    //Input is the depth metrics data
    void processOneframe(GPUTexture * source, PlaneExtraction::InputType inputType);

    void processOneframe(float* vertices, int rows, int cols, int i);

    void processOneframeRansac(float* vertices, float* normals, int rows, int cols, std::string type, const Eigen::Matrix4f& currpose);

//    PlaneFitter* getPointFitter()
//    {
//        return &pf;
//    }

private:

    //PointCloud* pc;
    //plane extration parameter
    std::map<std::string, std::string> initpara;

//    PlaneFitter pf;
};

#endif // PLANEEXTRACTION_H
