#ifndef LINESLAM_H
#define LINESLAM_H
#include <iostream>
#include<pangolin/pangolin.h>
#include "define.h"
// #include <cv.h>
// #include <cxcore.h>
// #include <highgui.h>
#include <opencv/cv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>
#ifdef QTPROJECT
#include <QThread>
#endif
#include "../external/lsd/lsd.h"
#include "../external/levmar-2.6/levmar.h"
#include"../../Shaders/Shaders.h"
#include"../../Utils/GlobalStateParams.h"

using namespace std;

// #define QTPROJECT  defined in preprocessor  // qt project or plain project
#define EPS	(1e-10)
#define PI (3.14159265)
//#define	SLAM_LBA 
//#define 	OS_WIN

namespace Line3D{

class RandomPoint3d 
{
public:
	cv::Point3d pos;
	double		xyz[3];
	cv::Mat		cov;
	cv::Mat		U, W; // cov = U*D*U.t, D = diag(W); W is vector

	RandomPoint3d(){}
	RandomPoint3d(cv::Point3d _pos) 
	{
		pos = _pos;
		xyz[0] = _pos.x;
		xyz[1] = _pos.y;
		xyz[2] = _pos.z;
		cov = cv::Mat::eye(3,3,CV_64F);
		U = cv::Mat::eye(3,3,CV_64F);
		W = cv::Mat::ones(3,1,CV_64F);
	}
	RandomPoint3d(cv::Point3d _pos, cv::Mat _cov)
	{
		pos = _pos;
		xyz[0] = _pos.x;
		xyz[1] = _pos.y;
		xyz[2] = _pos.z;
		cov = _cov.clone();
		cv::SVD svd(cov);
		U = svd.u.clone();
		W = svd.w.clone();
	}
};

class RandomLine3d 
{
public:
	vector<RandomPoint3d> pts;  //supporting collinear points
	cv::Point3d A, B;
	cv::Mat covA, covB;
	RandomPoint3d rndA, rndB;
	cv::Point3d u, d; // following the representation of Zhang's paper 'determining motion from...'
	RandomLine3d () {}
	RandomLine3d (cv::Point3d _A, cv::Point3d _B, cv::Mat _covA, cv::Mat _covB) 
	{
		A = _A;
		B = _B;
		covA = _covA.clone();
		covB = _covB.clone();
	}
	
};

class LmkLine
{
public:
	cv::Point3d			A, B;
	int					gid;
	vector<vector<int> >	frmId_lnLid;

	LmkLine(){}
};

class FrameLine 
// FrameLine represents a line segment detected from a rgb-d frame.
// It contains 2d image position (endpoints, line equation), and 3d info (if 
// observable from depth image).
{
public:
    cv::Point2d p, q;                   // image endpoints p and q
    cv::Mat l;                          // 3-vector of image line equation,
    double lineEq2d[3];

    bool haveDepth;			// whether have valid depth
    RandomLine3d line3d;

    bool haveMatched;

    cv::Point2d	 r;                     // image line gradient direction (polarity);
    cv::Mat des;                        // image line descriptor;

    int lid;				// local id in frame
    int gid;				// global id;

    int lid_prvKfrm;                    // correspondence's lid in previous keyframe

    int frameId;

    FrameLine() {gid = -1;}
    FrameLine(cv::Point2d p_, cv::Point2d q_, int fId);

    cv::Point2d getGradient(cv::Mat* xGradient, cv::Mat* yGradient);
    void complineEq2d()
    {
        cv::Mat pt1 = (cv::Mat_<double>(3,1)<<p.x, p.y, 1);
        cv::Mat pt2 = (cv::Mat_<double>(3,1)<<q.x, q.y, 1);
        cv::Mat lnEq = pt1.cross(pt2); // lnEq = pt1 x pt2
        lnEq = lnEq/sqrt(lnEq.at<double>(0)*lnEq.at<double>(0)
                +lnEq.at<double>(1)*lnEq.at<double>(1)); // normalize, optional
        lineEq2d[0] = lnEq.at<double>(0);
        lineEq2d[1] = lnEq.at<double>(1);
        lineEq2d[2] = lnEq.at<double>(2);

    }
};

class Frame
// Frame represents a rgb-d frame, including its feature info
{
public:
	int					id;
	double				timestamp;
	bool				isKeyFrame;
        vector<FrameLine>               lines;
	cv::Mat				R;
	cv::Mat				t;
	cv::Mat				rgb, gray;
	cv::Mat				depth,oriDepth;	// 
	double				lineLenThresh;

    float scaleFactor;

    //for visualization
    std::shared_ptr<Shader> drawLinesProgram;
    GLuint vbo;

    Frame () {}
    Frame (string rgbName, string depName, cv::Mat K, cv::Mat dc, float sf, int frameID);

    // detect line segments from rgb image: input: gray image, ouput: lines
    void detectFrameLines();
    void mergeColinearLines();
    void extractLineDepth();
    void update3DLinePose(cv::Mat R, cv::Mat t);
    void drawLines(pangolin::OpenGlMatrix mvp, const Eigen::Matrix4f & pose);
    void clear();
    void write3DLines2file(string);
};

class PoseConstraint
{
public:
	int from, to; // keyframe ids
	cv::Mat R, t;
	int numMatches;
};

}

class SystemParameters 
{
public:
	double	ratio_of_collinear_pts;		// decide if a frameline has enough collinear pts
	double	pt2line_dist_extractline;	// threshold pt to line distance when detect lines from pts
	double	pt2line_mahdist_extractline;// threshold for pt to line mahalanobis distance
	int		ransac_iters_extract_line;	// max ransac iters when detect lines from pts
        double	line_segment_len_thresh;		// min length of image line segment to use
	double	ratio_support_pts_on_line;	// the ratio of the number of filled cells over total cell number along a line
										// to check if a line has enough points covering the whole range
	int		num_cells_lineseg_range;	// divide a linesegment into multiple cells 
	double	line3d_length_thresh;		// frameline length threshold in 3d
	double	stdev_sample_pt_imgline;	// std dev of sample point from an image line
	double  depth_stdev_coeff_c1;		// c1,c2,c3: coefficients of depth noise quadratic function
	double  depth_stdev_coeff_c2;
	double  depth_stdev_coeff_c3;
	int		num_2dlinematch_keyframe;	// detect keyframe, minmum number of 2d line matches left
	int		num_3dlinematch_keyframe;
	double	pt2line3d_dist_relmotion;	// in meter, 
	double  line3d_angle_relmotion;		// in degree
	int		num_raw_frame_skip;			// number of raw frame to skip when tracking lines
	int		window_length_keyframe;		
	bool	fast_motion;
	double	inlier_ratio_constvel;
	int		num_pos_lba;
	int		num_frm_lba;
	// ----- lsd setting -----
	double lsd_angle_th;
	double lsd_density_th;
	// ----- loop closing -----
	double loopclose_interval;  // frames, check loop closure
	int	   loopclose_min_3dmatch;  // min_num for 3d line matches between two frames

	bool	g2o_BA_use_kernel;
	double  g2o_BA_kernel_delta;

	bool 	dark_ligthing;
	double	max_img_brightness;

	void init();
	SystemParameters(){}

};


#endif //LINESLAM_H
