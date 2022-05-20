#ifndef LINESLAM_UTILS_H
#define LINESLAM_UTILS_H
#ifdef OS_WIN
	#include <Windows.h>
#else
	#include <time.h>
#endif
#include "lineslam.h"

namespace Line3D {

ntuple_list callLsd (IplImage* src);
cv::Point2d mat2cvpt (const cv::Mat& m);
cv::Point3d mat2cvpt3d (cv::Mat m);
cv::Mat cvpt2mat( cv::Point2d p, bool homo=true);
cv::Mat cvpt2mat(const cv::Point3d& p, bool homo=true);
cv::Mat array2mat(double a[], int n);

double getMonoSubpix(const cv::Mat& img, cv::Point2d pt);

void showImage(string name, cv::Mat *img, int width=640) ;
string num2str(double i);

template<class bidiiter> //Fisher-Yates shuffle
bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random) {
	size_t left = std::distance(begin, end);
	while (num_random--) {
		bidiiter r = begin;
		std::advance(r, rand()%left);
		std::swap(*begin, *r);
		++begin;
		--left;
	}    
	return begin;
}

void computeLine3d_svd (vector<cv::Point3d> pts, cv::Point3d& mean, cv::Point3d& drct);
void computeLine3d_svd (vector<RandomPoint3d> pts, cv::Point3d& mean, cv::Point3d& drct);
void computeLine3d_svd (const vector<RandomPoint3d>& pts, const vector<int>& idx, cv::Point3d& mean, cv::Point3d& drct);
RandomLine3d extract3dline(const vector<cv::Point3d>& pts);
RandomLine3d extract3dline_mahdist(const vector<RandomPoint3d>& pts);
cv::Point3d projectPt3d2Ln3d (const cv::Point3d& P, const cv::Point3d& mid, const cv::Point3d& drct);
void derive3dLineWithCov(vector<cv::Point3d> pts, cv::Point3d mid, cv::Point3d drct,
	cv::Point3d& P, cv::Point3d& Q);
void derive3dLineWithCov(vector<RandomPoint3d> pts, cv::Point3d mid, cv::Point3d drct,
	RandomPoint3d& P, RandomPoint3d& Q);
bool verify3dLine(vector<cv::Point3d> pts, cv::Point3d A, cv::Point3d B);
bool verify3dLine(const vector<RandomPoint3d>& pts, const cv::Point3d& A,  const cv::Point3d& B);
double dist3d_pt_line (cv::Point3d X, cv::Point3d A, cv::Point3d B);
double dist3d_pt_line (cv::Mat x, cv::Point3d A, cv::Point3d B);

double depthStdDev (double d) ;
RandomPoint3d compPt3dCov (cv::Point3d pt, cv::Mat K);
double mah_dist3d_pt_line (cv::Point3d p, cv::Mat C, cv::Point3d q1, cv::Point3d q2);
double mah_dist3d_pt_line (cv::Point3d p, cv::Mat R, cv::Mat s, cv::Point3d q1, cv::Point3d q2);
double mah_dist3d_pt_line (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2);
double mah_dist3d_pt_line (const RandomPoint3d& pt, const cv::Mat& q1, const cv::Mat& q2);
cv::Point3d mahvec_3d_pt_line (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2);
cv::Point3d mahvec_3d_pt_line(const RandomPoint3d& pt, cv::Mat q1, cv::Mat q2);
cv::Point3d closest_3dpt_online_mah (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2);
double closest_3dpt_ratio_online_mah (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2);
void termReason(int info);
void MLEstimateLine3d (RandomLine3d& line, int maxIter=200);
void MLEstimateLine3d_compact (RandomLine3d& line,	int maxIter);
void MLEstimateLine3d_perpdist (vector<cv::Point3d> pts, cv::Point3d midpt, cv::Point3d dirct, RandomLine3d& line);
void MlestimateLine3dCov (double* p, int n, int i1, int i2, const cv::Mat& cov_meas_inv,
						  cv::Mat& cov_i1, cv::Mat& cov_i2);
void write_linepairs_tofile(vector<RandomLine3d> a, vector<RandomLine3d> b, string fname, double timestamp);
class MyTimer
{
#ifdef OS_WIN
public:
	MyTimer() {	QueryPerformanceFrequency(&TickPerSec);	}

	LARGE_INTEGER TickPerSec;        // ticks per second
	LARGE_INTEGER Tstart, Tend;           // ticks

	double time_ms;
	double time_s;
	void start()  {	QueryPerformanceCounter(&Tstart);}
	void end() 	{
		QueryPerformanceCounter(&Tend);
		time_ms = (Tend.QuadPart-Tstart.QuadPart)*1000.0/TickPerSec.QuadPart;
		time_s = time_ms/1000.0;
	}
#else
public:
	timespec t0, t1; 
	MyTimer() {}
	double time_ms;
	double time_s;
	void start() {
		clock_gettime(CLOCK_REALTIME, &t0);
	}
	void end() {
		clock_gettime(CLOCK_REALTIME, &t1);
		time_ms = t1.tv_sec * 1000 + t1.tv_nsec/1000000.0 - (t0.tv_sec * 1000 + t0.tv_nsec/1000000.0);
		time_s = time_ms/1000.0;			
	}
#endif	
};

int computeMSLD (FrameLine& l, cv::Mat* xGradient, cv::Mat* yGradient) ;
void trackLine (vector<FrameLine> f1, vector<FrameLine> f2, vector<vector<int> >& matches, vector<int>& addNew);
void trackLine3D (vector<FrameLine> f1, vector<FrameLine> f2, vector<vector<int> >& matches, vector<int>& addNew);
double lineSegmentOverlap(const FrameLine& a, const FrameLine& b);
void computeRelativeMotion_svd (vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t);
vector<int> computeRelativeMotion_Ransac (vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t);
void optimizeRelmotion(vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t);
cv::Mat q2r(cv::Mat q);
cv::Mat q2r (double* q);
cv::Mat r2q(cv::Mat R);
cv::Mat vec2SkewMat (cv::Point3d vec);
//void write2file (Map3d& m, string suffix);
double pesudoHuber(double e, double band);
double rotAngle (cv::Mat R);
void matchLine (vector<FrameLine> f1, vector<FrameLine> f2, vector<vector<int> >& matches);
double ave_img_bright(cv::Mat img);
vector<int> computeRelativeMotion_Ransac (vector<cv::Point3d> a, vector<cv::Point3d> b, cv::Mat& R, cv::Mat& t, double thresh);
bool get_pt_3d (cv::Point2d p2, cv::Point3d& p3, const cv::Mat& depth);
bool compute_motion_given_ptpair_file (string filename, const cv::Mat& depth, cv::Mat& R, cv::Mat& t);

}
#endif
