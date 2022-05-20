
#include "utils.h"

#define USE_VEC4MAT
#define SAVE_MEMORY
extern cv::Mat K, distCoeffs;
extern SystemParameters sysPara;

namespace Line3D{

ntuple_list callLsd (IplImage* src)
	// use LSD to extract line segments from an image
	// input : an image color/grayscale
	// output: a list of line segments - endpoint x-y coords + ...  
{
	IplImage* src_gray = src;
    image_double image;         //image_double is a struct defined in 'lsd.h'
	ntuple_list lsd_out;
	unsigned int w = src->width;
	unsigned int h = src->height;
    image = new_image_double(w, h);
    CvScalar s;         // to get image values
	for (int x=0;x<w;++x){
		for(int y=0;y<h;++y){			
			s=cvGet2D(src_gray,y,x); 
			image->data[x+y*image->xsize] = s.val[0];
		}
	}
//	MyTimer tm; tm.start();
	lsd_out = lsd(image);
	free_image_double(image);
	//	cvReleaseImage(&src_gray);
	return lsd_out;
}

cv::Point2d mat2cvpt (const cv::Mat& m)
	// 3x1 mat => point
{
	if (m.cols * m.rows == 2)
		return cv::Point2d(m.at<double>(0), m.at<double>(1));
	if (m.cols * m.rows ==3)
		return cv::Point2d(m.at<double>(0)/m.at<double>(2),
		m.at<double>(1)/m.at<double>(2));
	else
		cerr<<"input matrix dimmension wrong!";	
}

cv::Point3d mat2cvpt3d (cv::Mat m)
	// 3x1 mat => point
{
	if (m.cols * m.rows ==3)
		return cv::Point3d(m.at<double>(0),
		m.at<double>(1),
		m.at<double>(2));
	else
		cerr<<"input matrix dimmension wrong!";	
}

cv::Mat cvpt2mat(cv::Point2d p, bool homo)
{
	if (homo)
		return (cv::Mat_<double>(3,1)<<p.x, p.y, 1);
	else
		return (cv::Mat_<double>(2,1)<<p.x, p.y);
}

cv::Mat cvpt2mat(const cv::Point3d& p, bool homo)
	// this function is slow!
	// return cv::Mat(3,1,CV_64,arrary) does not work!
{
	if (homo)
		return (cv::Mat_<double>(4,1)<<p.x, p.y, p.z, 1);
	else {
		return (cv::Mat_<double>(3,1)<<p.x, p.y, p.z);
		
	}
}
cv::Mat array2mat(double a[], int n) // inhomo mat
	// n is the size of a[]
{
	return cv::Mat(n,1,CV_64F,a);
}

cv::Vec3b getColorSubpix(const cv::Mat& img, cv::Point2d pt)
{
	cv::Mat patch;
	cv::getRectSubPix(img, cv::Size(1,1), pt, patch);
	return patch.at<cv::Vec3b>(0,0);
}

double getMonoSubpix(const cv::Mat& img, cv::Point2d pt)
	// get pixel value of non-integer position
	// input: img must be CV_32F type
{
	cv::Mat patch;
	cv::getRectSubPix(img, cv::Size(1,1), pt, patch);
	return patch.at<float>(0,0);
}

void showImage(string name, cv::Mat *img, int width) 
{
	double ratio = double(width)/img->cols;
	cv::namedWindow(name,0);
	cv::imshow(name,*img);
	cvResizeWindow(name.c_str(),width,int(img->rows*ratio));
}

string num2str(double i)
{
	stringstream ss;
	ss<<i;
	return ss.str();
}

RandomLine3d extract3dline(const vector<cv::Point3d>& pts)
	// extract a single 3d line from point clouds using ransac
	// input: 3d points
	// output: inlier points, line parameters: midpt and direction
{
	int maxIterNo = sysPara.ransac_iters_extract_line;	
    double distThresh = sysPara.pt2line_dist_extractline;  //meter
	// distance threshold should be adapted to line length and depth
	int minSolSetSize = 2;

	vector<int> indexes(pts.size());
	for (int i=0; i<indexes.size(); ++i) indexes[i]=i;
	vector<cv::Point3d> maxInlierSet;
	cv::Point3d bestA, bestB;
    //find inliers
	for(int iter=0; iter<maxIterNo;iter++) {
		vector<cv::Point3d> inlierSet;
        random_unique(indexes.begin(), indexes.end(), minSolSetSize);       // shuffle
		cv::Point3d A = pts[indexes[0]], B = pts[indexes[1]];
		// compute a line from A and B
		if (cv::norm(B-A) < EPS ) continue; 
		for (int i=0; i<pts.size(); ++i) {
			// compute distance to AB
			//	MyTimer t;
			//	t.start();
			double dist = dist3d_pt_line(pts[i],A,B);
			//	t.end();cout<<"dist takes "<<t.time_ms<<endl;
			if (dist<distThresh) {
				inlierSet.push_back(pts[i]);
			}
		}		
		if(inlierSet.size() > maxInlierSet.size())	{
			if (verify3dLine(inlierSet, A, B)) {
				maxInlierSet = inlierSet;	
				bestA = A; bestB = B;
			}
		}
	}
	RandomLine3d rl;
	for(int i=0; i<maxInlierSet.size(); ++i)
		rl.pts.push_back(RandomPoint3d(maxInlierSet[i]));
	
	if (maxInlierSet.size() >= 2) {
		cv::Point3d m = (bestA+bestB)*0.5, d = bestB-bestA;
		// optimize and reselect inliers
		// compute a 3d line using algebraic method	
		while(true) {
			vector<cv::Point3d> tmpInlierSet;
			cv::Point3d tmp_m, tmp_d;
			computeLine3d_svd(maxInlierSet, tmp_m, tmp_d);
			for(int i=0; i<pts.size(); ++i) {				
				if(dist3d_pt_line(pts[i],tmp_m, tmp_m+tmp_d) < distThresh) {
					tmpInlierSet.push_back(pts[i]);			
				}
			}
			if(tmpInlierSet.size()>maxInlierSet.size()) {
				maxInlierSet = tmpInlierSet;
				m = tmp_m;
				d = tmp_d;
			} else 
				break;
		}
		// find out two endpoints
		double minv=100, maxv=-100;
        int	idx_end1 = 0, idx_end2 = 0;
		for(int i=0; i<maxInlierSet.size(); ++i) {
			double dproduct = (maxInlierSet[i]-m).dot(d);
			if ( dproduct < minv) {
				minv = dproduct;
				idx_end1 = i;
			}
			if (dproduct > maxv) {
				maxv = dproduct;
				idx_end2 = i;
			}
		}		
		rl.A = maxInlierSet[idx_end1];
		rl.B = maxInlierSet[idx_end2];
	
		rl.pts.clear();
		for(int i=0; i<maxInlierSet.size(); ++i)
			rl.pts.push_back(RandomPoint3d(maxInlierSet[i]));
	}

	return rl;	

}

RandomLine3d extract3dline_mahdist(const vector<RandomPoint3d>& pts)
	// extract a single 3d line from point clouds using ransac and mahalanobis distance
	// input: 3d points and covariances
	// output: inlier points, line parameters: midpt and direction
{
	int maxIterNo = min(sysPara.ransac_iters_extract_line,
		int(pts.size()*(pts.size()-1)*0.5));	
	double distThresh = sysPara.pt2line_mahdist_extractline; // meter
	// distance threshold should be adapted to line length and depth
	int minSolSetSize = 2;

	vector<int> indexes(pts.size());
	for (int i=0; i<indexes.size(); ++i) indexes[i]=i;
	vector<int> maxInlierSet;
	RandomPoint3d bestA, bestB;
	for(int iter=0; iter<maxIterNo;iter++) {
		vector<int> inlierSet;
		random_unique(indexes.begin(), indexes.end(),minSolSetSize);// shuffle
		const RandomPoint3d& A = pts[indexes[0]];
		const RandomPoint3d& B = pts[indexes[1]];
		// compute a line from A and B
		if (cv::norm(B.pos-A.pos) < EPS ) continue; 
		for (int i=0; i<pts.size(); ++i) {
			// compute distance to AB
			double dist = mah_dist3d_pt_line(pts[i], A.pos, B.pos);
			if (dist<distThresh) {
				inlierSet.push_back(i);
			}
		}		
		if(inlierSet.size() > maxInlierSet.size())	{
			vector<RandomPoint3d> inlierPts(inlierSet.size());
			for(int ii=0; ii<inlierSet.size(); ++ii)
				inlierPts[ii]=pts[inlierSet[ii]];
			if (verify3dLine(inlierPts, A.pos, B.pos)) {
				maxInlierSet = inlierSet;	
				bestA = pts[indexes[0]]; bestB = pts[indexes[1]];
			}
		}
		if( maxInlierSet.size() > pts.size()*0.9)
			break;
	}
		
	RandomLine3d rl;
	if (maxInlierSet.size() >= 2) {
		cv::Point3d m = (bestA.pos+bestB.pos)*0.5, d = bestB.pos-bestA.pos;		
		// optimize and reselect inliers
		// compute a 3d line using algebraic method	
		while(true) {
			vector<int> tmpInlierSet;
			cv::Point3d tmp_m, tmp_d;
			computeLine3d_svd(pts,maxInlierSet, tmp_m, tmp_d);
			for(int i=0; i<pts.size(); ++i) {
				if(mah_dist3d_pt_line(pts[i], tmp_m, tmp_m+tmp_d) < distThresh) {
					tmpInlierSet.push_back(i);					
				}
			}
			if(tmpInlierSet.size() > maxInlierSet.size()) {
				maxInlierSet = tmpInlierSet;
				m = tmp_m;
				d = tmp_d;
			} else 
				break;
		}
	  // find out two endpoints
		double minv=100, maxv=-100;
		int	   idx_end1 = 0, idx_end2 = 0;
		for(int i=0; i<maxInlierSet.size(); ++i) {
			double dproduct = (pts[maxInlierSet[i]].pos-m).dot(d);
			if ( dproduct < minv) {
				minv = dproduct;
				idx_end1 = i;
			}
			if (dproduct > maxv) {
				maxv = dproduct;
				idx_end2 = i;
			}
		}	
		rl.A = pts[maxInlierSet[idx_end1]].pos;
		rl.B = pts[maxInlierSet[idx_end2]].pos;	
	}	
	rl.pts.resize(maxInlierSet.size());
    for(int i=0; i< maxInlierSet.size();++i) rl.pts[i]= pts[maxInlierSet[i]];
	return rl;		 
}

void computeLine3d_svd (vector<cv::Point3d> pts, cv::Point3d& mean, cv::Point3d& drct)
	// input: collinear 3d points with noise
	// output: line direction vector and point
	// method: linear equation, PCA
{
	int n = pts.size();
	mean = cv::Point3d(0,0,0);
	for(int i=0; i<n; ++i) {
		mean =  mean + pts[i];
	}
	mean = mean * (1.0/n);
	cv::Mat P(3,n,CV_64F);
	for(int i=0; i<n; ++i) {
		pts[i] =  pts[i] - mean;
		cvpt2mat(pts[i],0).copyTo(P.col(i));
	}
	cv::SVD svd(P.t());
	drct = mat2cvpt3d(svd.vt.row(0));
}

void computeLine3d_svd (vector<RandomPoint3d> pts, cv::Point3d& mean, cv::Point3d& drct)
	// input: collinear 3d points with noise
	// output: line direction vector and point
	// method: linear equation, PCA
{
	int n = pts.size();
	mean = cv::Point3d(0,0,0);
	for(int i=0; i<n; ++i) {
		mean =  mean + pts[i].pos;
	}
	mean = mean * (1.0/n);
	cv::Mat P(3,n,CV_64F);
	for(int i=0; i<n; ++i) {
	//	pts[i].pos =  pts[i].pos - mean;
	//	cvpt2mat(pts[i].pos,0).copyTo(P.col(i));
		double pos[3] = {pts[i].pos.x - mean.x,pts[i].pos.y - mean.y,pts[i].pos.z - mean.z};
		array2mat(pos,3).copyTo(P.col(i));
	}
	cv::SVD svd(P.t());
	drct = mat2cvpt3d(svd.vt.row(0));
}

void computeLine3d_svd (const vector<RandomPoint3d>& pts, const vector<int>& idx, cv::Point3d& mean, cv::Point3d& drct)
	// input: collinear 3d points with noise
	// output: line direction vector and point
	// method: linear equation, PCA
{
	int n = idx.size();
	mean = cv::Point3d(0,0,0);
	for(int i=0; i<n; ++i) {
		mean =  mean + pts[idx[i]].pos;
	}
	mean = mean * (1.0/n);
	cv::Mat P(3,n,CV_64F);
	for(int i=0; i<n; ++i) {
	//	pts[i].pos =  pts[i].pos - mean;
	//	cvpt2mat(pts[i].pos,0).copyTo(P.col(i));
		double pos[3] = {pts[idx[i]].pos.x-mean.x, pts[idx[i]].pos.y-mean.y, pts[idx[i]].pos.z-mean.z};
		array2mat(pos,3).copyTo(P.col(i));
	}
	
	cv::SVD svd(P.t(), cv::SVD::MODIFY_A);  // FULL_UV is 60 times slower
	
	drct = mat2cvpt3d(svd.vt.row(0));
}


void derive3dLineWithCov(vector<cv::Point3d> pts, cv::Point3d mid, cv::Point3d drct,
	cv::Point3d& P, cv::Point3d& Q)
	// input: collinear 3d points (and their error cov)
	// output: mle of line, and cov
{
	/*******************************************************
	current implementation is not complete 
	********************************************************/
	// assuming pts is sorted
	int n = pts.size();
	P = projectPt3d2Ln3d(pts[0], mid, drct);
	Q = projectPt3d2Ln3d(pts[n-1], mid, drct);
}

void derive3dLineWithCov(vector<RandomPoint3d> pts, cv::Point3d mid, cv::Point3d drct,
	RandomPoint3d& P, RandomPoint3d& Q)
	// input: collinear 3d points (and their error cov)
	// output: mle of line, and cov
{
	/*******************************************************
	current implementation is not complete 
	********************************************************/
	// assuming pts is sorted
	int n = pts.size();
	P = RandomPoint3d(projectPt3d2Ln3d(pts[0].pos, mid, drct));
	Q = RandomPoint3d(projectPt3d2Ln3d(pts[n-1].pos, mid, drct));
}

cv::Point3d projectPt3d2Ln3d (const cv::Point3d& P, const cv::Point3d& mid, const cv::Point3d& drct)
	// project a 3d point P to a 3d line (represented with midpt and direction)
{
	cv::Point3d A = mid;
	cv::Point3d B = mid + drct;
	cv::Point3d AB = B-A;
	cv::Point3d AP = P-A;
	return A + (AB.dot(AP)/(AB.dot(AB)))*AB;
}

bool verify3dLine(vector<cv::Point3d> pts, cv::Point3d A, cv::Point3d B)
	// input: line AB, collinear points
	// output: whether AB is a good representation for points
	// method: divide AB (or CD, which is endpoints of the projected points on AB) 
	// into n sub-segments, detect how many sub-segments containing
	// at least one point(projected onto AB), if too few, then it implies invalid line
{
	int nCells = sysPara.num_cells_lineseg_range; // number of cells
	int* cells = new int[nCells];
	double ratio = sysPara.ratio_support_pts_on_line;
	for(int i=0; i<nCells; ++i) cells[i] = 0;
	int nPts = pts.size();
	// find 2 extremities of points along the line direction
	double minv=100, maxv=-100;
	int	   idx1 = 0, idx2 = 0;
	for(int i=0; i<nPts; ++i) {
		if ((pts[i]-A).dot(B-A) < minv) {
			minv = (pts[i]-A).dot(B-A);
			idx1 = i;
		}
		if ((pts[i]-A).dot(B-A) > maxv) {
			maxv = (pts[i]-A).dot(B-A);
			idx2 = i;
		}
	}	
	cv::Point3d C = projectPt3d2Ln3d (pts[idx1], (A+B)*0.5, B-A);
	cv::Point3d D = projectPt3d2Ln3d (pts[idx2], (A+B)*0.5, B-A);
	double cd = cv::norm(D-C);
	if(cd < EPS) {
		delete[] cells;
		return false;
	}
	for(int i=0; i<nPts; ++i) {
		cv::Point3d X = pts[i];
		double lambda = abs((X-C).dot(D-C)/cd/cd); // 0 <= lambd <=1
		if (lambda>=1) {
			cells[nCells-1] += 1;
		} else {			
			cells[(unsigned int)floor(lambda*10)] += 1;
		}		
	}
	double sum = 0;
	for (int i=0; i<nCells; ++i) {
		//		cout<<cells[i]<<"\t";
		if (cells[i] > 0 )
			sum ++;
	}
	//	cout<<'\t'<<sum<<endl;
	delete[] cells;
	if(sum/nCells > ratio) {
		return true;
	} else {
		return false;
	}
}

bool verify3dLine(const vector<RandomPoint3d>& pts, const cv::Point3d& A,  const cv::Point3d& B)
	// input: line AB, collinear points
	// output: whether AB is a good representation for points
	// method: divide AB (or CD, which is endpoints of the projected points on AB) 
	// into n sub-segments, detect how many sub-segments containing
	// at least one point(projected onto AB), if too few, then it implies invalid line
{
	int nCells = sysPara.num_cells_lineseg_range; // number of cells
	int* cells = new int[nCells];
	double ratio = sysPara.ratio_support_pts_on_line;
	for(int i=0; i<nCells; ++i) cells[i] = 0;
	int nPts = pts.size();
	// find 2 extremities of points along the line direction
	double minv=100, maxv=-100;
	int	   idx1 = 0, idx2 = 0;
	for(int i=0; i<nPts; ++i) {
		if ((pts[i].pos-A).dot(B-A) < minv) {
			minv = (pts[i].pos-A).dot(B-A);
			idx1 = i;
		}
		if ((pts[i].pos-A).dot(B-A) > maxv) {
			maxv = (pts[i].pos-A).dot(B-A);
			idx2 = i;
		}
	}	
	cv::Point3d C = projectPt3d2Ln3d (pts[idx1].pos, (A+B)*0.5, B-A);
	cv::Point3d D = projectPt3d2Ln3d (pts[idx2].pos, (A+B)*0.5, B-A);
	double cd = cv::norm(D-C);
	if(cd < EPS) {
		delete[] cells;
		return false;
	}
	for(int i=0; i<nPts; ++i) {
		cv::Point3d X = pts[i].pos;
		double lambda = abs((X-C).dot(D-C)/cd/cd); // 0 <= lambd <=1
		if (lambda>=1) {
			cells[nCells-1] += 1;
		} else {			
			cells[(unsigned int)floor(lambda*10)] += 1;
		}		
	}
	double sum = 0;
	for (int i=0; i<nCells; ++i) {
		//		cout<<cells[i]<<"\t";
		if (cells[i] > 0 )
			sum ++;
	}

	delete[] cells;
	if(sum/nCells > ratio) {
		return true;
	} else {
		return false;
	}
}

double dist3d_pt_line (cv::Point3d X, cv::Point3d A, cv::Point3d B)
	// input: point X, line (A,B)
{
	if(cv::norm(A-B)<EPS) {
		cerr<<"error in function dist3d_pt_line: line length can not be 0!"<<endl;
		return -1;
	}
	double ax = cv::norm(X-A);
	cv::Point3d nvAB = (B-A) * (1/cv::norm(A-B));
	return sqrt(abs( ax*ax - ((X-A).dot(nvAB))*((X-A).dot(nvAB))));
}

double dist3d_pt_line (cv::Mat X, cv::Point3d A, cv::Point3d B)
// input: point X, line (A,B)
{
	return dist3d_pt_line(mat2cvpt3d(X),A,B);
}

double depthStdDev (double d) 
	// standard deviation of depth d 
	// in meter
{
	double c1, c2, c3;
	c1 = sysPara.depth_stdev_coeff_c1;
	c2 = sysPara.depth_stdev_coeff_c2;
	c3 = sysPara.depth_stdev_coeff_c3; 
	return c1*d*d + c2*d + c3;
}

RandomPoint3d compPt3dCov (cv::Point3d pt, cv::Mat K)
{
	double f = K.at<double>(0,0), // focal length
		cu = K.at<double>(0,2),
		cv = K.at<double>(1,2);
	double sigma_impt = sysPara.stdev_sample_pt_imgline; // std dev of image sample point
	cv::Mat J = (cv::Mat_<double>(3,3)<< pt.z/f, 0, pt.x/pt.z,
		0, pt.z/f, pt.y/pt.z,
		0,0,1);
	cv::Mat cov_g_d = (cv::Mat_<double>(3,3)<<sigma_impt*sigma_impt, 0, 0,
		0, sigma_impt*sigma_impt, 0,
		0, 0, depthStdDev(pt.z)*depthStdDev(pt.z));
	cv::Mat cov = J * cov_g_d * J.t();
	return RandomPoint3d(pt,cov);
}

double mah_dist3d_pt_line (cv::Point3d p, cv::Mat C, cv::Point3d q1, cv::Point3d q2)
	// compute the Mahalanobis distance between a random 3d point p and line (q1,q2)
{
	if (C.cols != 3)	{
		cerr<<"Error in mah_dist3d_pt_line: cov matrix must be 3x3"<<endl;
		return -1;
	}
	cv::SVD svd(C);
	cv::Mat invRootD = (cv::Mat_<double>(3,3)<< 1/sqrt(svd.w.at<double>(0)), 0, 0,
		0, 1/sqrt(svd.w.at<double>(1)), 0,
		0, 0, 1/sqrt(svd.w.at<double>(2)));
	cv::Mat q1_ = invRootD * svd.u.t() * (cvpt2mat(q1-p,0)),
		q2_ = invRootD * svd.u.t() * (cvpt2mat(q2-p,0));
	return cv::norm(q1_.cross(q2_))/cv::norm(q1_-q2_);
}

double mah_dist3d_pt_line (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2)
// compute the Mahalanobis distance between a random 3d point p and line (q1,q2)
// this is fater version since the point cov has already been decomposed by svd
{
	if (pt.U.cols != 3)	{
		cerr<<"Error in mah_dist3d_pt_line: R matrix must be 3x3"<<endl;
		return -1;
	}

	double r11, r12, r13, r21, r22, r23, r31, r32, r33;
	r11 = pt.U.at<double>(0,0);
	r12 = pt.U.at<double>(0,1); 
	r13 = pt.U.at<double>(0,2);
	r21 = pt.U.at<double>(1,0);
	r22 = pt.U.at<double>(1,1);
	r23 = pt.U.at<double>(1,2);
	r31 = pt.U.at<double>(2,0);
	r32 = pt.U.at<double>(2,1);
	r33 = pt.U.at<double>(2,2);
	cv::Point3d q1_p = q1 - pt.pos, q2_p = q2 - pt.pos;
//MyTimer tm; tm.start();
	double s0 = sqrt(pt.W.at<double>(0)), s1 = sqrt(pt.W.at<double>(1)), s2 = sqrt(pt.W.at<double>(2));
	cv::Point3d q1n((q1_p.x * r11 + q1_p.y * r21 + q1_p.z * r31)/s0,
		(q1_p.x * r12 + q1_p.y * r22 + q1_p.z * r32)/s1,
		(q1_p.x * r13 + q1_p.y * r23 + q1_p.z * r33)/s2),
		q2n((q2_p.x * r11 + q2_p.y * r21 + q2_p.z * r31)/s0,
		(q2_p.x * r12 + q2_p.y * r22 + q2_p.z * r32)/s1,
		(q2_p.x * r13 + q2_p.y * r23 + q2_p.z * r33)/s2);
	double out = cv::norm(q1n.cross(q2n))/cv::norm(q1n-q2n);
//tm.end(); //cout<<"mah dist "<<tm.time_ms<<endl;
	return out;
}
cv::Point3d mahvec_3d_pt_line (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2)
// compute the Mahalanobis distance vector between a random 3d point p and line (q1,q2)
// this is fater version since the point cov has already been decomposed by svd
{
	if (pt.U.cols != 3)	{
		cerr<<"Error in mah_dist3d_pt_line: R matrix must be 3x3"<<endl;
		exit(0);
	}

	double r11, r12, r13, r21, r22, r23, r31, r32, r33;
	r11 = pt.U.at<double>(0,0);
	r12 = pt.U.at<double>(0,1); 
	r13 = pt.U.at<double>(0,2);
	r21 = pt.U.at<double>(1,0);
	r22 = pt.U.at<double>(1,1);
	r23 = pt.U.at<double>(1,2);
	r31 = pt.U.at<double>(2,0);
	r32 = pt.U.at<double>(2,1);
	r33 = pt.U.at<double>(2,2);
	cv::Point3d q1_p = q1 - pt.pos, q2_p = q2 - pt.pos;

	double s0 = sqrt(pt.W.at<double>(0)), s1 = sqrt(pt.W.at<double>(1)), s2 = sqrt(pt.W.at<double>(2));
	cv::Point3d q1n((q1_p.x * r11 + q1_p.y * r21 + q1_p.z * r31)/s0,
		(q1_p.x * r12 + q1_p.y * r22 + q1_p.z * r32)/s1,
		(q1_p.x * r13 + q1_p.y * r23 + q1_p.z * r33)/s2),
		q2n((q2_p.x * r11 + q2_p.y * r21 + q2_p.z * r31)/s0,
		(q2_p.x * r12 + q2_p.y * r22 + q2_p.z * r32)/s1,
		(q2_p.x * r13 + q2_p.y * r23 + q2_p.z * r33)/s2);
//	double out = cv::norm(q1n.cross(q2n))/cv::norm(q1n-q2n);
	double t = - q1n.dot(q2n-q1n)/((q2n-q1n).dot(q2n-q1n));
	return q1n + t * (q2n - q1n);
}

cv::Point3d closest_3dpt_online_mah (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2)
// compute the closest point using the Mahalanobis distance from a random 3d point p to line (q1,q2)
// this is fater version since the point cov has already been decomposed by svd
{
	if (pt.U.cols != 3)	{
		cerr<<"Error in mah_dist3d_pt_line: R matrix must be 3x3"<<endl;
		exit(0);
	}

	double r11, r12, r13, r21, r22, r23, r31, r32, r33;
	r11 = pt.U.at<double>(0,0);
	r12 = pt.U.at<double>(0,1); 
	r13 = pt.U.at<double>(0,2);
	r21 = pt.U.at<double>(1,0);
	r22 = pt.U.at<double>(1,1);
	r23 = pt.U.at<double>(1,2);
	r31 = pt.U.at<double>(2,0);
	r32 = pt.U.at<double>(2,1);
	r33 = pt.U.at<double>(2,2);
	cv::Point3d q1_p = q1 - pt.pos, q2_p = q2 - pt.pos;

	double s0 = sqrt(pt.W.at<double>(0)), s1 = sqrt(pt.W.at<double>(1)), s2 = sqrt(pt.W.at<double>(2));
	cv::Point3d q1n((q1_p.x * r11 + q1_p.y * r21 + q1_p.z * r31)/s0,
		(q1_p.x * r12 + q1_p.y * r22 + q1_p.z * r32)/s1,
		(q1_p.x * r13 + q1_p.y * r23 + q1_p.z * r33)/s2),
		q2n((q2_p.x * r11 + q2_p.y * r21 + q2_p.z * r31)/s0,
		(q2_p.x * r12 + q2_p.y * r22 + q2_p.z * r32)/s1,
		(q2_p.x * r13 + q2_p.y * r23 + q2_p.z * r33)/s2);
//	double out = cv::norm(q1n.cross(q2n))/cv::norm(q1n-q2n);
	double t = - q1n.dot(q2n-q1n)/((q2n-q1n).dot(q2n-q1n));
	cv::Point3d cpt =  q1n + t * (q2n - q1n);
	cv::Point3d cpt_w(cpt.x*s0*r11+cpt.y*s1*r12+cpt.z*s2*r13+pt.pos.x,
				cpt.x*s0*r21+cpt.y*s1*r22+cpt.z*s2*r23+pt.pos.y,
				cpt.x*s0*r31+cpt.y*s1*r32+cpt.z*s2*r33+pt.pos.z);
	return cpt_w;
}

double closest_3dpt_ratio_online_mah (const RandomPoint3d& pt, cv::Point3d q1, cv::Point3d q2)
// compute the closest point using the Mahalanobis distance from a random 3d point p to line (q1,q2)
// return the ratio t, such that q1 + t * (q2 - q1) is the closest point 
{
	if (pt.U.cols != 3)	{
		cerr<<"Error in mah_dist3d_pt_line: R matrix must be 3x3"<<endl;
		exit(0);
	}

	double r11, r12, r13, r21, r22, r23, r31, r32, r33;
	r11 = pt.U.at<double>(0,0);
	r12 = pt.U.at<double>(0,1); 
	r13 = pt.U.at<double>(0,2);
	r21 = pt.U.at<double>(1,0);
	r22 = pt.U.at<double>(1,1);
	r23 = pt.U.at<double>(1,2);
	r31 = pt.U.at<double>(2,0);
	r32 = pt.U.at<double>(2,1);
	r33 = pt.U.at<double>(2,2);
	cv::Point3d q1_p = q1 - pt.pos, q2_p = q2 - pt.pos;

	double s0 = sqrt(pt.W.at<double>(0)), s1 = sqrt(pt.W.at<double>(1)), s2 = sqrt(pt.W.at<double>(2));
	cv::Point3d q1n((q1_p.x * r11 + q1_p.y * r21 + q1_p.z * r31)/s0,
		(q1_p.x * r12 + q1_p.y * r22 + q1_p.z * r32)/s1,
		(q1_p.x * r13 + q1_p.y * r23 + q1_p.z * r33)/s2),
		q2n((q2_p.x * r11 + q2_p.y * r21 + q2_p.z * r31)/s0,
		(q2_p.x * r12 + q2_p.y * r22 + q2_p.z * r32)/s1,
		(q2_p.x * r13 + q2_p.y * r23 + q2_p.z * r33)/s2);
	double t = - q1n.dot(q2n-q1n)/((q2n-q1n).dot(q2n-q1n));
	return t;
}

double mah_dist3d_pt_line (const RandomPoint3d& pt, const cv::Mat& q1, const cv::Mat& q2)
// compute the Mahalanobis distance between a random 3d point p and line (q1,q2)
// this is fater version since the point cov has already been decomposed by svd
{
//	cout<<pt.cov<<endl;
	return mah_dist3d_pt_line(pt, mat2cvpt3d(q1), mat2cvpt3d(q2));
}

cv::Point3d mahvec_3d_pt_line(const RandomPoint3d& pt, cv::Mat q1, cv::Mat q2)
{
	return mahvec_3d_pt_line(pt, mat2cvpt3d(q1), mat2cvpt3d(q2));
}

struct Data_MLEstimateLine3d
{
	int idx1, idx2;
//	vector<double> meas;
#ifdef USE_VEC4MAT
	vector<vector<vector<double> > >& wmat;  
	Data_MLEstimateLine3d(vector<vector<vector<double> > >& wmt) :wmat(wmt) {}
#else 
	vector<cv::Mat>& whitenMat;
	Data_MLEstimateLine3d(vector<cv::Mat>& wm) : whitenMat(wm) {}
#endif

};

void costFun_MLEstimateLine3d(double *p, double *error, int m, int n, void *adata)
{
#ifdef USE_VEC4MAT
	struct Data_MLEstimateLine3d* dptr;
	dptr = (struct Data_MLEstimateLine3d *) adata;	
	int curParaIdx = 0, curErrIdx = 0;
	int idx1 = dptr->idx1, idx2 = dptr->idx2;

	double a0 = p[idx1], a1 = p[idx1+1], a2 = p[idx1+2];
	double b0 = p[idx2+2], b1 = p[idx2+1+2], b2 = p[idx2+2+2];


	for(int i=0; i < dptr->wmat.size(); ++i) {
		if (i == dptr->idx1) {			
			error[curErrIdx] = dptr->wmat[i][0][0]*a0 + dptr->wmat[i][0][1]*a1 + dptr->wmat[i][0][2]*a2;
			error[curErrIdx+1] = dptr->wmat[i][1][0]*a0 + dptr->wmat[i][1][1]*a1 + dptr->wmat[i][1][2]*a2;
			error[curErrIdx+2] = dptr->wmat[i][2][0]*a0 + dptr->wmat[i][2][1]*a1 + dptr->wmat[i][2][2]*a2;
			curParaIdx = curParaIdx + 3;
		} else	if (i == dptr->idx2){
			error[curErrIdx] = dptr->wmat[i][0][0]*b0 + dptr->wmat[i][0][1]*b1 + dptr->wmat[i][0][2]*b2;
			error[curErrIdx+1] = dptr->wmat[i][1][0]*b0 + dptr->wmat[i][1][1]*b1 + dptr->wmat[i][1][2]*b2;
			error[curErrIdx+2] = dptr->wmat[i][2][0]*b0 + dptr->wmat[i][2][1]*b1 + dptr->wmat[i][2][2]*b2;
			curParaIdx = curParaIdx + 3;
		} else {
			error[curErrIdx] = dptr->wmat[i][0][0]*(a0*p[curParaIdx]+b0*(1-p[curParaIdx])) 
				+ dptr->wmat[i][0][1]*(a1*p[curParaIdx]+b1*(1-p[curParaIdx])) 
				+ dptr->wmat[i][0][2]*(a2*p[curParaIdx]+b2*(1-p[curParaIdx]));
			error[curErrIdx+1]=dptr->wmat[i][1][0]*(a0*p[curParaIdx]+b0*(1-p[curParaIdx])) 
				+ dptr->wmat[i][1][1]*(a1*p[curParaIdx]+b1*(1-p[curParaIdx])) 
				+ dptr->wmat[i][1][2]*(a2*p[curParaIdx]+b2*(1-p[curParaIdx]));
			error[curErrIdx+2]=dptr->wmat[i][2][0]*(a0*p[curParaIdx]+b0*(1-p[curParaIdx])) 
				+ dptr->wmat[i][2][1]*(a1*p[curParaIdx]+b1*(1-p[curParaIdx])) 
				+ dptr->wmat[i][2][2]*(a2*p[curParaIdx]+b2*(1-p[curParaIdx]));
			curParaIdx = curParaIdx + 1;
		}

		curErrIdx = curErrIdx + 3;
	}
	

#else
	//	MyTimer t; t.start();
	struct Data_MLEstimateLine3d* dptr;
	dptr = (struct Data_MLEstimateLine3d *) adata;	
	int curParaIdx = 0, curErrIdx = 0;
	int idx1 = dptr->idx1, idx2 = dptr->idx2;
	cv::Mat A = (cv::Mat_<double>(3,1)<< p[idx1], p[idx1+1], p[idx1+2]);
	cv::Mat B = (cv::Mat_<double>(3,1)<< p[idx2+2], p[idx2+1+2], p[idx2+2+2]);


	for(int i=0; i < dptr->whitenMat.size(); ++i) {
		if (i == dptr->idx1) {
			cv::Mat wpt = dptr->whitenMat[i] * A;
			error[curErrIdx] = wpt.at<double>(0);
			error[curErrIdx+1] = wpt.at<double>(1);
			error[curErrIdx+2] = wpt.at<double>(2);

			curParaIdx = curParaIdx + 3;
		} else	if (i == dptr->idx2){
			cv::Mat wpt = dptr->whitenMat[i] * B;
			error[curErrIdx] = wpt.at<double>(0);
			error[curErrIdx+1] = wpt.at<double>(1);
			error[curErrIdx+2] = wpt.at<double>(2);

			curParaIdx = curParaIdx + 3;
		} else {
			cv::Mat wpt = dptr->whitenMat[i] * (p[curParaIdx] * A + (1-p[curParaIdx]) * B);
			error[curErrIdx] = wpt.at<double>(0);
			error[curErrIdx+1] = wpt.at<double>(1);
			error[curErrIdx+2] = wpt.at<double>(2);

			curParaIdx = curParaIdx + 1;
		}

		curErrIdx = curErrIdx + 3;
	}
	//	t.end(); cout<<"comput cost "<<t.time_ms<<" ms"<<endl;
#endif
//		double cost=0;
//		for(int i=0; i<dptr->meas.size();++i)
//			cost += (dptr->meas[i] - error[i])*(dptr->meas[i] - error[i]);
//		cout<<cost<<'\t';
}
void MLEstimateLine3d (RandomLine3d& line,	int maxIter)
	// optimally estimate a 3d line from a set of collinear random 3d points
	// 3d line is represented by two points
{
	static double acum=0;
	static int count = 0;
			MyTimer timer; 	timer.start();

	// ----- preprocessing: find 2 extremities of points along the line direction -----
	double minv=100, maxv=-100;
	int	   idx_end1 = 0, idx_end2 = 0;
	for(int i=0; i<line.pts.size(); ++i) {
		double dproduct = (line.pts[i].pos-line.A).dot(line.A-line.B);
		if ( dproduct < minv) {
			minv = dproduct;
			idx_end1 = i;
		}
		if (dproduct > maxv) {
			maxv = dproduct;
			idx_end2 = i;
		}
	}	
	if(idx_end1 > idx_end2) swap(idx_end1, idx_end2); // ensure idx_end1 < idx_end2
	// ----- LM parameter setting -----
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU; //
	opts[1] = 1E-10; // gradient threshold, original 1e-15
	opts[2] = 1E-20; // relative para change threshold? original 1e-50
	opts[3] = 1E-20; // error threshold (below it, stop)
	opts[4] = LM_DIFF_DELTA;

	// ----- optimization parameters -----
	vector<double> paraVec, measVec;

#ifdef USE_VEC4MAT
	vector<vector<vector<double> > > tmpwmat;
	Data_MLEstimateLine3d data(tmpwmat);
#else
	vector<cv::Mat> tmpwhitenMat;
	Data_MLEstimateLine3d data(tmpwhitenMat);
#endif

	data.idx1 = idx_end1;
	data.idx2 = idx_end2;
	for(int i = 0; i < line.pts.size(); ++i) {			
		cv::Mat DInvRoot = (cv::Mat_<double>(3,3)<< 1/sqrt(line.pts[i].W.at<double>(0)), 0, 0,
			0, 1/sqrt(line.pts[i].W.at<double>(1)), 0,
			0, 0, 1/sqrt(line.pts[i].W.at<double>(2)));
		cv::Mat CovInvRoot = DInvRoot * line.pts[i].U.t();
		cv::Mat whitenedMeasPt = CovInvRoot * cvpt2mat(line.pts[i].pos,0);
		measVec.push_back(whitenedMeasPt.at<double>(0));
		measVec.push_back(whitenedMeasPt.at<double>(1));
		measVec.push_back(whitenedMeasPt.at<double>(2));
#ifndef USE_VEC4MAT
		tmpwhitenMat.push_back(CovInvRoot);
#else
		vector<vector<double> > vm(3); vm[0].resize(3);vm[1].resize(3);vm[2].resize(3);
		for(int j=0; j<3;++j)
			for(int k=0; k<3; ++k)
				vm[j][k] = CovInvRoot.at<double>(j,k);
		tmpwmat.push_back(vm);
#endif
		if (i == idx_end1 || i == idx_end2) {
			paraVec.push_back(line.pts[i].pos.x);
			paraVec.push_back(line.pts[i].pos.y);
			paraVec.push_back(line.pts[i].pos.z);			
		} else {
			//	paraVec.push_back(0.5); // initialization need improvement
			paraVec.push_back(cv::norm(line.pts[i].pos-line.pts[idx_end2].pos)
				/(cv::norm(line.pts[i].pos-line.pts[idx_end2].pos)+cv::norm(line.pts[i].pos-line.pts[idx_end1].pos)));
		}
	}
	int numPara = paraVec.size();
	double* para = new double[numPara];
	for (int i=0; i<numPara; ++i) {
		para[i] = paraVec[i];
	}
	int numMeas = measVec.size();
	double* meas = new double[numMeas];
	for ( int i=0; i<numMeas; ++i) {
		meas[i] = measVec[i];
	}
//	data.meas = measVec;
	// ----- start LM solver -----
	int ret = dlevmar_dif(costFun_MLEstimateLine3d, para, meas, numPara, numMeas,
		100, opts, info, NULL, NULL, (void*)&data);
	//	termReason((int)info[6]);
	

	// ----- compute cov of MLE result -----
	cv::Mat cov_meas_inv = cv::Mat::zeros(3*line.pts.size(), 3*line.pts.size(), CV_64F);
	cv::Mat tmp;
	for(int i=0; i<line.pts.size();++i) {
		tmp = line.pts[i].cov.inv();
		tmp.copyTo(cov_meas_inv.rowRange(i*3,i*3+3).colRange(i*3,i*3+3));
	}

	// --- compute line endpoint uncerainty ---
	MlestimateLine3dCov (para, numPara, idx_end1, idx_end2, cov_meas_inv, line.covA, line.covB);

	// refine line endpoint positions
	line.A = cv::Point3d (para[idx_end1],para[idx_end1+1],para[idx_end1+2]);
	line.B = cv::Point3d (para[idx_end2+2],para[idx_end2+3],para[idx_end2+4]);
	line.rndA = RandomPoint3d(line.A, line.covA);
	line.rndB = RandomPoint3d(line.B, line.covB);
#ifdef SAVE_MEMORY
	line.pts.clear();
#endif
	timer.end();	
	count +=1;
	acum += timer.time_ms;
	if(count%50==0)	cout<<"mle accumulated: "<<acum<<endl;

	delete[] meas;
	delete[] para;
}

struct Data_MLEstimateLine3d_Compact
{
	int idx1, idx2;
	vector<RandomPoint3d>& pts;
	Data_MLEstimateLine3d_Compact(vector<RandomPoint3d>& _pts) :pts(_pts) {}

};

void costFun_MLEstimateLine3d_compact(double *p, double *error, int m, int n, void *adata)
{
	struct Data_MLEstimateLine3d_Compact* dptr;
	dptr = (struct Data_MLEstimateLine3d_Compact *) adata;	
	int curParaIdx = 0, curErrIdx = 0;
	int idx1 = dptr->idx1, idx2 = dptr->idx2;
	cv::Mat a(3,1,CV_64F, p), b(3,1,CV_64F, &p[3]);
	cv::Point3d ap(p[0], p[1], p[2]);
	cv::Point3d bp(p[3], p[4], p[5]);
	for(int i=0; i<dptr->pts.size(); ++i) {
		if(i==idx1) {
			cv::Mat mdist = (a-cvpt2mat(dptr->pts[i].pos,0)).t() *dptr->pts[i].cov.inv()*(a-cvpt2mat(dptr->pts[i].pos,0));
			error[i] = mdist.at<double>(0);
		} else if(i==idx2) {
			cv::Mat mdist = (b-cvpt2mat(dptr->pts[i].pos,0)).t() *dptr->pts[i].cov.inv()*(b-cvpt2mat(dptr->pts[i].pos,0));
			error[i] = mdist.at<double>(0);
		} else {
			error[i] = mah_dist3d_pt_line (dptr->pts[i], ap, bp);
		}
	}
	double cost=0;
	static int c=0;
	c++;
	assert(dptr->pts.size()==n);
	for(int i=0; i<dptr->pts.size(); ++i)
		cost+=error[i]*error[i];
//	cout<<cost<<"\t";
}
void MLEstimateLine3d_compact (RandomLine3d& line,	int maxIter)
	// optimally estimate a 3d line from a set of collinear random 3d points
	// 3d line is represented by two points
{
//	static double acum=0;
//	static int count = 0;
//	MyTimer timer; 	timer.start();

	// ----- preprocessing: find 2 extremities of points along the line direction -----
	double minv=100, maxv=-100;
	int	   idx_end1 = 0, idx_end2 = 0;
	for(int i=0; i<line.pts.size(); ++i) {
		double dproduct = (line.pts[i].pos-line.A).dot(line.A-line.B);
		if ( dproduct < minv) {
			minv = dproduct;
			idx_end1 = i;
		}
		if (dproduct > maxv) {
			maxv = dproduct;
			idx_end2 = i;
		}
	}	
	if(idx_end1 > idx_end2) swap(idx_end1, idx_end2); // ensure idx_end1 < idx_end2
	// ----- LM parameter setting -----
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU; //
	opts[1] = 1E-10; // gradient threshold, original 1e-15
	opts[2] = 1E-20; // relative para change threshold? original 1e-50
	opts[3] = 1E-20; // error threshold (below it, stop)
	opts[4] = LM_DIFF_DELTA;

	// ----- optimization parameters -----
	Data_MLEstimateLine3d_Compact data(line.pts);
	data.idx1 = idx_end1;
	data.idx2 = idx_end2;
	vector<double> paraVec, measVec;
	for(int i = 0; i < line.pts.size(); ++i) {	
			measVec.push_back(0);
		if (i == idx_end1 || i == idx_end2) {
			paraVec.push_back(line.pts[i].pos.x);
			paraVec.push_back(line.pts[i].pos.y);
			paraVec.push_back(line.pts[i].pos.z);			
		} 
	}
	int numPara = paraVec.size();
	double* para = new double[numPara];
	for (int i=0; i<numPara; ++i) {
		para[i] = paraVec[i];
	}
	int numMeas = measVec.size();
	double* meas = new double[numMeas];
	for ( int i=0; i<numMeas; ++i) {
		meas[i] = measVec[i];
	}
	// ----- start LM solver -----
	int ret = dlevmar_dif(costFun_MLEstimateLine3d_compact, para, meas, numPara, numMeas,
		maxIter, opts, info, NULL, NULL, (void*)&data);
//	termReason((int)info[6]);
//	cout<<endl<<endl;
	
	// ----- compute cov of MLE result -----
	cv::Mat cov_meas_inv = cv::Mat::zeros(3*line.pts.size(), 3*line.pts.size(), CV_64F);
	cv::Mat tmp;
	double* p = new double[line.pts.size()+4]; // mimic the full/long parameter vector
	int idx = 0;
	for(int i=0; i<line.pts.size();++i) {
		tmp = line.pts[i].cov.inv();
		tmp.copyTo(cov_meas_inv.rowRange(i*3,i*3+3).colRange(i*3,i*3+3));
		if(i==idx_end1) {
			p[idx] = para[0];
			p[idx+1] = para[1];
			p[idx+2] = para[2];
			idx = idx + 3;
		} else if(i==idx_end2){
			p[idx] = para[3];
			p[idx+1] = para[4];
			p[idx+2] = para[5];
			idx = idx + 3;
		}else {
			// project pt to line to get ratio
			p[idx] = 1- closest_3dpt_ratio_online_mah (line.pts[i], 
				cv::Point3d(para[0],para[1],para[2]), cv::Point3d(para[3],para[4],para[5]));			
			++idx;
		}		
	}

	// ---- compute line endpoint uncerainty ----
	MlestimateLine3dCov (p, line.pts.size()+4, idx_end1, idx_end2, cov_meas_inv, line.covA, line.covB);

	// refine line endpoint positions
	line.A = cv::Point3d (p[idx_end1],p[idx_end1+1],p[idx_end1+2]);
	line.B = cv::Point3d (p[idx_end2+2],p[idx_end2+3],p[idx_end2+4]);
	line.rndA = RandomPoint3d(line.A, line.covA);
	line.rndB = RandomPoint3d(line.B, line.covB);
#ifdef SAVE_MEMORY
	line.pts.clear();
#endif
//	timer.end();
//	count +=1;
//	acum += timer.time_ms;
//	if(count%50==0)	cout<<"mle-cmp accumulated: "<<acum<<endl;
	delete[] meas;
	delete[] para;
	delete[] p;
}


void MlestimateLine3dCov (double* p, int n, int i1, int i2, const cv::Mat& cov_meas_inv,
	cv::Mat& cov_i1, cv::Mat& cov_i2)
	// compute the covariance of the MLE of 3d line
	// input: p - MLE parameter vector 
	//		  n - dimension of p
	//		  i1,i2 - index of points used to represent 3d line (the i1-th and i2-th point of all points)
	//		  cov_meas - covariance of measurement, i.e. all collinear points 
	// output: cov_i1 - covariane of point i1
{	
	// compute jacobian
	cv::Mat J = cv::Mat::zeros(3*(n-4), n, CV_64F);
	
	J.at<double>(3*i1,i1) = 1;
	J.at<double>(3*i1+1,i1+1) = 1;
	J.at<double>(3*i1+2,i1+2) = 1;
	J.at<double>(3*i2,i2+2) = 1;
	J.at<double>(3*i2+1,i2+3) = 1;
	J.at<double>(3*i2+2,i2+4) = 1;
	for(int i=0; i<n-4; ++i) {
		if(i==i1 || i==i2) continue;
		int col =-1;
		if (i < i1) {
			col = i;
		}
		if (i > i1 && i < i2) {
			col = i + 2;
		}
		if (i > i2) {
			col = i + 4;
		}
		J.at<double>(3*i, i1) = p[col];
		J.at<double>(3*i+1, i1+1) = p[col];
		J.at<double>(3*i+2, i1+2) = p[col];
		J.at<double>(3*i, col) = p[i1] - p[i2+2];
		J.at<double>(3*i+1,col) = p[i1+1] - p[i2+3];
		J.at<double>(3*i+2,col) = p[i1+2] - p[i2+4];
		J.at<double>(3*i,i2+2) = 1-p[col];
		J.at<double>(3*i+1,i2+3) = 1-p[col];
		J.at<double>(3*i+2,i2+4) = 1-p[col];
	}

	cv::Mat cov_p = (J.t() * cov_meas_inv * J).inv();
	cov_i1 = cov_p.rowRange(i1,i1+3).colRange(i1,i1+3);
	cov_i2 = cov_p.rowRange(i2+2,i2+5).colRange(i2+2,i2+5);
	
	
	/*	cout<<"\nca0="<<cov_meas.rowRange(i1*3,i1*3+3).colRange(i1*3,i1*3+3)<<";"<<endl;
	cout<<"a="<<cv::Point3d(p[i1],p[i1+1],p[i1+2])<<";"<<endl;
	cout<<"ca1="<<cov_p.rowRange(i1,i1+3).colRange(i1,i1+3)<<";"<<endl;

	cout<<"cb0="<<cov_meas.rowRange(i2*3,i2*3+3).colRange(i2*3,i2*3+3)<<";"<<endl;
	cout<<"b="<<cv::Point3d(p[i2+2],p[i2+3],p[i2+4])<<";"<<endl;
	cout<<"cb1="<<cov_p.rowRange(i2+2,i2+5).colRange(i2+2,i2+5)<<";tmp;"<<endl;
	*/}


void termReason(int info)
{
	switch(info) {
	case 1:
		{cout<<"Termination reason 1: stopped by small gradient J^T e."<<endl;break;}
	case 2:
		{cout<<"Termination reason 2: stopped by small Dp."<<endl;break;}
	case 3:
		{cout<<"Termination reason 3: stopped by itmax."<<endl;break;}
	case 4:
		{cout<<"Termination reason 4: singular matrix. Restart from current p with increased mu."<<endl;break;}
	case 5:
		{cout<<"Termination reason 5: no further error reduction is possible. Restart with increased mu."<<endl;break;}
	case 6:
		{cout<<"Termination reason 6: stopped by small ||e||_2."<<endl;break;}
	case 7:
		{cout<<"Termination reason 7: stopped by invalid (i.e. NaN or Inf) 'func' values; a user error."<<endl;break;}
	default:
		{cout<<"Termination reason: Unknown..."<<endl;}
	}
}

double pt_to_line_dist2d(const cv::Point2d& p, double l[3])
	// distance from point(x,y) to line ax+by+c=0;
	// l=(a, b, c), p = (x,y)
{
	if(abs(l[0]*l[0]+l[1]*l[1]-1)>EPS) {
		cout<<"pt_to_line_dist2d: error,l should be normalized/initialized\n";
		exit(0);
	}
	double a = l[0],
		b = l[1],
		c = l[2],
		x = p.x,
		y = p.y;
	return abs((a*x+b*y+c))/sqrt(a*a+b*b);
}
double line_to_line_dist2d(FrameLine& a, FrameLine& b)
	// compute line to line distance by averaging 4 endpoint to line distances
	// a and b must be almost parallel to make any sense
{
	return 0.25*pt_to_line_dist2d(a.p,b.lineEq2d)
		+ 0.25*pt_to_line_dist2d(a.q, b.lineEq2d)
		+ 0.25*pt_to_line_dist2d(b.p, a.lineEq2d)
		+ 0.25*pt_to_line_dist2d(b.q, a.lineEq2d);
}

void trackLine (vector<FrameLine> f1, vector<FrameLine> f2, vector<vector<int> >& matches, vector<int>& addNew)
	// line segment tracking
	// input: 
	// finishing in <10 ms
{

	double lineDistThresh  = 25; // pixel
	double lineAngleThresh = 25 * PI/180; // 30 degree
	double desDiffThresh   = 0.85;
	double lineOverlapThresh = 3; // pixels
	double ratio_dist_1st2nd = 0.7;

	if(sysPara.fast_motion) {
		lineDistThresh = 45;
		lineAngleThresh = 30 * PI/180;
		desDiffThresh   = 0.85;
		lineOverlapThresh = -1;
	}

	if(sysPara.dark_ligthing) {
        lineDistThresh = 20;
        lineAngleThresh = 10 * PI/180;
        desDiffThresh = 1.5;
		ratio_dist_1st2nd = 0.85;
	}

	if(f1.size()==0 || f2.size()==0) {
		return;
	}

    cv::Mat desDiff = cv::Mat::zeros(f1.size(), f2.size(), CV_64F) + 100;
	#pragma omp  parallel for
	for(int i=0; i<f1.size(); ++i) {		
		for(int j=0; j<f2.size(); ++j) {
			if((f1[i].r.dot(f2[j].r) > cos(lineAngleThresh)) && // angle between gradients
				(line_to_line_dist2d(f1[i],f2[j]) < lineDistThresh) &&
				(lineSegmentOverlap(f1[i],f2[j]) > lineOverlapThresh )) // line (parallel) distance
			{
				desDiff.at<double>(i,j) = cv::norm(f1[i].des - f2[j].des);
			}
		}
	}

	for(int i=0; i<desDiff.rows; ++i) {
		vector<int> onePairIdx;
		double minVal;
		cv::Point minPos;
		cv::minMaxLoc(desDiff.row(i),&minVal,NULL,&minPos,NULL);
		if (minVal < desDiffThresh) {
			double minV;
			cv::Point minP;
			cv::minMaxLoc(desDiff.col(minPos.x),&minV,NULL,&minP,NULL);			
			if (i==minP.y) {    // commnent this for more potential matches 
				//further check distance ratio
				double rowmin2 = 100, colmin2 = 100;
				for(int j=0; j<desDiff.cols; ++j) {
					if (j == minPos.x) continue;
					if (rowmin2 > desDiff.at<double>(i,j)) {
						rowmin2 = desDiff.at<double>(i,j);
					}
				}
				for(int j=0; j<desDiff.rows; ++j) {
					if (j == minP.y) continue;
					if (colmin2 > desDiff.at<double>(j,minPos.x)) {
						colmin2 = desDiff.at<double>(j,minPos.x);
					}
				}
				if(rowmin2*ratio_dist_1st2nd > minVal && colmin2*ratio_dist_1st2nd > minVal) {
                    //find line matches
					onePairIdx.push_back(i);
					onePairIdx.push_back(minPos.x);
                    matches.push_back(onePairIdx);
                }else
                {
                    addNew.push_back(i);
                }
            }else
            {
                addNew.push_back(i);
            }
        }else
        {
            addNew.push_back(i);
        }
    }
}

double pt_to_line_dist3d(const cv::Point3d& p, const cv::Point3d& A, const cv::Point3d& B)
{
     cv::Point3d normal = (A - B) / sqrt((A - B).dot(A - B));
     cv::Point3d pA = (p - A);
     cv::Point3d dist_vector = normal.cross(pA);
     return sqrt(dist_vector.dot(dist_vector));
}

double line_to_line_dist3d(FrameLine& a, FrameLine& b)
{
    cv::Point3d midPoint_a = (a.line3d.A + a.line3d.B) / 2;
    double dist_mid_a = pt_to_line_dist3d(midPoint_a, b.line3d.A, b.line3d.B);
    cv::Point3d midPoint_b = (b.line3d.A + b.line3d.B) / 2;
    double dist_mid_b = pt_to_line_dist3d(midPoint_b, a.line3d.A, a.line3d.B);
    return dist_mid_a + dist_mid_b;
}
void trackLine3D (vector<FrameLine> f1, vector<FrameLine> f2, vector<vector<int> >& matches, vector<int>& addNew)
{
     double lineAngleThresh = 20 * PI/180;
     double lineDistThresh = 0.02;
     double desDiffThresh   = 0.85;

     if(sysPara.fast_motion) {
         lineAngleThresh = 30 * PI/180;
         lineDistThresh = 0.04;
         desDiffThresh   = 0.85;
         //lineOverlapThresh = -1;
     }

     if(sysPara.dark_ligthing) {
         lineAngleThresh = 10 * PI/180;
         lineDistThresh = 0.015;
         desDiffThresh = 1.5;
         //ratio_dist_1st2nd = 0.85;
     }

     cv::Mat desDiff = cv::Mat::zeros(f1.size(), f2.size(), CV_64F) + 100;
     for(int i = 0; i < f1.size(); ++i)
     {
         for(int j = 0; j < f2.size(); ++j)
         {
             double angle = 100; double dist = 200;
             if(f1[i].haveDepth && f2[j].haveDepth)
             {
                 cv::Point3d vector_AB_f1 = f1[i].line3d.A - f1[i].line3d.B;
                 cv::Point3d vector_AB_f2 = f2[j].line3d.A - f2[j].line3d.B;
                 angle = fabs(vector_AB_f1.dot(vector_AB_f2))/(sqrt(vector_AB_f1.dot(vector_AB_f1)) * sqrt(vector_AB_f2.dot(vector_AB_f2)));
                 dist = line_to_line_dist3d(f1[i], f2[j]);
                 if(angle > cos(lineAngleThresh) && dist < lineDistThresh)
                 {
                     desDiff.at<double>(i,j) = cv::norm(f1[i].des - f2[j].des);
                 }
             }
         }
     }

     for(int i=0; i<desDiff.rows; ++i) {
         vector<int> onePairIdx;
         double minVal;
         cv::Point minPos;
         cv::minMaxLoc(desDiff.row(i),&minVal,NULL,&minPos,NULL);
         if (minVal < desDiffThresh) {
             double minV;
             cv::Point minP;
             cv::minMaxLoc(desDiff.col(minPos.x),&minV,NULL,&minP,NULL);
             if (i==minP.y) {
                 onePairIdx.push_back(i);
                 onePairIdx.push_back(minPos.x);
                 matches.push_back(onePairIdx);
             }else
             {
                 addNew.push_back(i);
             }
         }else
         {
             addNew.push_back(i);
         }
     }
}

void matchLine (vector<FrameLine> f1, vector<FrameLine> f2, vector<vector<int> >& matches)
	// line segment matching (for loop closure)
	// input: 
	// finishing in <10 ms
{

	double lineDistThresh  = 80; // pixel
	double lineAngleThresh = 25 * PI/180; // 30 degree
	double desDiffThresh   = 0.7;
	double lineOverlapThresh = -1; // pixels
	double ratio_dist_1st2nd = 0.7;

	cv::Mat desDiff = cv::Mat::zeros(f1.size(), f2.size(), CV_64F)+100;
	#pragma omp  parallel for
	for(int i=0; i<f1.size(); ++i) {		
		for(int j=0; j<f2.size(); ++j) {
			if((f1[i].r.dot(f2[j].r) > cos(lineAngleThresh)) && // angle between gradients
				(line_to_line_dist2d(f1[i],f2[j]) < lineDistThresh) &&
				(lineSegmentOverlap(f1[i],f2[j]) > lineOverlapThresh )) // line (parallel) distance
			{
				desDiff.at<double>(i,j) = cv::norm(f1[i].des - f2[j].des);
			}
		}
	}
	
	for(int i=0; i<desDiff.rows; ++i) {
		vector<int> onePairIdx;
		double minVal;
		cv::Point minPos;
		cv::minMaxLoc(desDiff.row(i),&minVal,NULL,&minPos,NULL);
		if (minVal < desDiffThresh) {
			double minV;
			cv::Point minP;
			cv::minMaxLoc(desDiff.col(minPos.x),&minV,NULL,&minP,NULL);			
			if (i==minP.y) {    // commnent this for more potential matches 
				//further check distance ratio
				double rowmin2 = 100, colmin2 = 100;
				for(int j=0; j<desDiff.cols; ++j) {
					if (j == minPos.x) continue;
					if (rowmin2 > desDiff.at<double>(i,j)) {
						rowmin2 = desDiff.at<double>(i,j);
					}
				}
				for(int j=0; j<desDiff.rows; ++j) {
					if (j == minP.y) continue;
					if (colmin2 > desDiff.at<double>(j,minPos.x)) {
						colmin2 = desDiff.at<double>(j,minPos.x);
					}
				}
				if(rowmin2*ratio_dist_1st2nd > minVal && colmin2*ratio_dist_1st2nd > minVal) {
					onePairIdx.push_back(i);
					onePairIdx.push_back(minPos.x);
					matches.push_back(onePairIdx);
				}
			}
		}
	}	
}


int computeSubPSR (cv::Mat* xGradient, cv::Mat* yGradient,
	cv::Point2d p, double s, cv::Point2d g, vector<double>& vs) {
		/* input: p - 2D point position
		s - side length of square region
		g - unit vector of gradient of line 
		output: vs = (v1, v2, v3, v4)
		*/ 
		double tl_x = floor(p.x - s/2), tl_y = floor(p.y - s/2);
		if (tl_x < 0 || tl_y < 0 || 
			tl_x+s+1 > xGradient->cols || tl_y+s+1 > xGradient->rows)
			return 0; // out of image
		double v1=0, v2=0, v3=0, v4=0; 
		for (int x  = tl_x; x < tl_x+s; ++x) {
			for (int y = tl_y; y < tl_y+s; ++y) {
				//			cout<< xGradient->at<double>(y,x) <<","<<yGradient->at<double>(y,x) <<endl;
				//			cout<<"("<<y<<","<<x<<")"<<endl;
				double tmp1 = 
					xGradient->at<double>(y,x)*g.x + yGradient->at<double>(y,x)*g.y;
				double tmp2 = 
					xGradient->at<double>(y,x)*(-g.y) + yGradient->at<double>(y,x)*g.x;
				if ( tmp1 >= 0 )
					v1 = v1 + tmp1;
				else
					v2 = v2 - tmp1;
				if (tmp2 >= 0)
					v3 = v3 + tmp2;
				else
					v4 = v4 - tmp2;
			}
		}
		vs.resize(4);
		vs[0] = v1; vs[1] = v2;
		vs[2] = v3; vs[3] = v4;
		return 1;
}

int computeMSLD (FrameLine& l, cv::Mat* xGradient, cv::Mat* yGradient) 
	// compute msld and gradient
{	
	cv::Point2d gradient = l.getGradient(xGradient, yGradient);	
	l.r = gradient;
	int s = 5 * xGradient->cols/800.0;
	double len = cv::norm(l.p-l.q);

	vector<vector<double> > GDM; //GDM.reserve(2*(int)len);
	double step = 1; // the step length between sample points on line segment
	for (int i=0; i*step < len; ++i) {
		vector<double> col; col.reserve(9);
	//	col.clear();
		cv::Point2d pt =    // compute point position on the line
			l.p + (l.q - l.p) * (i*step/len);		
		bool fail = false;
		for (int j=-4; j <= 4; ++j ) { // 9 PSR for each point on line
			vector<double> psr(4);
			if (computeSubPSR (xGradient, yGradient, pt+j*s*gradient, s, gradient, psr)) {
				col.push_back(psr[0]);
				col.push_back(psr[1]);
				col.push_back(psr[2]);
				col.push_back(psr[3]);
			} else
				fail = true;
		}
		if (fail)
			continue;
		GDM.push_back(col);
	}

	cv::Mat MS(72, 1, CV_64F);
	if (GDM.size() ==0 ) {
		for (int i=0; i<MS.rows; ++i)
			MS.at<double>(i,0) = rand(); // if not computable, assign random num
		l.des = MS;
		return 0;
	}

	double gauss[9] = { 0.24142,0.30046,0.35127,0.38579,0.39804,
		0.38579,0.35127,0.30046,0.24142};
	for (int i=0; i < 36; ++i) {
		double sum=0, sum2=0, mean, std;
		for (int j=0; j < GDM.size(); ++j) {
			GDM[j][i] = GDM[j][i] * gauss[i/4];
			sum += GDM[j][i];
			sum2 += GDM[j][i]*GDM[j][i];
		}
		mean = sum/GDM.size();
		std = sqrt(abs(sum2/GDM.size() - mean*mean));
		MS.at<double>(i,0)		= mean;
		MS.at<double>(i+36, 0)	= std;
	}
	// normalize mean and std vector, respcectively
	MS.rowRange(0,36) = MS.rowRange(0,36) / cv::norm(MS.rowRange(0,36));
	MS.rowRange(36,72) = MS.rowRange(36,72) / cv::norm(MS.rowRange(36,72));
	for (int i=0; i < MS.rows; ++i) {
		if (MS.at<double>(i,0) > 0.4)
			MS.at<double>(i,0) = 0.4;
	}
	MS = MS/cv::norm(MS);
	l.des.create(72, 1, CV_64F);
	l.des = MS;
	return 1;
}

double projectPt2d_to_line2d(const cv::Point2d& X, const cv::Point2d& A, const cv::Point2d& B) 
{
	// X' = lambda*A + (1-lambda)*B; X' is the projection of X on AB
	cv::Point2d BX = X-B, BA = A-B;
	double lambda = BX.dot(BA)/cv::norm(BA)/cv::norm(BA);
	return lambda;
}

double lineSegmentOverlap(const FrameLine& a, const FrameLine& b)
	// compute the overlap length of two line segments in their parallel direction
{
	if(cv::norm(a.p-a.q) < cv::norm(b.p-b.q)) {// a is shorter than b
		double lambda_p = projectPt2d_to_line2d(a.p, b.p, b.q);
		double lambda_q = projectPt2d_to_line2d(a.q, b.p, b.q);
		if( (lambda_p < 0 && lambda_q < 0) || (lambda_p > 1 && lambda_q > 1) )
			return -1;
		else
			return abs(lambda_p - lambda_q) * cv::norm(b.p-b.q);
	} else {
		double lambda_p = projectPt2d_to_line2d(b.p, a.p, a.q);
		double lambda_q = projectPt2d_to_line2d(b.q, a.p, a.q);
		if( (lambda_p < 0 && lambda_q < 0) || (lambda_p > 1 && lambda_q > 1) )
			return -1;
		else
			return abs(lambda_p - lambda_q) * cv::norm(a.p-a.q);
	}
}


cv::Mat vec2SkewMat (cv::Mat vec)
{
	cv::Mat m = (cv::Mat_<double>(3,3) <<
		0, -vec.at<double>(2), vec.at<double>(1),
		vec.at<double>(2), 0, -vec.at<double>(0),
		-vec.at<double>(1), vec.at<double>(0), 0);
	return m;
}
cv::Mat vec2SkewMat (cv::Point3d vec)
{
	cv::Mat m = (cv::Mat_<double>(3,3) <<
		0, -vec.z, vec.y,
		vec.z, 0, -vec.x,
		-vec.y, vec.x, 0);
	return m;
}


cv::Mat q2r (cv::Mat q)
// input: unit quaternion representing rotation
// output: 3x3 rotation matrix
// note: q=(a,b,c,d)=a + b i + c j + d k, where (b,c,d) is the rotation axis
{
	double a = q.at<double>(0),	b = q.at<double>(1),
		c = q.at<double>(2), d = q.at<double>(3);
	double nm = sqrt(a*a+b*b+c*c+d*d);
		a = a/nm;
		b = b/nm;
		c = c/nm;
		d = d/nm;
	cv::Mat R = (cv::Mat_<double>(3,3)<< 
		a*a+b*b-c*c-d*d,	2*b*c-2*a*d,		2*b*d+2*a*c,
		2*b*c+2*a*d,		a*a-b*b+c*c-d*d,	2*c*d-2*a*b,
		2*b*d-2*a*c,		2*c*d+2*a*b,		a*a-b*b-c*c+d*d);
	return R.clone();
}
cv::Mat q2r (double* q)
	// input: unit quaternion representing rotation
	// output: 3x3 rotation matrix
	// note: q=(a,b,c,d)=a + b i + c j + d k, where (b,c,d) is the rotation axis
{
	double  a = q[0],	b = q[1],
			c = q[2],	d = q[3];
	double nm = sqrt(a*a+b*b+c*c+d*d);	
		a = a/nm;
		b = b/nm;
		c = c/nm;
		d = d/nm;
	cv::Mat R = (cv::Mat_<double>(3,3)<< 
		a*a+b*b-c*c-d*d,	2*b*c-2*a*d,		2*b*d+2*a*c,
		2*b*c+2*a*d,		a*a-b*b+c*c-d*d,	2*c*d-2*a*b,
		2*b*d-2*a*c,		2*c*d+2*a*b,		a*a-b*b-c*c+d*d);
	return R.clone();
}

cv::Mat r2q(cv::Mat R)
{	
	double t = R.at<double>(0,0)+R.at<double>(1,1)+R.at<double>(2,2);
	double r = sqrt(1+t);
	double s = 0.5/r;
	double w = 0.5*r; 
	double x = (R.at<double>(2,1)-R.at<double>(1,2))*s;
	double y = (R.at<double>(0,2)-R.at<double>(2,0))*s;
	double z = (R.at<double>(1,0)-R.at<double>(0,1))*s;
	cv::Mat q = (cv::Mat_<double>(4,1)<<w,x,y,z);
	return q;
}

//void write2file (Map3d& m, string suffix)
//{
	
//#ifdef OS_WIN
//	string prefix = m.datapath.substr(51, m.datapath.size()-51-1);
//	string fname = "./src/"+prefix+"_line"+suffix+".txt",
//		fname2 = "./src/"+prefix+"_lmk"+suffix+".txt";
//#else
//	//string prefix = m.datapath.substr(59, m.datapath.size()-59-1);
//	string prefix = "./Dataset/livo";
	
//	string fname = prefix+"_line"+suffix+".txt",
//		fname2 = prefix+"_lmk"+suffix+".txt";
//#endif
//	ofstream f(fname.c_str()), f2(fname2.c_str());
//	f.precision(16);
//	f.precision(16);
//	// === Note 1 ===
//	// The output format is [tx ty tz qx qy qz qw], following that of the RGBD-SLAM dataset groundtruth, as explained below.
//	// Letting t = [tx ty tz]', q = [qw qx qy qz], and R = q2r(q),
//	// then (R,t) transform a point from current-camera coordsys to world coordsys, i.e. X_w = R*X_c + t;
//	// So, the (R,t) pair here is DIFFERENT (actually, inverse) to conventional camera projection matrix
//	// === Note 2 ===
//	// The output quaternion format is different than what used in this c++ program.
//	for(int i=0; i<m.keyframeIdx.size();++i){
//		cv::Mat t = -m.frames[m.keyframeIdx[i]].R.t()*m.frames[m.keyframeIdx[i]].t;
//		cv::Mat q = r2q(m.frames[m.keyframeIdx[i]].R.t());
//		f<<m.frames[m.keyframeIdx[i]].timestamp<<'\t'
//		 <<t.at<double>(0)<<'\t'<<t.at<double>(1)<<'\t'<<t.at<double>(2)<<'\t'
//		 <<q.at<double>(1)<<'\t'<<q.at<double>(2)<<'\t'<<q.at<double>(3)<<'\t'<<q.at<double>(0)<<endl;
//	}
//	f.close();
//	//
//	for(int i=0; i<m.lmklines.size(); ++i) {
//		f2<<m.lmklines[i].A.x<<'\t'<<m.lmklines[i].A.y<<'\t'<<m.lmklines[i].A.z<<'\t'
//		  <<m.lmklines[i].B.x<<'\t'<<m.lmklines[i].B.y<<'\t'<<m.lmklines[i].B.z<<'\n';
//	}
//	f2.close();

//}

void write_linepairs_tofile(vector<RandomLine3d> a, vector<RandomLine3d> b, string fname, double timestamp)
{
	ofstream file(fname.c_str());
	file.precision(16);
	for(int i=0; i<a.size(); ++i) {		
		file<<a[i].A.x<<'\t'<<a[i].A.y<<'\t'<<a[i].A.z<<'\t'
			<<a[i].B.x<<'\t'<<a[i].B.y<<'\t'<<a[i].B.z<<'\t'
			<<b[i].A.x<<'\t'<<b[i].A.y<<'\t'<<b[i].A.z<<'\t'
			<<b[i].B.x<<'\t'<<b[i].B.y<<'\t'<<b[i].B.z<<'\t'<<timestamp<<endl;
	}
	file.close();
}

double pesudoHuber(double e, double band)
// pesudo Huber cost function
// e : error (distance)
// ouput: bounded e*e
{
	return 2*band*band*(sqrt(1+(e/band)*(e/band))-1);
}

double rotAngle (cv::Mat R)
{
	return acos(abs((R.at<double>(0,0) + R.at<double>(1,1) + R.at<double>(2,2) - 1)/2));
}

double ave_img_bright(cv::Mat img)
// (grayscale) image brightness average value
{
	cv::Mat gray;
	if(img.channels()>1)
	{
		cv::cvtColor(img, gray, CV_BGR2GRAY);
	} else {
		gray = img.clone();
	}
	cv::Scalar m = cv::mean(gray);
	return m.val[0];
}

bool get_pt_3d (cv::Point2d p2, cv::Point3d& p3, const cv::Mat& depth)
{
	if(p2.x<0 || p2.y<0 || p2.x >= depth.cols || p2.y >= depth.rows ) 
		return false;
	int row, col; // nearest pixel for pt
	if((floor(p2.x) == p2.x) && (floor(p2.y) == p2.y)) {// boundary issue
		col = max(int(p2.x-1),0);
		row = max(int(p2.y-1),0);
	} else {
		col = int(p2.x);
		row = int(p2.y);
	}

	if(depth.at<double>(row,col) < EPS) { // no depth info
		return false;
	} else {
		double zval = depth.at<double>(row,col)/5000; // in meter, z-value
		cv::Point2d tmp = mat2cvpt(K.inv()*cvpt2mat(p2))*zval;	
		p3.x = tmp.x;
		p3.y = tmp.y;
		p3.z = zval;
		return true;
	}
}

bool compute_motion_given_ptpair_file (string filename, const cv::Mat& depth, cv::Mat& R, cv::Mat& t)
{
	ifstream infile(filename.c_str());
	if(!infile.good())
		return false;
	string sx0, sy0, sx1, sy1;
	vector<cv::Point2d> pts0, pts1;
	while(infile>>sx0>>sy0>>sx1>>sy1) {
		cv::Point2d p0(atof(sx0.c_str()), atof(sy0.c_str()));
		pts0.push_back(p0);
		cv::Point2d p1(atof(sx1.c_str()), atof(sy1.c_str()));
		pts1.push_back(p1);
	}

	vector<cv::Point3d> PTS0, PTS1;
	for(int i=0; i<pts0.size(); ++i) {
		cv::Point3d Pt0, Pt1;
		if (get_pt_3d (pts0[i], Pt0, depth) && get_pt_3d (pts1[i], Pt1, depth)) {
			PTS0.push_back(Pt0);
			PTS1.push_back(Pt1);
		}
	}
	if(PTS0.size()<3) return false;
	
	vector<int> inlier = computeRelativeMotion_Ransac(PTS0, PTS1, R, t, 0.02);
	if(inlier.size()<3)
		return false;
	return true;
}

}
