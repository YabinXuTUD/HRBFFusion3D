#include "utils.h"
extern cv::Mat K, distCoeffs;
extern SystemParameters sysPara;
#define OPT_USE_MAHDIST

namespace Line3D{

struct Data_optimizeRelmotion
{
	vector<RandomLine3d>& a;
	vector<RandomLine3d>& b;
	Data_optimizeRelmotion(vector<RandomLine3d>&ina, vector<RandomLine3d>& inb):a(ina),b(inb){}
};
void costFun_optimizeRelmotion(double *p, double *error, int m, int n, void *adata)
{
//	MyTimer t; t.start();
	struct Data_optimizeRelmotion* dptr;
	dptr = (struct Data_optimizeRelmotion *) adata;
	cv::Mat R = q2r(p);
	cv::Mat t = cv::Mat(3,1,CV_64F, &p[4]);// (cv::Mat_<double>(3,1)<<p[4],p[5],p[6]);
	double cost = 0;
	#pragma omp  parallel for
	for(int i=0; i< dptr->a.size() ; ++i)	{
#ifdef OPT_USE_MAHDIST
	/*	error[i] = 0.25*(mah_dist3d_pt_line(dptr->b[i].rndA, R*cvpt2mat(dptr->a[i].A,0)+t, R*cvpt2mat(dptr->a[i].B,0)+t)+
				   mah_dist3d_pt_line(dptr->b[i].rndB, R*cvpt2mat(dptr->a[i].A,0)+t, R*cvpt2mat(dptr->a[i].B,0)+t)+
				   mah_dist3d_pt_line(dptr->a[i].rndA, R.t()*(cvpt2mat(dptr->b[i].A,0)-t), R.t()*(cvpt2mat(dptr->b[i].B,0)-t))+
				   mah_dist3d_pt_line(dptr->a[i].rndB, R.t()*(cvpt2mat(dptr->b[i].A,0)-t), R.t()*(cvpt2mat(dptr->b[i].B,0)-t)));
	*/
		// faster computing than above
		double aiA[3] = {dptr->a[i].A.x,dptr->a[i].A.y,dptr->a[i].A.z},
			aiB[3] = {dptr->a[i].B.x,dptr->a[i].B.y,dptr->a[i].B.z},
			biA[3] = {dptr->b[i].A.x,dptr->b[i].A.y,dptr->b[i].A.z},
			biB[3] = {dptr->b[i].B.x,dptr->b[i].B.y,dptr->b[i].B.z};
		error[i] = 0.25*(mah_dist3d_pt_line(dptr->b[i].rndA, R*array2mat(aiA,3)+t, R*array2mat(aiB,3)+t)+
				   mah_dist3d_pt_line(dptr->b[i].rndB, R*array2mat(aiA,3)+t, R*array2mat(aiB,3)+t)+
				   mah_dist3d_pt_line(dptr->a[i].rndA, R.t()*(array2mat(biA,3)-t), R.t()*(array2mat(biB,3)-t))+
				   mah_dist3d_pt_line(dptr->a[i].rndB, R.t()*(array2mat(biA,3)-t), R.t()*(array2mat(biB,3)-t)));
	
#else
		error[i]= 0.25*(dist3d_pt_line(dptr->b[i].A, mat2cvpt3d(R*cvpt2mat(dptr->a[i].A,0)+t), mat2cvpt3d(R*cvpt2mat(dptr->a[i].B,0)+t))
				+ dist3d_pt_line(dptr->b[i].B, mat2cvpt3d(R*cvpt2mat(dptr->a[i].A,0)+t), mat2cvpt3d(R*cvpt2mat(dptr->a[i].B,0)+t))
				+ dist3d_pt_line(dptr->a[i].A, mat2cvpt3d(R.t()*(cvpt2mat(dptr->b[i].A,0)-t)), mat2cvpt3d(R.t()*(cvpt2mat(dptr->b[i].B,0)-t)))
				+ dist3d_pt_line(dptr->a[i].B, mat2cvpt3d(R.t()*(cvpt2mat(dptr->b[i].A,0)-t)), mat2cvpt3d(R.t()*(cvpt2mat(dptr->b[i].B,0)-t))));
	
#endif
	}
}

void optimizeRelmotion(vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t)
{
	
	cv::Mat q = r2q(R);
	// ----- LM parameter setting -----
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU; //
	opts[1] = 1E-10; // gradient threshold, original 1e-15
	opts[2] = 1E-20; // relative para change threshold? original 1e-50
	opts[3] = 1E-20; // error threshold (below it, stop)
	opts[4] = LM_DIFF_DELTA;
	int maxIter = 500;

	// ----- optimization parameters -----
	int numPara = 7;
	double* para = new double[numPara];
	para[0] = q.at<double>(0);
	para[1] = q.at<double>(1);
	para[2] = q.at<double>(2);
	para[3] = q.at<double>(3);
	para[4] = t.at<double>(0);
	para[5] = t.at<double>(1);
	para[6] = t.at<double>(2);
	
	// ----- measurements -----
	int numMeas = a.size();
	double* meas = new double[numMeas];
	for(int i=0; i<numMeas; ++i) meas[i] = 0;

	Data_optimizeRelmotion data(a,b);
	// ----- start LM solver -----
		MyTimer timer; 	timer.start();
	int ret = dlevmar_dif(costFun_optimizeRelmotion, para, meas, numPara, numMeas,
							maxIter, opts, info, NULL, NULL, (void*)&data);
	//	timer.end();	cout<<"optimizeRelmotion Time used: "<<timer.time_ms<<" ms. "<<endl;
	
	R = q2r(para);
	t = (cv::Mat_<double>(3,1)<<para[4],para[5],para[6]);
	delete[] meas;
	delete[] para;

}


void computeRelativeMotion_svd (vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& R, cv::Mat& t)
	// input needs at least 2 correspondences of non-parallel lines
	// the resulting R and t works as below: x'=Rx+t for point pair(x,x');
{
	if(a.size()<2)	{
		cerr<<"Error in computeRelativeMotion_svd: input needs at least 2 pairs!\n";
		return;
	}
	cv::Mat A = cv::Mat::zeros(4,4,CV_64F);
	for(int i=0; i<a.size(); ++i) {
		cv::Mat Ai = cv::Mat::zeros(4,4,CV_64F);
		Ai.at<double>(0,1) = (a[i].u-b[i].u).x;
		Ai.at<double>(0,2) = (a[i].u-b[i].u).y;
		Ai.at<double>(0,3) = (a[i].u-b[i].u).z;
		Ai.at<double>(1,0) = (b[i].u-a[i].u).x;
		Ai.at<double>(2,0) = (b[i].u-a[i].u).y;
		Ai.at<double>(3,0) = (b[i].u-a[i].u).z;
		vec2SkewMat(a[i].u+b[i].u).copyTo(Ai.rowRange(1,4).colRange(1,4));
		A = A + Ai.t()*Ai;
	}
	cv::SVD svd(A);
	cv::Mat q = svd.u.col(3);
	//	cout<<"q="<<q<<endl;
	R = q2r(q);
	cv::Mat uu = cv::Mat::zeros(3,3,CV_64F),
		udr= cv::Mat::zeros(3,1,CV_64F);
	for(int i=0; i<a.size(); ++i) {
		uu = uu + vec2SkewMat(b[i].u)*vec2SkewMat(b[i].u).t();
		udr = udr + vec2SkewMat(b[i].u).t()* (cvpt2mat(b[i].d,0)-R*cvpt2mat(a[i].d,0));
	}
	t = uu.inv()*udr;	
}

vector<int> computeRelativeMotion_Ransac (vector<RandomLine3d> a, vector<RandomLine3d> b, cv::Mat& Ro, cv::Mat& to)
	// compute relative pose between two cameras using 3d line correspondences
	// ransac
{
	if (a.size()<3) {		
		return vector<int>();
	}
//	MyTimer t1; t1.start();
	// convert to the representation of Zhang's paper
	vector<vector<double> > aA(a.size()), aB(a.size()), aAB(a.size()), bAB(a.size());
	for(int i=0; i<a.size(); ++i) {
		cv::Point3d l = a[i].B - a[i].A;
		cv::Point3d m = (a[i].A + a[i].B) * 0.5;
		a[i].u = l * (1/cv::norm(l));
		a[i].d = a[i].u.cross(m);
		l = b[i].B - b[i].A;
		m = (b[i].A + b[i].B) * 0.5;
		b[i].u = l * (1/cv::norm(l));
		b[i].d = b[i].u.cross(m);
		aA[i].resize(3);
		aB[i].resize(3);
		aAB[i].resize(3);
		bAB[i].resize(3);
		aA[i][0] = a[i].A.x;
		aA[i][1] = a[i].A.y;
		aA[i][2] = a[i].A.z;		
		aB[i][0] = a[i].B.x;
		aB[i][1] = a[i].B.y;
		aB[i][2] = a[i].B.z;
		aAB[i][0] = a[i].A.x-a[i].B.x;
		aAB[i][1] = a[i].A.y-a[i].B.y;
		aAB[i][2] = a[i].A.z-a[i].B.z;
		bAB[i][0] = b[i].A.x-b[i].B.x;
		bAB[i][1] = b[i].A.y-b[i].B.y;
		bAB[i][2] = b[i].A.z-b[i].B.z;
	}	

	// ----- start ransac -----
	int minSolSetSize = 3, maxIters = 500;
	double distThresh = sysPara.pt2line3d_dist_relmotion; // in meter
	double angThresh  = sysPara.line3d_angle_relmotion; // deg
	double lineAngleThresh_degeneracy = 5*PI/180; //5 degree

	vector<int> indexes;
	for(int i=0; i<a.size(); ++i)	indexes.push_back(i);
	int iter = 0;
	vector<int> maxConSet;
	cv::Mat bR, bt;
	while(iter<maxIters) {
		vector<int> inlier;
		iter++;
		random_unique(indexes.begin(), indexes.end(),minSolSetSize);// shuffle
		
		vector<RandomLine3d> suba, subb;
		for(int i=0; i<minSolSetSize; ++i) {
			suba.push_back(a[indexes[i]]);
			subb.push_back(b[indexes[i]]);
		}
		// ---- check degeneracy ---- 	
		bool degenerate = true;
		// if at least one pair is not parallel, then it's non-degenerate
		for(int i=0; i<minSolSetSize; ++i){
			for(int j=i+1; j<minSolSetSize; ++j) {
				if(abs(suba[i].u.dot(suba[j].u)) < cos(lineAngleThresh_degeneracy)) {
					degenerate = false;
					break;
				}
			}
			if(!degenerate)
				break;
		}
		if(degenerate) continue; // degenerate set is not usable

		cv::Mat R, t;
		computeRelativeMotion_svd(suba, subb, R, t);
		// find consensus
		for(int i=0; i<a.size(); ++i) {
	/*		double aiA[3] = {a[i].A.x,a[i].A.y,a[i].A.z},
			aiB[3] = {a[i].B.x,a[i].B.y,a[i].B.z},
			aiA_B[3] = {(a[i].A-a[i].B).x,(a[i].A-a[i].B).y,(a[i].A-a[i].B).z},
			biA_B[3] = {(b[i].A-b[i].B).x,(b[i].A-b[i].B).y,(b[i].A-b[i].B).z};
			double dist = 0.5*dist3d_pt_line (mat2cvpt3d(R*array2mat(aiA,3)+t), b[i].A, b[i].B)
						+ 0.5*dist3d_pt_line (mat2cvpt3d(R*array2mat(aiB,3)+t), b[i].A, b[i].B);
			double angle = 180*acos(abs((R*array2mat(aiA_B, 3)).dot(array2mat(biA_B,3))/
							cv::norm(a[i].A - a[i].B)/cv::norm(b[i].A - b[i].B)))/PI; // degree
	*/		
			double dist = 0.5*dist3d_pt_line (mat2cvpt3d(R*array2mat(&aA[i][0],3)+t), b[i].A, b[i].B)
						+ 0.5*dist3d_pt_line (mat2cvpt3d(R*array2mat(&aB[i][0],3)+t), b[i].A, b[i].B);
			double angle = 180*acos(abs((R*array2mat(&aAB[i][0], 3)).dot(array2mat(&bAB[i][0],3))/
							cv::norm(a[i].A - a[i].B)/cv::norm(b[i].A - b[i].B)))/PI; 
			if(dist < distThresh && angle < angThresh) {		
				inlier.push_back(i);
			}			
		}
		if(inlier.size() > maxConSet.size()) {
			maxConSet = inlier;
			bR = R;
			bt = t;
		}

	}
	if(maxConSet.size()<1)
		return maxConSet;
	Ro = bR; to = bt;
	if(maxConSet.size()<4)
		return maxConSet;
	// ---- apply svd to all inliers ----
	vector<RandomLine3d> ina, inb;
	for(int i=0; i<maxConSet.size();++i) {
		ina.push_back(a[maxConSet[i]]);
		inb.push_back(b[maxConSet[i]]);
	}

//	computeRelativeMotion_svd(ina, inb, Ro, to);
	optimizeRelmotion(ina, inb, Ro, to);
	cv::Mat R = Ro, t = to;
	vector<int> prevConSet;
	while(1) {		
		vector<int> conset;
		for(int i=0; i<a.size(); ++i) {
		/*	double dist = 0.5*dist3d_pt_line (mat2cvpt3d(R*cvpt2mat(a[i].A,0)+t), b[i].A, b[i].B)
				+ 0.5*dist3d_pt_line (mat2cvpt3d(R*cvpt2mat(a[i].B,0)+t), b[i].A, b[i].B);
			double angle = 180*acos(abs((R*cvpt2mat(a[i].A - a[i].B, 0)).dot(cvpt2mat(b[i].A-b[i].B,0))/
				cv::norm(a[i].A - a[i].B)/cv::norm(b[i].A - b[i].B)))/PI; // degree
			double aiA[3] = {a[i].A.x,a[i].A.y,a[i].A.z},
			aiB[3] = {a[i].B.x,a[i].B.y,a[i].B.z},
			aiA_B[3] = {(a[i].A-a[i].B).x,(a[i].A-a[i].B).y,(a[i].A-a[i].B).z},
			biA_B[3] = {(b[i].A-b[i].B).x,(b[i].A-b[i].B).y,(b[i].A-b[i].B).z};
		*/	double dist = 0.5*dist3d_pt_line (mat2cvpt3d(R*array2mat(&aA[i][0],3)+t), b[i].A, b[i].B)
						+ 0.5*dist3d_pt_line (mat2cvpt3d(R*array2mat(&aB[i][0],3)+t), b[i].A, b[i].B);
			double angle = 180*acos(abs((R*array2mat(&aAB[i][0], 3)).dot(array2mat(&bAB[i][0],3))/
							cv::norm(a[i].A - a[i].B)/cv::norm(b[i].A - b[i].B)))/PI; 
			if(dist < distThresh && angle < angThresh) {
				conset.push_back(i);
			}
		}
		if(conset.size() <= prevConSet.size()) 
			break;
		else {
			prevConSet = conset;
			Ro = R;
			to = t;
	//		cout<<Ro<<endl<<to<<'\t'<<prevConSet.size()<<endl<<endl;
			ina.clear(); inb.clear();
			for(int i=0; i<prevConSet.size();++i) {
				ina.push_back(a[prevConSet[i]]);
				inb.push_back(b[prevConSet[i]]);
			}
			optimizeRelmotion(ina, inb, R, t);
		}	
	}
	return prevConSet;
}

bool computeRelativeMotion_svd (vector<cv::Point3d> a, vector<cv::Point3d> b, cv::Mat& R, cv::Mat& t)
{
	int n = a.size();
	if (n<3) {
		return false;
	}
	// compute centroid
	cv::Point3d ac(0,0,0), bc(0,0,0);
	for (int i=0; i<n; ++i) {
		ac += a[i];
		bc += b[i];
	}
	ac = ac * (1.0/n);
	bc = bc * (1.0/n);

	cv::Mat H = cv::Mat::zeros(3,3,CV_64F);
	for(int i=0; i<n; ++i) {
		H = H + cvpt2mat(a[i]-ac, 0) * cvpt2mat(b[i]-bc,0).t();
	}
	cv::SVD svd(H);
	R = (svd.vt * svd.u).t();
	if(cv::determinant(R)<0)
		R.col(2) = R.col(2) * -1;

	t = -R*cvpt2mat(ac,0) + cvpt2mat(bc,0);
	return true;
}

void optimizeRelmotion(vector<cv::Point3d> a, vector<cv::Point3d> b, cv::Mat& R, cv::Mat& t);
vector<int> computeRelativeMotion_Ransac (vector<cv::Point3d> a, vector<cv::Point3d> b, cv::Mat& R, cv::Mat& t, double thresh)
{
	// ----- start ransac -----
	int minSolSetSize = 3, maxIters = 500;
	double distThresh = thresh; // in meter

	vector<int> indexes;
	for(int i=0; i<a.size(); ++i)	indexes.push_back(i);
	int iter = 0;
	vector<int> maxConSet;
	cv::Mat bR, bt;
	while(iter<maxIters) {
		vector<int> inlier;
		iter++;
		random_unique(indexes.begin(), indexes.end(),minSolSetSize);// shuffle
		vector<cv::Point3d> suba, subb;
		for(int i=0; i<minSolSetSize; ++i){
			suba.push_back(a[indexes[i]]);
			subb.push_back(b[indexes[i]]);
		}
		cv::Mat Ri, ti;
		if(!computeRelativeMotion_svd(suba, subb, Ri, ti)) continue;
		for(int i=0; i<a.size(); ++i){
			if(cv::norm(Ri * cvpt2mat(a[i],0) + ti - cvpt2mat(b[i],0)) < distThresh) {
				inlier.push_back(i);
			}
		}
		if(inlier.size()>maxConSet.size()) {
			bR = Ri.clone();
			bt = ti.clone();
			maxConSet = inlier;
		}
	}
	if(maxConSet.size()>=3) {
		vector<cv::Point3d> ina, inb;
		for(int i=0; i<maxConSet.size();++i) {
			ina.push_back(a[maxConSet[i]]);
			inb.push_back(b[maxConSet[i]]);
		}
		optimizeRelmotion(ina, inb, bR, bt);
	}
	R = bR.clone();
	t = bt.clone();
	
	return maxConSet;
}

struct Data_optimizeRelmotion_pts
{
	vector<cv::Point3d>& a;
	vector<cv::Point3d>& b;
	Data_optimizeRelmotion_pts(vector<cv::Point3d>&ina, vector<cv::Point3d>& inb):a(ina),b(inb){}
};
void costFun_optimizeRelmotion_pts(double *p, double *error, int m, int n, void *adata)
{
	struct Data_optimizeRelmotion_pts* dptr;
	dptr = (struct Data_optimizeRelmotion_pts *) adata;
	cv::Mat R = q2r(p);
	cv::Mat t = cv::Mat(3,1,CV_64F, &p[4]);// (cv::Mat_<double>(3,1)<<p[4],p[5],p[6]);
	double cost = 0;
	#pragma omp  parallel for
	for(int i=0; i<dptr->a.size(); ++i)	{
		error[i]= cv::norm(R*cvpt2mat(dptr->a[i],0)+t-cvpt2mat(dptr->b[i],0));
		cost += error[i]*error[i];
	}
//	cout<<cost<<"\t";
}

void optimizeRelmotion(vector<cv::Point3d> a, vector<cv::Point3d> b, cv::Mat& R, cv::Mat& t)
{
	cv::Mat q = r2q(R);
	// ----- LM parameter setting -----
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU; //
	opts[1] = 1E-10; // gradient threshold, original 1e-15
	opts[2] = 1E-20; // relative para change threshold? original 1e-50
	opts[3] = 1E-20; // error threshold (below it, stop)
	opts[4] = LM_DIFF_DELTA;
	int maxIter = 500;

	// ----- optimization parameters -----
	int numPara = 7;
	double* para = new double[numPara];
	para[0] = q.at<double>(0);
	para[1] = q.at<double>(1);
	para[2] = q.at<double>(2);
	para[3] = q.at<double>(3);
	para[4] = t.at<double>(0);
	para[5] = t.at<double>(1);
	para[6] = t.at<double>(2);
	
	// ----- measurements -----
	int numMeas = a.size();
	double* meas = new double[numMeas];
	for(int i=0; i<numMeas; ++i) meas[i] = 0;

	Data_optimizeRelmotion_pts data(a,b);
	// ----- start LM solver -----
	//	MyTimer timer; 	timer.start();
	int ret = dlevmar_dif(costFun_optimizeRelmotion_pts, para, meas, numPara, numMeas,
							maxIter, opts, info, NULL, NULL, (void*)&data);
	//	timer.end();	cout<<" Time used: "<<timer.time_ms<<" ms. "<<endl;
	//	termReason((int)info[6]);
	

	//	cv::Mat q = (cv::Mat_<double>(4,1)<<p[0],p[1],p[2],p[3]);
	R = q2r(para);
	t = (cv::Mat_<double>(3,1)<<para[4],para[5],para[6]);
	delete[] meas;
	delete[] para;

}

}
