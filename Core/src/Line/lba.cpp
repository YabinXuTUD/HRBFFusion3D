#include "lineslam.h"
#include "utils.h"

#define LINEDIST_6D // use 6-d error for line distance

extern cv::Mat K, distCoeffs;
extern SystemParameters sysPara;

struct Data_LBA
{
	int						numView, frontPosIdx, frontFrmIdx;
	vector<int>				idx2Opt;    // idx of key points to optimize
	vector<int>				idx2Rpj_notOpt; 
	vector<int>				kfidx;
	vector<Frame>&			frames;
	vector<LmkLine>&		lines;
	vector<double>			ref, ref_t;

	double* ms; // for debugging puropse

	Data_LBA(vector<Frame>&	frm, vector<LmkLine>& ln) : frames(frm), lines(ln) {}

};

void costFun_LBA(double *p, double *error, int numPara, int numMeas, void *adata)
{
	struct Data_LBA* dp = (struct Data_LBA *) adata;
	double kernel = 5;
	vector<double> ref = dp->ref;
	vector<double> ref_t = dp->ref_t;
	// ----- recover parameters for each view and landmark -----
	// ---- pose para ----
	int pidx = 0;
	vector<cv::Mat>  Rs(dp->frames.size()), ts(dp->frames.size()); // projection matrices
	for (int i = dp->frontFrmIdx; i < dp->frontPosIdx; ++i)	{
		Rs[dp->kfidx[i]] = dp->frames[dp->kfidx[i]].R;
		ts[dp->kfidx[i]] = dp->frames[dp->kfidx[i]].t;
	}
	Rs[0] = cv::Mat::eye(3,3, CV_64F);
	ts[0] = cv::Mat::zeros(3,1,CV_64F);
	for (int i = dp->frontPosIdx; i < dp->kfidx.size(); ++i) {
		cv::Mat Ri = q2r(&p[pidx]);
		cv::Mat ti;
		if (dp->frontPosIdx<=1) {
			ti = (cv::Mat_<double>(3,1)<<p[pidx+4],p[pidx+5],p[pidx+6]);
		} else {
			ti = (cv::Mat_<double>(3,1)<<p[pidx+4] + ref_t[0],
				p[pidx+5] + ref_t[1], p[pidx+6] + ref_t[2]); //....................
		}
		pidx = pidx + 7;
		Rs[dp->kfidx[i]] = Ri.clone();
		ts[dp->kfidx[i]] = ti.clone();
	}
		
	// ----- initialize error to be 0 -----
	for(int i=0; i<numMeas; ++i) 
		error[i] = 0;
	// ----- reproject features -----		
	int eidx = 0;
	for(int i=0; i < dp->idx2Opt.size(); ++i) {
		int idx =  dp->idx2Opt[i];
		cv::Mat edpA = (cv::Mat_<double>(3,1)<<p[pidx]+ref[0],p[pidx+1]+ref[1],p[pidx+2]+ref[2]);
		cv::Mat edpB = (cv::Mat_<double>(3,1)<<p[pidx+3]+ref[0],p[pidx+4]+ref[1],p[pidx+5]+ref[2]);
		for(int j=0; j < dp->lines[idx].frmId_lnLid.size(); ++j) {
			int fid = dp->lines[idx].frmId_lnLid[j][0];
			int lid = dp->lines[idx].frmId_lnLid[j][1];
			if (dp->frames[fid].isKeyFrame && fid >= dp->kfidx[dp->frontFrmIdx]) {
#ifdef LINEDIST_6D
				cv::Point3d ev1 = mahvec_3d_pt_line(dp->frames[fid].lines[lid].line3d.rndA, Rs[fid]*edpA + ts[fid], Rs[fid]*edpB + ts[fid]);
				error[eidx] = ev1.x;
				error[eidx+1] = ev1.y;
				error[eidx+2] = ev1.z;
				cv::Point3d ev2 = mahvec_3d_pt_line(dp->frames[fid].lines[lid].line3d.rndB, Rs[fid]*edpA + ts[fid], Rs[fid]*edpB + ts[fid]);
				error[eidx+3] = ev2.x;
				error[eidx+4] = ev2.y;
				error[eidx+5] = ev2.z;
				eidx += 6;
#else
				error[eidx] = mah_dist3d_pt_line(dp->frames[fid].lines[lid].line3d.rndA, Rs[fid]*edpA + ts[fid], Rs[fid]*edpB + ts[fid]);
				error[eidx+1] = mah_dist3d_pt_line(dp->frames[fid].lines[lid].line3d.rndB, Rs[fid]*edpA + ts[fid], Rs[fid]*edpB + ts[fid]);
		//		error[eidx] = sqrt(pesudoHuber(error[eidx], kernel));
		//		error[eidx+1] = sqrt(pesudoHuber(error[eidx+1], kernel));
				eidx = eidx + 2;
#endif
			}
		}
		pidx = pidx + 6;
	}
	for(int i=0; i < dp->idx2Rpj_notOpt.size(); ++i) {
		int idx =  dp->idx2Rpj_notOpt[i];
		cv::Mat edpA = cvpt2mat(dp->lines[idx].A,0);
		cv::Mat edpB = cvpt2mat(dp->lines[idx].B,0);
		for(int j=0; j < dp->lines[idx].frmId_lnLid.size(); ++j) {
			int fid = dp->lines[idx].frmId_lnLid[j][0];
			int lid = dp->lines[idx].frmId_lnLid[j][1];
			if (fid >= dp->kfidx[dp->frontFrmIdx]) {
#ifdef LINEDIST_6D
				cv::Point3d ev1 = mahvec_3d_pt_line(dp->frames[fid].lines[lid].line3d.rndA, Rs[fid]*edpA + ts[fid], Rs[fid]*edpB + ts[fid]);
				error[eidx] = ev1.x;
				error[eidx+1] = ev1.y;
				error[eidx+2] = ev1.z;
				cv::Point3d ev2 = mahvec_3d_pt_line(dp->frames[fid].lines[lid].line3d.rndB, Rs[fid]*edpA + ts[fid], Rs[fid]*edpB + ts[fid]);
				error[eidx+3] = ev2.x;
				error[eidx+4] = ev2.y;
				error[eidx+5] = ev2.z;
				eidx += 6;
#else
				error[eidx] = mah_dist3d_pt_line(dp->frames[fid].lines[lid].line3d.rndA, Rs[fid]*edpA + ts[fid], Rs[fid]*edpB + ts[fid]);
				error[eidx+1] = mah_dist3d_pt_line(dp->frames[fid].lines[lid].line3d.rndB, Rs[fid]*edpA + ts[fid], Rs[fid]*edpB + ts[fid]);
		//		error[eidx] = sqrt(pesudoHuber(error[eidx], kernel));
		//		error[eidx+1] = sqrt(pesudoHuber(error[eidx+1], kernel));
				eidx = eidx + 2;
#endif
			}
		}
	}
	
	double cost = 0;
	for(int i=0; i < numMeas; ++i) {
		cost = cost + error[i]*error[i];
	}
	static int count = 0;
	count++;
	if(!(count % 100)) {
		cout << cost<<"\t"; 
	}
}

#ifdef SLAM_LBA
void Map3d::lba (int numPos, int numFrm)
// local bundle adjustment
{
	// ----- BA setting -----
	// Note: numFrm should be larger or equal to numPos+2, to fix scale

	// ----- LM parameter setting -----
	double opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	opts[0] = LM_INIT_MU; //
	opts[1] = 1E-10; // gradient threshold, original 1e-15
	opts[2] = 1E-50; // relative para change threshold? original 1e-50
	opts[3] = 1E-20; // error threshold (below it, stop)
	opts[4] = LM_DIFF_DELTA;
	int maxIter = 500;

	int frontPosIdx = max(1, (int)keyframeIdx.size() - numPos);
	int frontFrmIdx = max(0, (int)keyframeIdx.size() - numFrm);
	
	// ----- Relative frame -----
	cv::Mat refmat = -frames[keyframeIdx[frontFrmIdx]].R.t()*frames[keyframeIdx[frontFrmIdx]].t;
	vector<double> ref(3);
	ref[0] = refmat.at<double>(0);
	ref[1] = refmat.at<double>(1);
	ref[2] = refmat.at<double>(2);

	vector<double> ref_t(3);
	ref_t[0] = frames[keyframeIdx[frontFrmIdx]].t.at<double>(0);
	ref_t[1] = frames[keyframeIdx[frontFrmIdx]].t.at<double>(1);
	ref_t[2] = frames[keyframeIdx[frontFrmIdx]].t.at<double>(2);


	// ----- optimization parameters -----
	vector<double> paraVec; 
	// ---- camera pose parameters ----
	for(int i = frontPosIdx; i < keyframeIdx.size(); ++i) {	
		cv::Mat qi = r2q(frames[keyframeIdx[i]].R);
		paraVec.push_back(qi.at<double>(0));
		paraVec.push_back(qi.at<double>(1));
		paraVec.push_back(qi.at<double>(2));
		paraVec.push_back(qi.at<double>(3));
		if ( frontPosIdx <= 1) {
			paraVec.push_back(frames[keyframeIdx[i]].t.at<double>(0));
			paraVec.push_back(frames[keyframeIdx[i]].t.at<double>(1));
			paraVec.push_back(frames[keyframeIdx[i]].t.at<double>(2));
		} else {
			paraVec.push_back(frames[keyframeIdx[i]].t.at<double>(0) - ref_t[0]); //....................
			paraVec.push_back(frames[keyframeIdx[i]].t.at<double>(1) - ref_t[1]);     // ...................
			paraVec.push_back(frames[keyframeIdx[i]].t.at<double>(2) - ref_t[2]);
		}
	}
	// ---- structure parameters ----
	vector<int> idx2Opt; // landmark idx to optimize 
	vector<int> idx2Rpj_notOpt; // landmarkt idx to reproject but not optimize
	// landmark-to-optimize contains those first appearing after frontFrmIdx and still being observed after frontPosIdx
	for(int i=0; i < lmklines.size(); ++i) {
		for(int j=0; j < lmklines[i].frmId_lnLid.size(); ++j) {
			if (lmklines[i].frmId_lnLid[j][0] >= keyframeIdx[frontPosIdx]) {
		// don't optimize too-old (established before frontFrmIdx) lmklines, 
		// but still use their recent observations/reprojections after frontPosIdx	
				if(lmklines[i].frmId_lnLid[0][0] < keyframeIdx[frontFrmIdx]) {
					idx2Rpj_notOpt.push_back(i);
				} else {
					paraVec.push_back(lmklines[i].A.x - ref[0]);
					paraVec.push_back(lmklines[i].A.y - ref[1]);
					paraVec.push_back(lmklines[i].A.z - ref[2]);
					paraVec.push_back(lmklines[i].B.x - ref[0]);
					paraVec.push_back(lmklines[i].B.y - ref[1]);
					paraVec.push_back(lmklines[i].B.z - ref[2]);
					idx2Opt.push_back(i);
				}
				break;
			}
		}		
	}
	double pn = 0, pmax = 0;
	int numPara = paraVec.size();
	double* para = new double[numPara];
	for (int i=0; i<numPara; ++i) {
		para[i] = paraVec[i];
		pn = pn + para[i]*para[i];
		if (pmax < abs(para[i]))
			pmax = abs(para[i]);
	}
	cout<<"para vector norm = "<< sqrt(pn) <<"\t max="<<pmax<<endl;
	// ----- optimization measurements -----
	vector<double> measVec;
	for(int i=0; i < idx2Opt.size(); ++i) {
		for(int j=0; j < lmklines[idx2Opt[i]].frmId_lnLid.size(); ++j) {
			if(frames[lmklines[idx2Opt[i]].frmId_lnLid[j][0]].isKeyFrame &&
			   lmklines[idx2Opt[i]].frmId_lnLid[j][0] >= keyframeIdx[frontFrmIdx]) {
#ifdef LINEDIST_6D
				for (int k=0; k<6; ++k) 
					measVec.push_back(0);
#else
				measVec.push_back(0);
				measVec.push_back(0);
#endif
			}
		}
	}
	for(int i=0; i < idx2Rpj_notOpt.size(); ++i) {
		for(int j=0; j < lmklines[idx2Rpj_notOpt[i]].frmId_lnLid.size(); ++j) {
			if(frames[lmklines[idx2Rpj_notOpt[i]].frmId_lnLid[j][0]].isKeyFrame &&
				lmklines[idx2Rpj_notOpt[i]].frmId_lnLid[j][0] >= keyframeIdx[frontFrmIdx]) {
#ifdef LINEDIST_6D
				for (int k=0; k<6; ++k) 
					measVec.push_back(0);
#else
				measVec.push_back(0);
				measVec.push_back(0);
#endif
			}
		}
	}
	int numMeas = std::max((int)measVec.size(), numPara);
	double* meas = new double[numMeas];
	for ( int i=0; i<numMeas; ++i) {
		if(i<measVec.size())
			meas[i] = measVec[i];
		else
			meas[i] = 0; // make measurment vector as long as paramter vector
	}

	// ----- pass additional data -----
	Data_LBA data(frames,lmklines);
	data.idx2Opt = idx2Opt;
	data.idx2Rpj_notOpt = idx2Rpj_notOpt;
	data.frontPosIdx = frontPosIdx;
	data.frontFrmIdx = frontFrmIdx;
	data.kfidx = keyframeIdx;
	data.ref = ref;
	data.ref_t = ref_t;
	
	// ----- start LM solver -----
	MyTimer timer;
	timer.start();
	cout<<"View "+num2str(frames.back().id)<<", paraDim="<<numPara<<", measDim="<<numMeas<<endl;
	int ret = dlevmar_dif(costFun_LBA, para, meas, numPara, numMeas,
						  maxIter, opts, info, NULL, NULL, (void*)&data);
	timer.end();
	cout<<" Time used: "<<timer.time_s<<" sec. ";
	termReason((int)info[6]);
	delete[] meas;	

	// ----- update camera and structure parameters -----
	int pidx = 0;
	for(int i = frontPosIdx; i < keyframeIdx.size(); ++i) {			
		frames[keyframeIdx[i]].R = q2r(&para[pidx]);
		if(frontPosIdx <=1 ) {
			frames[keyframeIdx[i]].t = (cv::Mat_<double>(3,1)<<para[pidx+4],para[pidx+5],para[pidx+6]);
		} else	{
			frames[keyframeIdx[i]].t = (cv::Mat_<double>(3,1)<<para[pidx+4] + ref_t[0],
					para[pidx+5] + ref_t[1], para[pidx+6] + ref_t[2]);//....................
		}
		pidx = pidx + 7;
	}
	// ---- structure parameters ----
	for(int i=0; i < idx2Opt.size(); ++i) {
		int idx = idx2Opt[i];
		lmklines[idx].A.x = para[pidx]  +ref[0];  
		lmklines[idx].A.y = para[pidx+1]+ref[1];
		lmklines[idx].A.z = para[pidx+2]+ref[2];
		lmklines[idx].B.x = para[pidx+3]+ref[0];  
		lmklines[idx].B.y = para[pidx+4]+ref[1];
		lmklines[idx].B.z = para[pidx+5]+ref[2];
		pidx = pidx + 6;
	}
}
#endif
