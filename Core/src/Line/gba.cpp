#include "lineslam.h"
#include "utils.h"
#include <Eigen/StdVector>
#include <stdint.h>

#ifdef _MSC_VER
#include <unordered_set>
#else
#include <tr1/unordered_set>
#endif

#include "g2o/config.h"
#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/core/optimization_algorithm_gauss_newton.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"

#if defined G2O_HAVE_CHOLMOD
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#elif defined G2O_HAVE_CSPARSE
#include "g2o/solvers/csparse/linear_solver_csparse.h"
#endif

#include "g2o/types/slam3d/vertex_se3.h"
#include "edge_se3_lineendpts.h"
#include <fstream>
#include "../external/levmar-2.6/levmar.h"
#include "lineslam.h"
//#include <Windows.h>

using namespace Eigen;

extern cv::Mat K, distCoeffs;
extern SystemParameters sysPara;

#ifdef SLAM_LBA
void Map3d::correctAll (vector<vector<int> > lmkGidMatches)
	// use global BA to correct map+pose for loop closure
{
	// ======== landmark management ========
	for(int i=0; i<lmkGidMatches.size(); ++i) {		
		int gid0 = lmkGidMatches[i][0], gid1 = lmkGidMatches[i][1]; // gid1 should be larger than gid0
		if (gid0 >= gid1) continue;
		
		lmklines[gid0].frmId_lnLid.insert(lmklines[gid0].frmId_lnLid.end(), lmklines[gid1].frmId_lnLid.begin(),
											lmklines[gid1].frmId_lnLid.end());
		for(int j=0; j<lmklines[gid1].frmId_lnLid.size(); ++j) {
			int fid = lmklines[gid1].frmId_lnLid[j][0];
			int lid = lmklines[gid1].frmId_lnLid[j][1];
		
			frames[fid].lines[lid].gid = gid0;			
		}	
		// delete gid1
		lmklines[gid1].frmId_lnLid.clear();
		lmklines[gid1].gid = -1;
		
	}

	// ======== global bundle adjustment ========
	// ----- G2O parameter setting -----
	int maxIters = 15;
	// some handy typedefs
	typedef g2o::BlockSolver< g2o::BlockSolverTraits<Eigen::Dynamic, Eigen::Dynamic> >  MyBlockSolver;
	typedef g2o::LinearSolverCSparse<MyBlockSolver::PoseMatrixType> MyLinearSolver;
	
	// setup the solver
	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);
	MyLinearSolver* linearSolver = new MyLinearSolver();
	MyBlockSolver* solver_ptr = new MyBlockSolver(linearSolver);
	g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
	optimizer.setAlgorithm(solver);

	// -- add the parameter representing the sensor offset  !!!!
	g2o::ParameterSE3Offset* sensorOffset = new g2o::ParameterSE3Offset;
	sensorOffset->setOffset(Eigen::Isometry3d::Identity());
	sensorOffset->setId(0);
	optimizer.addParameter(sensorOffset);
	
	// ----- set g2o vertices ------
	int vertex_id = 0;  
	tr1::unordered_map<int,int> camvid2fid, camfid2vid, linvid2gid, lingid2vid;
  
	// ----- optimization parameters -----
	vector<g2o::VertexSE3*> camvertVec;
	// ---- camera pose parameters ----
	for(int i = 0; i < keyframeIdx.size(); ++i) {	
		cv::Mat qi = r2q(frames[keyframeIdx[i]].R);
		Eigen::Isometry3d pose;
		Eigen:: Quaterniond q(qi.at<double>(0),qi.at<double>(1),qi.at<double>(2),qi.at<double>(3));
		pose = q;
		pose.translation() = Eigen::Vector3d(frames[keyframeIdx[i]].t.at<double>(0),
			frames[keyframeIdx[i]].t.at<double>(1),frames[keyframeIdx[i]].t.at<double>(2));
		g2o::VertexSE3 * v_se3 = new g2o::VertexSE3();
		v_se3->setId(vertex_id);
		v_se3->setEstimate(pose);
		if (i == 0 ) 
			v_se3->setFixed(true);
		else 
			v_se3->setFixed(false);

		optimizer.addVertex(v_se3);
		camvid2fid[vertex_id] = keyframeIdx[i];
		camfid2vid[keyframeIdx[i]] = vertex_id;
		++vertex_id;
		camvertVec.push_back(v_se3);
  	}
	
	// ---- structure parameters ----
	vector<int> idx2Opt; // landmark idx to optimize,including fixed points 
	vector<g2o::VertexLineEndpts*> linevertVec;
	linevertVec.reserve(lmklines.size());
	for(int i=0; i < lmklines.size(); ++i) {
		if(lmklines[i].gid < 0 ) continue;
		g2o::VertexLineEndpts* vln = new g2o::VertexLineEndpts();
		Eigen::VectorXd tmpLine(6);
		tmpLine<<lmklines[i].A.x, lmklines[i].A.y, lmklines[i].A.z,
		lmklines[i].B.x, lmklines[i].B.y, lmklines[i].B.z;
		vln->setEstimate(tmpLine);
		vln->setId(vertex_id);
		vln->setFixed(false);
		optimizer.addVertex(vln);
		lingid2vid[lmklines[i].gid] = vertex_id;
		linvid2gid[vertex_id] = lmklines[i].gid;
		++vertex_id;
		idx2Opt.push_back(i);
		linevertVec.push_back(vln);			
	}
	// ----- set g2o edges: optimization measurements -----
	vector<g2o::EdgeSE3LineEndpts *> edgeVec;
	for(int i=0; i < idx2Opt.size(); ++i) {
		for(int j=0; j < lmklines[idx2Opt[i]].frmId_lnLid.size(); ++j) {
			int fid = lmklines[idx2Opt[i]].frmId_lnLid[j][0];
			int lid = lmklines[idx2Opt[i]].frmId_lnLid[j][1];
			if(frames[fid].isKeyFrame) {
					g2o::EdgeSE3LineEndpts * e = new g2o::EdgeSE3LineEndpts();
					assert(camfid2vid.find(fid)!=camfid2vid.end());
					e->vertices()[0] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
						(optimizer.vertices().find(camfid2vid[fid])->second);	
					if(e->vertices()[0]==0) {
						cerr<<"no cam vert found ... terminated \n";
						exit(0);
					}
					assert(lingid2vid.find(idx2Opt[i])!=lingid2vid.end());
					e->vertices()[1] = dynamic_cast<g2o::OptimizableGraph::Vertex*>
						(optimizer.vertices().find(lingid2vid[idx2Opt[i]])->second);
					if(e->vertices()[1]==0) {
						cerr<<"no line vert found ... terminated \n";
						exit(0);
					}
					Eigen::VectorXd tmpob(6);
					tmpob<<frames[fid].lines[lid].line3d.rndA.pos.x,frames[fid].lines[lid].line3d.rndA.pos.y,frames[fid].lines[lid].line3d.rndA.pos.z,
						   frames[fid].lines[lid].line3d.rndB.pos.x,frames[fid].lines[lid].line3d.rndB.pos.y,frames[fid].lines[lid].line3d.rndB.pos.z;
					e->setMeasurement(tmpob);
					e->information() = Eigen::MatrixXd::Identity(6,6);	// must be identity!
					cv::Mat covA = frames[fid].lines[lid].line3d.rndA.cov,
						covB = frames[fid].lines[lid].line3d.rndB.cov;
					e->endptCov = Eigen::MatrixXd::Identity(6,6);
					for(int ii=0; ii<3; ++ii) {
						for(int jj=0; jj<3; ++jj) {
							e->endptCov(ii,jj) = covA.at<double>(ii,jj);
						}
					}
					for(int ii=0; ii<3; ++ii) {
						for(int jj=0; jj<3; ++jj) {
							e->endptCov(ii+3,jj+3) = covB.at<double>(ii,jj);
						}
					}
					optimizer.addEdge(e);
					edgeVec.push_back(e);
			}
		}
	}
	optimizer.initializeOptimization();
	optimizer.computeActiveErrors();
	cout<<"gba cost " << optimizer.activeChi2();
 
	// ----- start g2o -----
	MyTimer timer;
	timer.start();
	optimizer.setVerbose(true);
	optimizer.optimize(maxIters);
//	optimizer.save("after.txt");
	timer.end();
	
	optimizer.computeActiveErrors();
	cout<<" ==> " <<optimizer.activeChi2();
	cout<<", used time: "<<timer.time_s<<" sec. \n ";
	// ----- update camera and structure parameters -----
	for(int i = 0; i < keyframeIdx.size(); ++i) {	
		if(camvertVec[i]->fixed()) continue;
		if(camvid2fid.find(camvertVec[i]->id())==camvid2fid.end())
			cout<<camvertVec[i]->id()<<" vert not found ...\n";
//		cout<<"frame "<<camvid2fid[camvertVec[idx]->id()]<<" pose updated \n";
		Vector3d t = camvertVec[i]->estimate().translation();
		Quaterniond q(camvertVec[i]->estimate().rotation());
		double qd[] = {q.w(), q.x(), q.y(), q.z()};
		frames[keyframeIdx[i]].R = q2r(qd);
		frames[keyframeIdx[i]].t = (cv::Mat_<double>(3,1)<<t(0),t(1),t(2));
	}
	// ---- structure parameters ----
	for(int i=0; i < idx2Opt.size(); ++i) {
		if(! linevertVec[i]->fixed()){		
			VectorXd ln = linevertVec[i]->estimate();
			lmklines[idx2Opt[i]].A.x = ln(0);  
			lmklines[idx2Opt[i]].A.y = ln(1);
			lmklines[idx2Opt[i]].A.z = ln(2);
			lmklines[idx2Opt[i]].B.x = ln(3);  
			lmklines[idx2Opt[i]].B.y = ln(4);
			lmklines[idx2Opt[i]].B.z = ln(5);
		}
	}
	optimizer.clear();
	write2file (*this, "_lpcls");
//	cin>>maxIters;
}
#endif
