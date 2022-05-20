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
#include <fstream>
#include "../external/levmar-2.6/levmar.h"

using namespace Eigen;

extern cv::Mat K, distCoeffs;
extern SystemParameters sysPara;

#ifdef SLAM_LBA
void Map3d::correctPose(vector<PoseConstraint> pcs)
{
	// form a pose graph
	int numLocalConstraint = 10;
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
//	g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton(solver_ptr);
	optimizer.setAlgorithm(solver);

	// -- add the parameter representing the sensor offset  !!!!
	g2o::ParameterSE3Offset* sensorOffset = new g2o::ParameterSE3Offset;
	sensorOffset->setOffset(Eigen::Isometry3d::Identity());
	sensorOffset->setId(0);
	optimizer.addParameter(sensorOffset);
	
	// ----- set g2o vertices ------
	int vertex_id = 0;  
	tr1::unordered_map<int,int> camvid2fid, camfid2vid;
 
	// ----- optimization parameters -----
	vector<g2o::VertexSE3*> vertices;
	vector<g2o::EdgeSE3*> edges;
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
		if (i==0) 
			v_se3->setFixed(true);
		else 
			v_se3->setFixed(false);
		optimizer.addVertex(v_se3);
		camvid2fid[vertex_id] = keyframeIdx[i];
		camfid2vid[keyframeIdx[i]] = vertex_id;
		vertices.push_back(v_se3);
		++vertex_id;		
  	}
	// g2o edges
	// visual odometry edges
	for(int i=1; i<vertices.size(); ++i) {
		g2o::VertexSE3* cur  = vertices[i];
		for(int j=1; j <= numLocalConstraint; ++j) {
			if(i-j <0 ) continue;
			g2o::VertexSE3* prev = vertices[i-j];		
			Eigen::Isometry3d t = prev->estimate().inverse() * cur->estimate();
			g2o::EdgeSE3* e = new g2o::EdgeSE3;
			e->setVertex(0, prev);
			e->setVertex(1, cur);
			e->setMeasurement(t);
			// approximate information by number of common lanmarks
			int comLmk = 0;
			for(int k=0; k<frames[keyframeIdx[i]].lines.size(); ++k) {
				int gid = frames[keyframeIdx[i]].lines[k].gid;
				if (gid < 0) continue;
				for(int h = 0; h < lmklines[gid].frmId_lnLid.size(); ++h) {
					if (lmklines[gid].frmId_lnLid[h][0] == keyframeIdx[i-j]) {
						++comLmk;
						break;
					}
				}
			}
			e->information() = e->information() * comLmk;
	//		cout<<"keyframe "<<i<<" & "<<i-j<<": "<<comLmk <<endl; 
			optimizer.addEdge(e);
			edges.push_back(e);
		}
	}
	// loop closure edges
	for(int i=0; i<pcs.size(); ++i) {
		g2o::VertexSE3* from = vertices[pcs[i].from];
        g2o::VertexSE3* to   = vertices[pcs[i].to];
        Eigen::Isometry3d relpose;
		cv::Mat qi = r2q(pcs[i].R);
		Eigen:: Quaterniond q(qi.at<double>(0),qi.at<double>(1),qi.at<double>(2),qi.at<double>(3));
		relpose = q;
		relpose.translation() = Eigen::Vector3d(pcs[i].t.at<double>(0),
								pcs[i].t.at<double>(1),pcs[i].t.at<double>(2));		
        g2o::EdgeSE3* e = new g2o::EdgeSE3;
        e->setVertex(0, from);
        e->setVertex(1, to);
        e->setMeasurement(relpose);
		e->information() = e->information() * pcs[i].numMatches;
		optimizer.addEdge(e);
        edges.push_back(e);
	}
	// optimize
	// ----- start g2o -----
	MyTimer timer;
	timer.start();
	optimizer.initializeOptimization();
	optimizer.computeActiveErrors();
	cout<<"loopclosing cost "<<optimizer.activeChi2();
//	optimizer.setVerbose(1);
	optimizer.optimize(maxIters);
	timer.end();
	
	optimizer.computeActiveErrors();
	cout<<" ==> " <<optimizer.activeChi2()<<endl;
	cout<<", used time: "<<timer.time_s<<" sec. \n ";

	// ====== update poses =======
	for(int i = 0; i < keyframeIdx.size(); ++i) {	
		if(vertices[i]->fixed()) continue;
		Vector3d t = vertices[i]->estimate().translation();
		Quaterniond q(vertices[i]->estimate().rotation());
		double qd[] = {q.w(), q.x(), q.y(), q.z()};
		frames[keyframeIdx[i]].R = q2r(qd);
		frames[keyframeIdx[i]].t = (cv::Mat_<double>(3,1)<<t(0),t(1),t(2));
	}

	optimizer.clear();

	// correct landmarks
	lba_g2o(sysPara.num_pos_lba+5,sysPara.num_frm_lba+5, 1);
	write2file (*this, "_lpcls");
	//cin>>maxIters;
}
#endif
