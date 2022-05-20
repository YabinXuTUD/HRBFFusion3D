

#include "g2o/types/slam3d/parameter_se3_offset.h"

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include <iostream>

#ifdef G2O_HAVE_OPENGL
#include "g2o/stuff/opengl_wrapper.h"
#endif

#include "Eigen/src/SVD/JacobiSVD.h"
#include "edge_se3_lineendpts.h"


Eigen::Vector3d normalized_pt2line_vec (const Eigen::VectorXd& line, const Eigen::Vector3d& pt, const Eigen::Matrix3d& pt_cov) {
// normalized by pt uncertainty
// return the vector from pt to the closest point on the line, vector direction is meaningless, but norm is of interest.
	Eigen::JacobiSVD<Eigen::Matrix3d> svd(pt_cov,Eigen::ComputeFullU);
	Eigen::Matrix3d D_invsqrt;
	D_invsqrt.fill(0);
	D_invsqrt(0,0) = sqrt(1/svd.singularValues()(0));
	D_invsqrt(1,1) = sqrt(1/svd.singularValues()(1));
	D_invsqrt(2,2) = sqrt(1/svd.singularValues()(2));
	Eigen::Vector3d A(line(0),line(1),line(2));
	Eigen::Vector3d B(line(3),line(4),line(5));
	Eigen::Vector3d Ap = D_invsqrt * svd.matrixU().transpose() * (A - pt);
	Eigen::Vector3d Bp = D_invsqrt * svd.matrixU().transpose() * (B - pt);
	double t = - Ap.dot(Bp-Ap)/((Bp-Ap).dot(Bp-Ap));
	return Ap+ t*(Bp-Ap);
}

Eigen::Vector3d closetPtonLine (const Eigen::VectorXd& line, const Eigen::Vector3d& pt) {
// return the closest point on the line
	Eigen::Vector3d A(line(0),line(1),line(2));
	Eigen::Vector3d B(line(3),line(4),line(5));
	double t = - (A - pt).dot(B-A)/((B-A).dot(B-A));
	return A + t*(B-A);
}

Eigen::Vector3d closetPtonLine (const Eigen::Vector3d& A, const Eigen::Vector3d& B, const Eigen::Vector3d& pt) {
// return the closest point on the line AB
	double t = - (A - pt).dot(B-A)/((B-A).dot(B-A));
	return A + t*(B-A);
}

namespace g2o {
	using namespace std;

	// point to camera projection, monocular
	EdgeSE3LineEndpts::EdgeSE3LineEndpts() : BaseBinaryEdge<6, VectorXd, VertexSE3, VertexLineEndpts>() {
		information().setIdentity();
		J.fill(0);
		J.block<3,3>(0,0) = -Eigen::Matrix3d::Identity();
		cache = 0;
		offsetParam = 0;
		resizeParameters(1);
		installParameter(offsetParam, 0, 0);  // third 0 ?????
	
	}

	bool EdgeSE3LineEndpts::resolveCaches(){
		ParameterVector pv(1);
		pv[0]=offsetParam;
		resolveCache(cache, (OptimizableGraph::Vertex*)_vertices[0],"CACHE_SE3_OFFSET",pv);
		return cache != 0;
	}


	bool EdgeSE3LineEndpts::read(std::istream& is) {
		int pId;
		is >> pId;
		setParameterId(0, pId);
		// measured endpoints
		VectorXd meas;
		for (int i=0; i<6; i++) is >> meas[i];
		setMeasurement(meas);
		// information matrix is the identity for features, could be changed to allow arbitrary covariances    
		if (is.bad()) {
			return false;
		}
		for ( int i=0; i<information().rows() && is.good(); i++)
			for (int j=i; j<information().cols() && is.good(); j++){
				is >> information()(i,j);
				if (i!=j)
					information()(j,i)=information()(i,j);
			}
			if (is.bad()) {
				//  we overwrite the information matrix
				information().setIdentity();
			} 
			return true;
	}

	bool EdgeSE3LineEndpts::write(std::ostream& os) const {
		os << offsetParam->id() << " ";
		for (int i=0; i<6; i++) os  << measurement()[i] << " ";
		for (int i=0; i<information().rows(); i++)
			for (int j=i; j<information().cols(); j++) {
				os <<  information()(i,j) << " ";
			}
			return os.good();
	}


	void EdgeSE3LineEndpts::computeError() {
		VertexLineEndpts *endpts = static_cast<VertexLineEndpts*>(_vertices[1]);

		Vector3d ptAw(endpts->estimate()[0],endpts->estimate()[1],endpts->estimate()[2]);
		Vector3d ptBw(endpts->estimate()[3],endpts->estimate()[4],endpts->estimate()[5]);

		Vector3d ptA = cache->n2w() * ptAw; // line endpoint tranformed to the camera frame
		Vector3d ptB = cache->n2w() * ptBw;
		Vector3d measpt1(_measurement(0),_measurement(1),_measurement(2));
		Vector3d measpt2(_measurement(3),_measurement(4),_measurement(5));
		// compute the mahalanobis distance		
		VectorXd ln(6);
		ln<<ptA(0),ptA(1),ptA(2),ptB(0),ptB(1),ptB(2);
		Vector3d normalized_pt2line_vec1 = normalized_pt2line_vec(ln, measpt1, endptCov.block<3,3>(0,0));
		Vector3d normalized_pt2line_vec2 = normalized_pt2line_vec(ln, measpt2, endptCov.block<3,3>(3,3));
		_error.resize(6);
		_error(0) = normalized_pt2line_vec1(0);
		_error(1) = normalized_pt2line_vec1(1);
		_error(2) = normalized_pt2line_vec1(2);
		_error(3) = normalized_pt2line_vec2(0);
		_error(4) = normalized_pt2line_vec2(1);
		_error(5) = normalized_pt2line_vec2(2);
	}

//	void EdgeSE3LineEndpts::linearizeOplus() {	}


	bool EdgeSE3LineEndpts::setMeasurementFromState(){ // what is the use of this function ???
		//VertexSE3 *cam = static_cast<VertexSE3*>(_vertices[0]);
		VertexLineEndpts *lpts = static_cast<VertexLineEndpts*>(_vertices[1]);

		// SE3OffsetCache* vcache = (SE3OffsetCache*) cam->getCache(_cacheIds[0]);
		// if (! vcache){
		//   cerr << "fatal error in retrieving cache" << endl;
		// }

		VertexLineEndpts *endpts = static_cast<VertexLineEndpts*>(_vertices[1]);
		Vector3d ptAw(endpts->estimate()[0],endpts->estimate()[1],endpts->estimate()[2]);
		Vector3d ptBw(endpts->estimate()[3],endpts->estimate()[4],endpts->estimate()[5]);
		Vector3d ptA = cache->w2n() * ptAw; // line endpoint tranformed to the camera frame
		Vector3d ptB = cache->w2n() * ptBw;
		_measurement.resize(6);
		_measurement(0) = ptA(0);
		_measurement(1) = ptA(1);
		_measurement(2) = ptA(2);
		_measurement(3) = ptB(0);
		_measurement(4) = ptB(1);
		_measurement(5) = ptB(2);
		return true;

	}


	void EdgeSE3LineEndpts::initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* /*to_*/) // ???
	{ // estimate 3d pt world position by cam pose and current meas pt
		(void) from;
		assert(from.size() == 1 && from.count(_vertices[0]) == 1 && "Can not initialize VertexDepthCam position by VertexTrackXYZ");

		VertexSE3 *cam = dynamic_cast<VertexSE3*>(_vertices[0]);
		VertexLineEndpts *point = dynamic_cast<VertexLineEndpts*>(_vertices[1]);
		// SE3OffsetCache* vcache = (SE3OffsetCache* ) cam->getCache(_cacheIds[0]);
		// if (! vcache){
		//   cerr << "fatal error in retrieving cache" << endl;
		// }
		// SE3OffsetParameters* params=vcache->params;
	//	Eigen::Vector3d p=_measurement;
	//	point->setEstimate(cam->estimate() * (offsetParam->offset() * p));
	}

#ifdef G2O_HAVE_OPENGL
	EdgeSE3LineEndptsDrawAction::EdgeSE3LineEndptsDrawAction(): DrawAction(typeid(EdgeSE3LineEndpts).name()){}

	HyperGraphElementAction* EdgeSE3LineEndptsDrawAction::operator()(HyperGraph::HyperGraphElement* element,
		HyperGraphElementAction::Parameters* params_){
			if (typeid(*element).name()!=_typeName)
				return 0;
			refreshPropertyPtrs(params_);
			if (! _previousParams)
				return this;

			if (_show && !_show->value())
				return this;

			EdgeSE3LineEndpts* e =  static_cast<EdgeSE3LineEndpts*>(element);
			VertexSE3* fromEdge = static_cast<VertexSE3*>(e->vertex(0));
			VertexLineEndpts* toEdge   = static_cast<VertexLineEndpts*>(e->vertex(1));
			glColor3f(0.8f,0.3f,0.3f);
			glPushAttrib(GL_ENABLE_BIT);
			glDisable(GL_LIGHTING);
			glBegin(GL_LINES);
			glVertex3f((float)fromEdge->estimate().translation().x(),(float)fromEdge->estimate().translation().y(),(float)fromEdge->estimate().translation().z());
			glVertex3f((float)toEdge->estimate()(0),(float)toEdge->estimate()(1),(float)toEdge->estimate()(2));
			glEnd();
			glPopAttrib();
			return this;
	}
#endif

} // end namespace
