#ifndef G2O_EDGE_SE3_LINE_ENDPTS_H_
#define G2O_EDGE_SE3_LINE_ENDPTS_H_

#include "g2o/core/base_binary_edge.h"

#include "g2o/types/slam3d/vertex_se3.h"
#include "vertex_lineendpts.h"
#include "g2o/types/slam3d/parameter_se3_offset.h"
//#include "g2o_types_slam3d_api.h"

namespace g2o {
  // first two args are the measurement type, second two the connection classes
  class EdgeSE3LineEndpts : public BaseBinaryEdge<6, VectorXd, VertexSE3, VertexLineEndpts> {
  public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSE3LineEndpts();
    virtual bool read(std::istream& is);
    virtual bool write(std::ostream& os) const;

    // return the error estimate as a 3-vector
    void computeError();
    // jacobian
 //   virtual void linearizeOplus();

    virtual void setMeasurement(const VectorXd& m){
      _measurement = m;
    }

    virtual bool setMeasurementData(const double* d){
      Map<const VectorXd> v(d,6);
      _measurement = v;
      return true;
    }

    virtual bool getMeasurementData(double* d) const{
      Map<VectorXd> v(d, 6);
      v=_measurement;
      return true;
    }
    
    virtual int measurementDimension() const {return 6;}

    virtual bool setMeasurementFromState() ;

    virtual double initialEstimatePossible(const OptimizableGraph::VertexSet& from, 
             OptimizableGraph::Vertex* to) { 
      (void) to; 
      return (from.count(_vertices[0]) == 1 ? 1.0 : -1.0);
    }

    virtual void initialEstimate(const OptimizableGraph::VertexSet& from, OptimizableGraph::Vertex* to);

	Eigen::Matrix<double,6,6> endptCov;
  private:
    Eigen::Matrix<double,6,6+6> J; // jacobian before projection
    ParameterSE3Offset* offsetParam;
    CacheSE3Offset* cache;
    virtual bool resolveCaches();
  };

#ifdef G2O_HAVE_OPENGL
  class EdgeSE3LineEndptsDrawAction: public DrawAction{
  public:
    EdgeSE3LineEndptsDrawAction();
    virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element,
            HyperGraphElementAction::Parameters* params_);
  };
#endif

}
#endif
