
#ifndef G2O_VERTEX_LINE6D_H_
#define G2O_VERTEX_LINE6D_H_

#include "g2o/types/slam3d/g2o_types_slam3d_api.h"
#include "g2o/core/base_vertex.h"
#include "g2o/core/hyper_graph_action.h"

using namespace std;
using namespace Eigen;

namespace g2o {
  /**
   * \brief Vertex for a tracked point in space
   */
	class VertexLineEndpts : public BaseVertex<6, Eigen::VectorXd>
  {
    public:
 //     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
      VertexLineEndpts() {}
      virtual bool read(std::istream& is);
      virtual bool write(std::ostream& os) const;

      virtual void setToOriginImpl() { _estimate.fill(0.); }

      virtual void oplusImpl(const double* update_) {
        Map<const VectorXd> update(update_, 6);
        _estimate += update;
      }

      virtual bool setEstimateDataImpl(const double* est){
        Map<const VectorXd> _est(est, 6);
        _estimate = _est;
        return true;
      }

      virtual bool getEstimateData(double* est) const{
        Map<VectorXd> _est(est, 6);
        _est = _estimate;
        return true;
      }

      virtual int estimateDimension() const {
        return 6;
      }

      virtual bool setMinimalEstimateDataImpl(const double* est){
        _estimate = Map<const VectorXd>(est, 6);
        return true;
      }

      virtual bool getMinimalEstimateData(double* est) const{
        Map<VectorXd> v(est, 6);
        v = _estimate;
        return true;
      }

      virtual int minimalEstimateDimension() const {
        return 6;
      }

  };

  class VertexLineEndptsWriteGnuplotAction: public WriteGnuplotAction
  {
    public:
      VertexLineEndptsWriteGnuplotAction();
      virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element, HyperGraphElementAction::Parameters* params_ );
  };

#ifdef G2O_HAVE_OPENGL
  /**
   * \brief visualize a 3D point
   */
  class VertexLineEndptsDrawAction: public DrawAction{
    public:
      VertexLineEndptsDrawAction();
      virtual HyperGraphElementAction* operator()(HyperGraph::HyperGraphElement* element, 
          HyperGraphElementAction::Parameters* params_);


    protected:
      FloatProperty *_pointSize;
      virtual bool refreshPropertyPtrs(HyperGraphElementAction::Parameters* params_);
  };
#endif

}
#endif
