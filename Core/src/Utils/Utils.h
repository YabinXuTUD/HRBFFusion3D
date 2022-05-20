#include <Eigen/Core>
#include <vector>

using namespace Eigen;
using namespace std;

typedef std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f>> PoseVec;

void saveTrajectoryAsPointCloud(PoseVec& poses);
