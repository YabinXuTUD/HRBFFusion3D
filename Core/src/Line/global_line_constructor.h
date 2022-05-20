#include"lineslam.h"
#include"utils.h"
#include <Eigen/Core>
#include<pangolin/pangolin.h>

#include"../../Shaders/Shaders.h"
using namespace std;

namespace Line3D {

class GlobalLineConstructor{

public:
    GlobalLineConstructor();
    ~GlobalLineConstructor(){};

    void initilize(Frame* f);

    void fuse(Frame* f,  cv::Mat R, cv::Mat t);

    void drawGlobalLines(pangolin::OpenGlMatrix mvp);

public:

    std::vector<FrameLine> globalLines;

private:
    std::shared_ptr<Shader> drawGlobalLinesProgram;
    GLuint vbo;
};

}
