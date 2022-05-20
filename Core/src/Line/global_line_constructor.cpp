#include"global_line_constructor.h"

using namespace std;
namespace Line3D {

GlobalLineConstructor::GlobalLineConstructor():drawGlobalLinesProgram(loadProgramFromFile("draw_lines.vert", "draw_lines.frag"))
{
    glGenBuffers(1, &vbo);
}

void GlobalLineConstructor::initilize(Frame* f)
{
    globalLines.reserve(200);
    for(int i = 0; i < f->lines.size(); i++)
    {
        if(f->lines[i].haveDepth)
            globalLines.push_back(f->lines[i]);
    }
    std::cout << "initial global 3D lines: " << globalLines.size() << "\n";
}

void GlobalLineConstructor::fuse(Frame* f, cv::Mat R, cv::Mat t)
{
    //update current frame according to estimated poses, should be in CV mat format
    f->update3DLinePose(R, t);
    std::vector<std::vector<int>> matches;
    std::vector<int> addNew;

    //just add all 3D lines;
//    for(int i = 0; i < f->lines.size(); i++)
//    {
//        if(f->lines[i].haveDepth)
//            globalLines.push_back(f->lines[i]);
//    }
    //Line matching
    Line3D::trackLine3D(f->lines, globalLines, matches, addNew);
    std::cout << "find number of matched lines: " << matches.size() << std::endl;
//if matched, fuse them
    for(int i = 0; i < matches.size(); i++)
    {
        //update all line attributes in the GlobalLines, not only the 3D part
        if(f->lines[matches[i][0]].haveDepth)
        {
            //update 2D descriptor(use the new line descriptor to update the global line descriptor)
            f->lines[matches[i][0]].des.copyTo(globalLines[matches[i][1]].des);
//            std::cout << "f1 inlier size: " << f->lines[matches[i][0]].line3d.pts.size() << std::endl;
//            std::cout << "global lines inlier size: " << globalLines[matches[i][1]].line3d.pts.size() << std::endl;
            //update Random 3D lines
            RandomLine3d tmpLine;
            vector<RandomPoint3d> rndpts3d;
            rndpts3d.reserve(1000);
            rndpts3d.insert(rndpts3d.end(), f->lines[matches[i][0]].line3d.pts.begin(), f->lines[matches[i][0]].line3d.pts.end());
            rndpts3d.insert(rndpts3d.end(), globalLines[matches[i][1]].line3d.pts.begin(), globalLines[matches[i][1]].line3d.pts.end());
            tmpLine = extract3dline_mahdist(rndpts3d);
            globalLines[matches[i][1]].line3d = tmpLine;
        }
    }

    std::cout << "find unmatched lines: " << addNew.size() << " ";
    //if not matched, add as new
    for(int i = 0; i < addNew.size(); i++)
    {
        if(f->lines[addNew[i]].haveDepth)
            globalLines.push_back(f->lines[addNew[i]]);
    }
}

void GlobalLineConstructor::drawGlobalLines(pangolin::OpenGlMatrix mvp)
{
    //render global lines here;
    std::vector<float> lb;
    int N_lines = globalLines.size();
    lb.reserve(6 * N_lines); //2 point (3 floats) for a line;
    for(int i = 0; i < N_lines; i++){
        if(globalLines[i].haveDepth)
        {
            lb.push_back(static_cast<float>(globalLines[i].line3d.A.x));
            lb.push_back(static_cast<float>(globalLines[i].line3d.A.y));
            lb.push_back(static_cast<float>(globalLines[i].line3d.A.z));

            lb.push_back(static_cast<float>(globalLines[i].line3d.B.x));
            lb.push_back(static_cast<float>(globalLines[i].line3d.B.y));
            lb.push_back(static_cast<float>(globalLines[i].line3d.B.z));
        }
    }
    drawGlobalLinesProgram->Bind();
    drawGlobalLinesProgram->setUniform(Uniform("MVP", mvp));
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    drawGlobalLinesProgram->setUniform(Uniform("pose", pose));

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, N_lines * 6 * sizeof(float), &lb[0], GL_STREAM_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);

    glDrawArrays(GL_LINES, 0, N_lines);
    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    drawGlobalLinesProgram->Unbind();
}
}
