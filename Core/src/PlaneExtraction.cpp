#include "PlaneExtraction.h"

inline std::vector<Eigen::Vector3i> pseudocolor(int ncolors)
{
    srand((unsigned int)time(0));
    if(ncolors<=0) ncolors=10;
    std::vector<Eigen::Vector3i> ret(ncolors);
    for(int i=0; i<ncolors; ++i) {
        Eigen::Vector3i& color=ret[i];
        color[0]=rand()%255;
        color[1]=rand()%255;
        color[2]=rand()%255;
    }
    return ret;
}

PlaneExtraction::PlaneExtraction()
{


}

PlaneExtraction::~PlaneExtraction()
{
       //    delete cloud;
}

bool PlaneExtraction::initParameterLoad(std::string iniFileName)
{
    std::ifstream in(iniFileName);
    if(!in.is_open()){
        std::cout << " [iniLoad] " << iniFileName << " not found, use default parameters!" << std::endl;
        return false;
    }
    while(in){
        std::string line;
        std::getline(in, line);
        if(line.empty() || line[0] == '#') continue;
        std::string key, value;
        size_t eqPos = line.find_first_of("=");
        if(eqPos == std::string::npos || eqPos == 0)
        {
            //std::cout<<"[iniLoad] ignore line:" << line << std::endl;
            continue;
        }
        key = line.substr(0, eqPos);
        value = line.substr(eqPos + 1);
        //std::cout << "[iniLoad]" << key << "=>" << value << std::endl;
        initpara[key] = value;
    }
    return true;
}

void PlaneExtraction::SetParametersPF(std::string iniFileName){

//    if(initParameterLoad(iniFileName))
//    {
//        pf.minSupport = iniGet("minSupport", pf.minSupport);
//        pf.windowWidth = iniGet("windowWidth", pf.windowWidth);
//        pf.windowHeight = iniGet("windowHeight", pf.windowHeight);
//        pf.doRefine = iniGet("doRefine", pf.doRefine);

//        pf.params.initType = (ahc::InitType)iniGet("initType", (int)pf.params.initType);
//        //T_mse
//        pf.params.stdTol_merge = ("stdTol_merge", pf.params.stdTol_merge);
//        pf.params.stdTol_init = iniGet("stdTol_init", pf.params.stdTol_init);
//        pf.params.depthSigma =iniGet("depthSigma", pf.params.depthSigma);

//        pf.params.z_near = iniGet("z_near", pf.params.z_near);
//        pf.params.z_far = iniGet("z_far", pf.params.z_far);
//        pf.params.angle_near = MACRO_DEG2RAD(iniGet("angleDegree_near", MACRO_RAD2DEG(pf.params.angle_near)));
//        pf.params.angle_far =  MACRO_DEG2RAD(iniGet("angleDegree_far", MACRO_RAD2DEG(pf.params.angle_far)));
//        pf.params.similarityTh_merge = std::cos(MACRO_DEG2RAD(iniGet("similarityDegreeTh_merge", MACRO_RAD2DEG(pf.params.similarityTh_merge))));
//        pf.params.similarityTh_refine = std::cos(MACRO_DEG2RAD(iniGet("similarityDegreeTh_refine", MACRO_RAD2DEG(pf.params.similarityTh_refine))));
//    }

}

void PlaneExtraction::processOneframe(GPUTexture * source, PlaneExtraction::InputType inputType){

//    int scalePC = 1000;
//    cloud = new pcl::PointCloud<pcl::PointXYZ>;

//    cloud->width = Resolution::getInstance().cols();
//    cloud->height = Resolution::getInstance().rows();

//    cloud->reserve(cloud->width * cloud->height);

//    float fx = Intrinsics::getInstance().fx();
//    float fy = Intrinsics::getInstance().fy();
//    float cx = Intrinsics::getInstance().cx();
//    float cy = Intrinsics::getInstance().cy();

//    //download texture data to PCL
//    if(inputType == PlaneExtraction::VERTICES){
//        Img <Eigen::Vector4f> verticesImag(Resolution::getInstance().rows(),Resolution::getInstance().cols());
//        source->texture->Download(verticesImag.data , GL_RGBA, GL_FLOAT);
//        for(int i = 0; i < verticesImag.rows; i++){
//            for(int j = 0; j < verticesImag.cols; j++)
//            {
//                pcl::PointXYZ * p = new pcl::PointXYZ(verticesImag.at<Eigen::Vector4f>(i, j)(0) * scalePC,verticesImag.at<Eigen::Vector4f>(i, j)(1) * scalePC,verticesImag.at<Eigen::Vector4f>(i, j)(2) * scalePC);
//                //std::cout << p->_PointXYZ::x <<" "<< p->_PointXYZ::y <<" "<< p->_PointXYZ::z << std::endl;
//                if(p->_PointXYZ::z == 0)
//                {
//                    p->_PointXYZ::x = NAN;
//                    p->_PointXYZ::y = NAN;
//                    p->_PointXYZ::z = NAN;
//                }
//                cloud->points.push_back(*p);
//                delete p;
//            }
//        }
//    }else if(inputType == PlaneExtraction::DEPTH){
//        //depth covert to point cloud
//        Img <unsigned short> depthfImag(Resolution::getInstance().rows(), Resolution::getInstance().cols());
//        source->texture->Download(depthfImag.data , GL_LUMINANCE_INTEGER_EXT, GL_UNSIGNED_SHORT);
//        for(int i = 0; i < depthfImag.rows; i++){
//            for(int j = 0; j < depthfImag.cols; j++)
//            {
//               //since peac use depth in mm, we do not change it
//               float Z = depthfImag.at<unsigned short>(i, j);
//               pcl::PointXYZ p;
//               if(Z == 0)
//               {
//                  p._PointXYZ::x = NAN;
//                  p._PointXYZ::y = NAN;
//                  p._PointXYZ::z = NAN;
//               }else{
//                  p._PointXYZ::x = (j - cx) * Z / fx;
//                  p._PointXYZ::y = (i - cy) * Z / fy;
//                  p._PointXYZ::z = Z;
//               }
//               cloud->points.push_back(p);
//            }
//        }
//    }else
//    {
//        assert(false);
//    }


    //scale data, Note that in cloud operation the scale of point cloud is in mm
//    std::cout << cloud->points.size() << " " << cloud->width * cloud->height << std::endl;

//    cv::Mat seg(cloud->height, cloud->width, CV_8UC3);

//    ImageXYZ Ixyz(*cloud);
//    pf.run(&Ixyz, &p_member_ship, &seg);

    //save membership in vector
    //std::cout << p_member_ship.size() << std::endl;
//    for(int i = 0; i < p_member_ship.size(); i++)
//    {
//        Eigen::Vector3f pl_c(pf.extractedPlanes[i]->center[0], pf.extractedPlanes[i]->center[1], pf.extractedPlanes[i]->center[2]);
//        Eigen::Vector3f pl_n(pf.extractedPlanes[i]->normal[0], pf.extractedPlanes[i]->normal[1], pf.extractedPlanes[i]->normal[2]);

//        std::cout <<"current plane contains "<< p_member_ship[i].size()<< " points" << std::endl;
//        for(int j = 0; j < p_member_ship[i].size(); j++)
//        {
//            int pid = p_member_ship[i][j];
//            //std::cout <<"Point ID: "<< pid <<" ";
//            Eigen::Vector3f p(verticesImag.at<Eigen::Vector4f>(pid)(0) * 1000, verticesImag.at<Eigen::Vector4f>(pid)(1) * 1000, verticesImag.at<Eigen::Vector4f>(pid)(2) * 1000);
//            float dist_inlier = (p - pl_c).dot(pl_n);
//            if(dist_inlier > 50)
//            std::cout <<"check inlier distance: "<< dist_inlier <<" ";
//        }

//        std::cout << std::endl;
//    }

//    for(int i = 0; i < pf.extractedPlanes.size(); i++)
//    {
//        std::cout << "The plane ceter is " <<pf.extractedPlanes[i]->center[0] <<" "<< pf.extractedPlanes[i]->center[1] <<" "<< pf.extractedPlanes[i]->center[2] << std::endl;
//        std::cout << "The normal is " <<pf.extractedPlanes[i]->normal[0] <<" "<< pf.extractedPlanes[i]->normal[1] <<" "<< pf.extractedPlanes[i]->normal[2] << std::endl;
//    }

    //save seg cloud
//    CloudXYZRGB xyzrgb(cloud->width, cloud->height);

//    for(int r=0; r<(int)xyzrgb.height; ++r) {
//        for(int c=0; c<(int)xyzrgb.width; ++c) {
//            pcl::PointXYZRGB& pix = xyzrgb.at(c, r);
//            const pcl::PointXYZ& pxyz = cloud->at(c, r);
//            const cv::Vec3b& prgb = seg.at<cv::Vec3b>(r,c);;
//            pix.x=pxyz.x;
//            pix.y=pxyz.y;
//            pix.z=pxyz.z;
//            pix.r=prgb(2);
//            pix.g=prgb(1);
//            pix.b=prgb(0);
//        }
//    }

//    pcl::io::savePLYFile("frameseg.ply", xyzrgb);
    std::cout <<"segment done"<<std::endl;
}

void PlaneExtraction::processOneframe(float* vertices, int rows, int cols, int i)
{
//     cloud = new pcl::PointCloud<pcl::PointXYZ>;
//     cloud->width = cols;
//     cloud->height = rows;

//     cloud->reserve(cloud->width * cloud->height);

//     //unit in ahc is mm, unit in vertices is m, scale with 1000
//     int scale = 1000;
//     int stride = rows * cols;
//     for(int i = 0; i < stride; i++){
//         pcl::PointXYZ * p = new pcl::PointXYZ(scale * vertices[i],scale * vertices[ i + stride],scale * vertices[i + 2 * stride]);
//         if(isnan(p->_PointXYZ::x) || isnan(p->_PointXYZ::y) || isnan(p->_PointXYZ::z))
//         {
//             p->_PointXYZ::x = NAN;
//             p->_PointXYZ::y = NAN;
//             p->_PointXYZ::z = NAN;
//         }
//         cloud->points.push_back(*p);
//         delete p;
//     }
//     //scale data, Note that in cloud operation the scale of point cloud is in mm
//     std::cout << cloud->points.size() << " " << cloud->width * cloud->height << std::endl;
//     cv::Mat seg(cloud->height, cloud->width, CV_8UC3);
//     ImageXYZ Ixyz(*cloud);
//     pf.run(&Ixyz, &p_member_ship, &seg);
//     //save seg cloud
////     CloudXYZRGB xyzrgb(cloud->width, cloud->height);

////     for(int r=0; r<(int)xyzrgb.height; ++r) {
////         for(int c=0; c<(int)xyzrgb.width; ++c) {
////             pcl::PointXYZRGB& pix = xyzrgb.at(c, r);
////             const pcl::PointXYZ& pxyz = cloud->at(c, r);
////             const cv::Vec3b& prgb = seg.at<cv::Vec3b>(r,c);;
////             pix.x=pxyz.x;
////             pix.y=pxyz.y;
////             pix.z=pxyz.z;
////             pix.r=prgb(2);
////             pix.g=prgb(1);
////             pix.b=prgb(0);
////         }
////     }

////     char filename[36];
////     snprintf(filename, sizeof(filename), "segresults_%d.ply", i);
////     pcl::io::savePLYFile(filename, xyzrgb);
}

void PlaneExtraction::processOneframeRansac(float* vertices, float* normals,int rows, int cols, std::string type,
                                            const Eigen::Matrix4f& currpose)
{
    size_t stride = static_cast<size_t>(rows * cols);
    pc.reserve(stride);

    //fill in pointcloud set
    for(size_t i = 0; i < stride; i++){
          Point p1;
          if(std::isnan(vertices[i]) || std::isnan(vertices[i + stride])|| std::isnan(vertices[i + 2 * stride]) || vertices[i + 2 * stride] == 0.0f ||
                  std::isnan(normals[i]) || std::isnan(normals[i + stride])|| std::isnan(normals[i +2 * stride]))
          {
              continue;
          }
          Eigen::Vector4f vPoint(vertices[i], vertices[i + stride], vertices[i + 2 * stride], 1.0);
          Eigen::Vector4f vPoint_g = currpose * vPoint;
          p1.pos[0] = vPoint_g(0);
          p1.pos[1] = vPoint_g(1);
          p1.pos[2] = vPoint_g(2);

          Eigen::Vector3f nPoint(normals[i], normals[i + stride], normals[i + 2 * stride]);
          Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rot = currpose.topLeftCorner(3, 3);
          Eigen::Vector3f nPoint_g = Rot * nPoint;
          p1.normal[0] = nPoint_g(0);
          p1.normal[1] = nPoint_g(1);
          p1.normal[2] = nPoint_g(2);
          p1.index = i;
          pc.push_back(p1);
    }

    std::cout << "There suppose to have " << stride <<" points in the frame But with " << pc.size() <<" points valid" << std::endl;

    Vec3f min, max;
    min[0] = 1e8;
    min[1] = 1e8;
    min[2] = 1e8;
    max[0] = -1e8;
    max[1] = -1e8;
    max[2] = -1e8;

    for(size_t i = 0; i < pc.size(); i++)
    {
       max[0] = std::max(max[0],pc.at(i).pos[0]);
       max[1] = std::max(max[1],pc.at(i).pos[1]);
       max[2] = std::max(max[2],pc.at(i).pos[2]);

       min[0] = std::min(min[0],pc.at(i).pos[0]);
       min[1] = std::min(min[1],pc.at(i).pos[1]);
       min[2] = std::min(min[2],pc.at(i).pos[2]);
    }

    float bounding_box_with = -1e8;
    for(int i = 0; i < 3; i++)
    {
        bounding_box_with = std::max(bounding_box_with, max[i] - min[i]);
    }

    pc.setBBox(min, max);

    std::cout << "boundingbox min: " <<min[0] <<" " <<min[1]<<" "<< min[2] << std::endl;
    std::cout << "max: " <<max[0] <<" " <<max[1]<<" "<< max[2] << std::endl;

    RansacShapeDetector::Options ransacOptions;
    ransacOptions.m_epsilon = 0.005; // set distance threshold to .01f of bounding box width
        // NOTE: Internally the distance threshold is taken as 3 * ransacOptions.m_epsilon!!!
    std::cout << "m_epsilon: " << 3 * ransacOptions.m_epsilon * bounding_box_with << std::endl;
    ransacOptions.m_bitmapEpsilon = 0.02; // set bitmap resolution to .02f of bounding box width
        // NOTE: This threshold is NOT multiplied internally!
    ransacOptions.m_normalThresh = 0.8f; // this is the cos of the maximal normal deviation
    ransacOptions.m_minSupport = 1000; // this is the minimal numer of points required for a primitive
    ransacOptions.m_probability = 0.001f; // this is the "probability" with which a primitive is overlooked

    std::cout<<"option set done"<< std::endl;
    RansacShapeDetector detector(ransacOptions); // the detector object
    // set which primitives are to be detected by adding the respective constructors
    detector.Add(new PlanePrimitiveShapeConstructor());

    MiscLib::Vector< std::pair< MiscLib::RefCountPtr< PrimitiveShape >, size_t > > shapes;
    size_t remaining = detector.Detect(pc, 0, pc.size(), &shapes); // run detection

    std::cout<<"detect "<<shapes.size() << " shapes"<< std::endl;
    std::cout<<"detection done still remain: "<<remaining<<" points"<< std::endl;

    //assign each shape a struct to indicate the corresponding start point and interval
    int accum = 0;
    for(int i = 0; i < shapes.size(); i++)
    {
        accum = accum + shapes[i].second;
        shape2pointAssoc s2p;
        s2p.start = pc.size() - accum;
        s2p.interval = shapes[i].second;
        shape2pointAssocSet.push_back(s2p);
    }

    //output detected points
    std::ofstream primitive_points;
    primitive_points.open("primitivepoints_" + type + ".txt");
    //output primitives
    std::vector<Eigen::Vector3i> colorSet = pseudocolor(shapes.size());
    size_t interval = 0;
    planeParaSet.reserve(shapes.size());
    //get plane parameter from <Serialize> float array structure: normal(3 floats) + mean_dist(1 float) + position(3 floats)
    for(int i = 0; i < static_cast<int>(shapes.size()); i++)
    {
        interval = interval + shapes[i].second;
        float* para = new float[shapes[i].first->SerializedFloatSize()];
        shapes[i].first->Serialize(para);
        //std::cout << shapes[i].first->SerializedFloatSize() << std::endl;
        planePara pl;
        pl.normal[0] = para[0];
        pl.normal[1] = para[1];
        pl.normal[2] = para[2];
        pl.mean_dist = para[3];
        pl.pos[0] = para[4];
        pl.pos[1] = para[5];
        pl.pos[2] = para[6];
        //std::cout << std::endl;
        planeParaSet.push_back(pl);
        for(size_t j = 0; j < shapes[i].second; j++)
        {
            size_t start = pc.size() - interval;
            size_t pointID = pc.at(start + j).index;
//            primitive_points << vertices[pointID] << " " << vertices[pointID + stride] << " " << vertices[pointID + 2 * stride] <<" "
//                             << colorSet[i](0)<<" "<< colorSet[i](1)<<" "<<colorSet[i](2)<<"\n";
        }
        delete [] para;
    }
    primitive_points.close();
};



