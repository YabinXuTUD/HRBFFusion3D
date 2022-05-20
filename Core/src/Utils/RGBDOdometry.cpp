/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 * 
 * The use of the code within this file and all code within files that 
 * make up the software that is ElasticFusion is permitted for 
 * non-commercial purposes only.  The full terms and conditions that 
 * apply to the code within this file are detailed within the LICENSE.txt 
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/> 
 * unless explicitly stated.  By downloading this file you agree to 
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then 
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */

#include "RGBDOdometry.h"

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

RGBDOdometry::RGBDOdometry(int width,
                           int height,
                           float cx, float cy, float fx, float fy,
                           float distThresh,
                           float angleThresh
                           )
:
  lastICPError(0),
  lastICPCount(width * height),
  lastRGBError(0),
  lastRGBCount(width * height),
  lastSO3Error(0),
  lastSO3Count(width * height),
  lastA(Eigen::Matrix<double, 6, 6, Eigen::RowMajor>::Zero()),
  lastb(Eigen::Matrix<double, 6, 1>::Zero()),
  sobelSize(3),
  sobelScale(1.0 / pow(2.0, sobelSize)),
  maxDepthDeltaRGB(0.07),
  maxDepthRGB(6.0),
  distThres_(distThresh),
  angleThres_(angleThresh),
  width(width),
  height(height),
  cx(cx), cy(cy), fx(fx), fy(fy)
{
    sumDataSE3.create(MAX_THREADS);
    outDataSE3.create(1);
    sumResidualRGB.create(MAX_THREADS);

    sumDataSO3.create(MAX_THREADS);
    outDataSO3.create(1);

    for(int i = 0; i < NUM_PYRS; i++)
    {
        int2 nextDim = {height >> i, width >> i};
        pyrDims.push_back(nextDim);
    }

    for(int i = 0; i < NUM_PYRS; i++)
    {
        lastDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        lastImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        nextDepth[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        lastNextImage[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        nextdIdx[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        nextdIdy[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        pointClouds[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        corresImg[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        corresICP[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        cuda_out_[i].create(pyrDims.at(i).x, pyrDims.at(i).y);

        z_thrinkMap[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
        lambdaMap[i].create(pyrDims.at(i).x, pyrDims.at(i).y);
    }

    intr.cx = cx;
    intr.cy = cy;
    intr.fx = fx;
    intr.fy = fy;

    iterations.resize(NUM_PYRS);

    depth_tmp.resize(NUM_PYRS);

    vmaps_g_prev_.resize(NUM_PYRS);
    nmaps_g_prev_.resize(NUM_PYRS);
    ck1maps_g_prev_.resize(NUM_PYRS);
    ck2maps_g_prev_.resize(NUM_PYRS);

    vmaps_curr_.resize(NUM_PYRS);
    nmaps_curr_.resize(NUM_PYRS);
    ck1maps_curr_.resize(NUM_PYRS);
    ck2maps_curr_.resize(NUM_PYRS);

    icpWeightMap_.resize(NUM_PYRS);

    plane_match_map_g_.resize(NUM_PYRS);
    plane_match_map_curr_.resize(NUM_PYRS);

    for (int i = 0; i < NUM_PYRS; ++i)
    {
        int pyr_rows = height >> i;
        int pyr_cols = width >> i;

        depth_tmp[i].create (pyr_rows, pyr_cols);

        vmaps_g_prev_[i].create (pyr_rows * 4, pyr_cols);
        nmaps_g_prev_[i].create (pyr_rows * 4, pyr_cols);
        ck1maps_g_prev_[i].create (pyr_rows * 4, pyr_cols);
        ck2maps_g_prev_[i].create (pyr_rows * 4, pyr_cols);

        vmaps_curr_[i].create (pyr_rows * 4, pyr_cols);
        nmaps_curr_[i].create (pyr_rows * 4, pyr_cols);
        ck1maps_curr_[i].create (pyr_rows * 4, pyr_cols);
        ck2maps_curr_[i].create (pyr_rows * 4, pyr_cols);

        icpWeightMap_[i].create(pyr_rows, pyr_cols);

        plane_match_map_g_[i].create(pyr_rows, pyr_cols);
        plane_match_map_curr_[i].create(pyr_rows, pyr_cols);
    }

    vmaps_tmp.create(height * 4 * width);
    nmaps_tmp.create(height * 4 * width);
    ck1maps_tmp.create(height * 4 * width);
    ck2maps_tmp.create(height * 4 * width);
    icpweight_tmp.create(height * width);

    minimumGradientMagnitudes.resize(NUM_PYRS);
    minimumGradientMagnitudes[0] = 5;
    minimumGradientMagnitudes[1] = 3;
    minimumGradientMagnitudes[2] = 1;
}

RGBDOdometry::~RGBDOdometry()
{

}

void RGBDOdometry::initICP(GPUTexture * filteredDepth, const float depthCutoff, const float mDepthMapFactor)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &filteredDepth->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, filteredDepth->cudaRes, 0, 0); //get an array through which to access a subresource of a mapped graphics resource.
    cudaMemcpy2DFromArray(depth_tmp[0].ptr(0), depth_tmp[0].step(), textPtr, 0, 0, depth_tmp[0].colsBytes(), depth_tmp[0].rows(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &filteredDepth->cudaRes);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        pyrDown(depth_tmp[i - 1], depth_tmp[i]);
    }

    for(int i = 0; i < NUM_PYRS; ++i)
    {
        createVMap(intr(i), depth_tmp[i], vmaps_curr_[i], depthCutoff, mDepthMapFactor);
        createNMap(vmaps_curr_[i], nmaps_curr_[i]);
    }
    cudaDeviceSynchronize();
}

void RGBDOdometry::initICP(GPUTexture * predictedVertices, GPUTexture * predictedNormals, const float depthCutoff)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
    cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);

    cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedNormals->cudaRes, 0, 0);
    cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);

    copyMaps(vmaps_tmp, nmaps_tmp, vmaps_curr_[0], nmaps_curr_[0]);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        resizeVMap(vmaps_curr_[i - 1], vmaps_curr_[i]);
        resizeNMap(nmaps_curr_[i - 1], nmaps_curr_[i]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initICPModel(GPUTexture * predictedVertices,
                                GPUTexture * predictedNormals,
                                const float depthCutoff,
                                const Eigen::Matrix4f & modelPose)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &predictedVertices->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedVertices->cudaRes, 0, 0);
    cudaMemcpyFromArray(vmaps_tmp.ptr(), textPtr, 0, 0, vmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedVertices->cudaRes);

    cudaGraphicsMapResources(1, &predictedNormals->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, predictedNormals->cudaRes, 0, 0);
    cudaMemcpyFromArray(nmaps_tmp.ptr(), textPtr, 0, 0, nmaps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &predictedNormals->cudaRes);

    copyMaps(vmaps_tmp, nmaps_tmp, vmaps_g_prev_[0], nmaps_g_prev_[0]);

    for(int i = 1; i < NUM_PYRS; ++i)
    {
        resizeVMap(vmaps_g_prev_[i - 1], vmaps_g_prev_[i]);
        resizeNMap(nmaps_g_prev_[i - 1], nmaps_g_prev_[i]);
    }

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = modelPose.topLeftCorner(3, 3);
    Eigen::Vector3f tcam = modelPose.topRightCorner(3, 1);

    //send to device arrray
    mat33 device_Rcam = Rcam;
    float3 device_tcam = *reinterpret_cast<float3*>(tcam.data());

    //transform map to global model frame (reference)
    for(int i = 0; i < NUM_PYRS; ++i)
    {
        tranformMaps(vmaps_g_prev_[i], nmaps_g_prev_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::correspondPlaneSearch(GPUTexture* depthfiltered, GPUTexture* vertices){

//    PlaneExtraction* pe_f = new PlaneExtraction;
//    pe_f->SetParametersPF("../src/plane_fitter_pcd.ini");
//    pe_f->processOneframe(depthfiltered, PlaneExtraction::DEPTH);

//    PlaneExtraction* pe_g = new PlaneExtraction;
//    pe_g->SetParametersPF("../src/plane_fitter_pcd.ini");
//    pe_g->processOneframe(vertices, PlaneExtraction::VERTICES);

//    //find plane match
//    std::vector<std::pair<int , int>> plane_matches;

//    for(int i = 0; i < pe_f->getPointFitter()->extractedPlanes.size();i++)
//    {
//        double pl_f_norm[3], pl_f_center[3];

//        pl_f_norm[0] = pe_f->getPointFitter()->extractedPlanes[i]->normal[0];
//        pl_f_norm[1] = pe_f->getPointFitter()->extractedPlanes[i]->normal[1];
//        pl_f_norm[2] = pe_f->getPointFitter()->extractedPlanes[i]->normal[2];

//        pl_f_center[0] = pe_f->getPointFitter()->extractedPlanes[i]->center[0];
//        pl_f_center[1] = pe_f->getPointFitter()->extractedPlanes[i]->center[1];
//        pl_f_center[2] = pe_f->getPointFitter()->extractedPlanes[i]->center[2];

//        for(int j = 0; j < pe_g->getPointFitter()->extractedPlanes.size();j++)
//        {
//            double pl_g_norm[3], pl_g_center[3];

//            pl_g_norm[0] = pe_g->getPointFitter()->extractedPlanes[j]->normal[0];
//            pl_g_norm[1] = pe_g->getPointFitter()->extractedPlanes[j]->normal[1];
//            pl_g_norm[2] = pe_g->getPointFitter()->extractedPlanes[j]->normal[2];

//            pl_g_center[0] = pe_g->getPointFitter()->extractedPlanes[j]->center[0];
//            pl_g_center[1] = pe_g->getPointFitter()->extractedPlanes[j]->center[1];
//            pl_g_center[2] = pe_g->getPointFitter()->extractedPlanes[j]->center[2];

//            //normal deviation < 10 deg + center distance + center to plane distance < 50

//            double normal_dev = pl_f_norm[0] * pl_g_norm[0] + pl_f_norm[1] * pl_g_norm[1] + pl_f_norm[2] * pl_g_norm[2];
//            if(fabs(normal_dev) > 0.95)
//            {
//                double center_dist = sqrt((pl_f_center[0] - pl_g_center[0])* (pl_f_center[0] - pl_g_center[0]) +
//                     (pl_f_center[0] - pl_g_center[0])* (pl_f_center[0] - pl_g_center[0]) +
//                        (pl_f_center[0] - pl_g_center[0])* (pl_f_center[0] - pl_g_center[0]));
//                std::cout<< "center difference is: "<< center_dist <<" ";
//                if(center_dist < 200)
//                {
//                    //set as matched plane
//                    std::pair<int, int> match;
//                    match.first = j;
//                    match.second = i;
//                    plane_matches.push_back(match);

//                }
//            }
//        }
//    }

//    std::cout << "We find plane matches: " << std::endl;

//    //test,save corresponding points
//    std::ofstream pc1;
//    pc1.open("corres1.txt");

//    std::ofstream pc2;
//    pc2.open("corres2.txt");
//    for(int i = 0; i< plane_matches.size(); i++)
//    {

//        int pl1_id = plane_matches[i].first;
//        for(int j = 0 ; j < pe_f->p_member_ship[pl1_id].size() ; j++){
//            int pt_id = pe_f->p_member_ship[pl1_id][j];
//            pc1 << pe_f->cloud->points.at(pt_id)._PointXYZ::x <<" "<< pe_f->cloud->points.at(pt_id)._PointXYZ::y<<" "<< pe_f->cloud->points.at(pt_id)._PointXYZ::z <<"\n";
//        }


//        int pl2_id = plane_matches[i].second;
//        for(int j = 0 ; j < pe_g->p_member_ship[pl2_id].size() ; j++){
//            int pt_id = pe_g->p_member_ship[pl2_id][j];
//            pc2 << pe_g->cloud->points.at(pt_id)._PointXYZ::x <<" "<< pe_g->cloud->points.at(pt_id)._PointXYZ::y<<" "<< pe_g->cloud->points.at(pt_id)._PointXYZ::z <<"\n";
//        }
//    }

//    pc1.close();
//    pc2.close();

//    delete pe_f;
//    delete pe_g;
}
void RGBDOdometry::correspondPlaneSearch(){
//    //Test
//    //for each sub map, download hierachy data structure
//    //std::cout<<"data dimension is: " << NUM_PYRS <<std::endl;
//    for(int i = NUM_PYRS - 1; i >= 0 ; i--)
//    {
//        //note that rows in vmaps_curr_ is 3 * actual rows and the structure is more like first a stride of X and then Y and then Z
//        int rows = vmaps_curr_[i].rows();
//        int cols = vmaps_curr_[i].cols();
//        //use different parameters files for different level of details
//        char para_file[36];
//        snprintf(para_file, sizeof(para_file), "../src/plane_fitter_pcd_%d.ini", i);

//        size_t step = sizeof(float);
//        const size_t buffersize = step * rows * cols;
//        float* vmap_host_g_ = new float[rows * cols];
//        memset(&vmap_host_g_[0], 0, buffersize);
//        vmaps_g_prev_[i].download(vmap_host_g_, step * cols);
//        PlaneExtraction* pe_g = new PlaneExtraction;
//        pe_g->SetParametersPF(para_file);
//        pe_g->processOneframe(vmap_host_g_, rows / 3, cols, i);

//        float* vmap_host_curr_ = new float[rows * cols];
//        memset(&vmap_host_curr_[0], 0, buffersize);
//        vmaps_curr_[i].download(vmap_host_curr_, step * cols);
//        PlaneExtraction* pe_f = new PlaneExtraction;
//        pe_f->SetParametersPF(para_file);
//        pe_f->processOneframe(vmap_host_curr_, rows / 3, cols, i);

//        //find plane correpondence(match) between rendered texture and captured depth map
//        std::vector<std::pair<int , int>> plane_matches;

//        for(int m = 0; m < pe_g->getPointFitter()->extractedPlanes.size(); m++)
//        {
//            Eigen::Vector3d pl_g_center, pl_g_norm;
//            pl_g_center << pe_g->getPointFitter()->extractedPlanes[m]->center[0], pe_g->getPointFitter()->extractedPlanes[m]->center[1],
//                    pe_g->getPointFitter()->extractedPlanes[m]->center[2];
//            pl_g_norm << pe_g->getPointFitter()->extractedPlanes[m]->normal[0], pe_g->getPointFitter()->extractedPlanes[m]->normal[1],
//                    pe_g->getPointFitter()->extractedPlanes[m]->normal[2];
//            for(int n = 0; n < pe_f->getPointFitter()->extractedPlanes.size(); n++)
//            {
//                Eigen::Vector3d pl_f_center, pl_f_norm;
//                pl_f_center << pe_f->getPointFitter()->extractedPlanes[n]->center[0], pe_f->getPointFitter()->extractedPlanes[n]->center[1],
//                        pe_f->getPointFitter()->extractedPlanes[n]->center[2];
//                pl_f_norm << pe_f->getPointFitter()->extractedPlanes[n]->normal[0], pe_f->getPointFitter()->extractedPlanes[n]->normal[1],
//                        pe_f->getPointFitter()->extractedPlanes[n]->normal[2];
//                //normal deviation < 10 deg + center distance + center to plane distance < 50
//                double normal_dev = pl_g_norm.dot(pl_f_norm);
//                if(fabs(normal_dev) > 0.95)
//                {
//                    //unit: mm
//                    double center_dist = sqrt((pl_g_center-pl_f_center).dot(pl_g_center-pl_f_center));
//                    double center_g_to_planef_dist = fabs((pl_g_center - pl_f_center).dot(pl_f_norm));
//                    double center_f_to_planeg_dist = fabs((pl_f_center - pl_g_center).dot(pl_g_norm));
////                    std::cout << "center_dist : " << center_dist << std::endl;
////                    std::cout << "center_g_to_planef_dist : " << center_g_to_planef_dist << std::endl;
////                    std::cout << "center_f_to_planeg_dist : " << center_f_to_planeg_dist << std::endl;
//                    if(center_dist < 500 && center_g_to_planef_dist < 50 && center_f_to_planeg_dist < 50)
//                    {
//                        //set as matched plane
//                        std::pair<int, int> match;
//                        match.first = m;
//                        match.second = n;
//                        plane_matches.push_back(match);
//                    }

//                 }
//             }
//        }
////        std::cout<<"find plane matches: "<<std::endl;
////        for(int k = 0; k < plane_matches.size(); k++)
////        {
////            std::cout << plane_matches[k].first <<" "<< plane_matches[k].second << "\n";
////        }

//        //covert to plane_match vector with member ship
//        std::vector<unsigned short> pl_mat_gv_(rows * cols / 3, 0);
//        std::vector<unsigned short> pl_mat_fv_(rows * cols / 3 , 0);
//        //fill in the vector;
//        for(int k = 0; k < plane_matches.size(); k++)
//        {
//            for(int m = 0; m < pe_g->p_member_ship[plane_matches[k].first].size(); m++)
//            {
//                int pid = pe_g->p_member_ship[plane_matches[k].first][m];
//                pl_mat_gv_[pid] = k + 1;
//            }

//            for(int m = 0; m < pe_f->p_member_ship[plane_matches[k].second].size(); m++)
//            {
//                int pid = pe_f->p_member_ship[plane_matches[k].second][m];
//                pl_mat_fv_[pid] = k + 1;
//            }
//        }

       // plane_match_map_g_[i].upload(pl_mat_gv_,sizeof(unsigned short) * cols);
        //plane_match_map_curr_[i].upload(pl_mat_fv_,sizeof(unsigned short) * cols);

//        delete [] vmap_host_g_;
//        delete [] vmap_host_curr_;
//        delete pe_g;
//        delete pe_f;
//    }
}

void RGBDOdometry::correspondPlaneSearchRANSAC(const Eigen::Matrix4f& currpose)
{
    //in GPU rows = rows * 4;
    //first extract plane on the detailed(highest level)frame, then down sample to the other map
    //in GPU rows = rows * 4;
    int rows = vmaps_curr_[0].rows();
    int cols = vmaps_curr_[0].cols();

    //buffer size
    size_t step = sizeof(float);
    const size_t buffersize = step * rows * cols;

    //process plane extraction from global map;
    std::string type = "g";
    float* vmap_host_g_ = new float[rows * cols];
    memset(&vmap_host_g_[0], 0, buffersize);
    vmaps_g_prev_[0].download(vmap_host_g_, step * cols);
    float* nmap_host_g_ = new float[rows * cols];
    memset(&nmap_host_g_[0], 0, buffersize);
    nmaps_g_prev_[0].download(nmap_host_g_, step * cols);
    PlaneExtraction* plEx_g = new PlaneExtraction;
    plEx_g->processOneframeRansac(vmap_host_g_, nmap_host_g_, rows / 4, cols, type, Eigen::Matrix4f::Identity());

    //process data from current frame
    type = "curr";
    float* vmap_host_curr_ = new float[rows * cols];
    memset(&vmap_host_curr_[0], 0, buffersize);
    vmaps_curr_[0].download(vmap_host_curr_, step * cols);
    float* nmap_host_curr_ = new float[rows * cols];
    memset(&nmap_host_curr_[0], 0, buffersize);
    nmaps_curr_[0].download(nmap_host_curr_, step * cols);
    PlaneExtraction* plEx_curr = new PlaneExtraction;
    plEx_curr->processOneframeRansac(vmap_host_curr_, nmap_host_curr_, rows / 4, cols, type, currpose);

    //plane correspondence match; plane_matches.push_back(match);
    std::vector<std::pair<int, int>> plane_matches;
    for(int m = 0; m < plEx_g->planeParaSet.size(); m++)
    {
        Eigen::Vector3f pl_g_center(plEx_g->planeParaSet[m].pos);
        Eigen::Vector3f pl_g_normal(plEx_g->planeParaSet[m].normal);
        for(int n = 0; n < plEx_curr->planeParaSet.size(); n++){
            Eigen::Vector3f pl_curr_center(plEx_curr->planeParaSet[n].pos);
            Eigen::Vector3f pl_curr_normal(plEx_curr->planeParaSet[n].normal);

            //normal deviation < 10 deg + center distance + center to plane distance < 0.015(same as inlier distance threshold)
            double normal_dev = pl_g_normal.dot(pl_curr_normal);
            if(fabs(normal_dev) > 0.95)
            {
                double center_dist = sqrt((pl_g_center-pl_curr_center).dot(pl_g_center-pl_curr_center));
                double center_g_to_plane_curr_dist = fabs((pl_g_center - pl_curr_center).dot(pl_curr_normal));
                double center_curr_to_plane_g_dist = fabs((pl_curr_center - pl_g_center).dot(pl_g_normal));
                std::cout << "center_dist : " << center_dist << std::endl;
                std::cout << "center_g_to_planef_dist : " << center_g_to_plane_curr_dist << std::endl;
                std::cout << "center_f_to_planeg_dist : " << center_curr_to_plane_g_dist << std::endl;
                if(center_dist < 0.3 && center_g_to_plane_curr_dist < 0.015 && center_curr_to_plane_g_dist < 0.015)
                {
                   //set as matched plane
                   std::pair<int, int> match;
                   match.first = m;
                   match.second = n;
                   plane_matches.push_back(match);
                }
            }
        }

    }
    std::cout<<"find plane matches: "<<std::endl;

    for(int k = 0; k < plane_matches.size(); k++)
    {
        std::cout << plane_matches[k].first <<" "<< plane_matches[k].second << "\n";
    }
    //merge in groups
    //first step: merge second element when first one is the same
    std::vector<std::pair<std::vector<int>, std::vector<int>>> group_match;
    group_match.reserve(1000);
    int last =  -1;
    for(int k = 0; k < plane_matches.size(); k++)
    {
        std::pair<std::vector<int>, std::vector<int>> match;
        match.first.push_back(plane_matches[k].first);
        match.second.push_back(plane_matches[k].second);
        if(plane_matches[k].first == last)
        {
            size_t sizeg = group_match.size();
            group_match[sizeg - 1].second.push_back(plane_matches[k].second);
            continue;
        }
        group_match.push_back(match);
        last = plane_matches[k].first;
    }

    //print to screen to check the results
    std::cout<<"after 1st step: element match as follows: "<< std::endl;
    for(int i = 0; i < group_match.size(); i++)
    {
        for (int j = 0; j < group_match[i].first.size(); j++)
        {
            std::cout << group_match[i].first[j] << " ";
        }
        std::cout << "-- ";
        for (int j = 0; j < group_match[i].second.size(); j++)
        {
            std::cout << group_match[i].second[j] << " ";
        }
        std::cout << std::endl;
    }

    //second step: merge first element when second has the same element
    for(int k = 0; k < group_match.size(); k++)
    {
        for(int l = k + 1; l < group_match.size(); l++)
        {
            for(int m = 0; m < group_match[k].second.size(); m++)
            {
                std::vector<int>::iterator it;
                it = std::find (group_match[l].second.begin(), group_match[l].second.end(), group_match[k].second[m]);
                //if find match, proceed first element and second element
                if(it != group_match[l].second.end())
                {
                    group_match[k].first.insert(group_match[k].first.end(), group_match[l].first.begin(), group_match[l].first.end());
                    for(int n = 0; n < group_match[l].second.size(); n++)
                    {
                        std::vector<int>::iterator it_k;
                        it_k = std::find(group_match[k].second.begin(), group_match[k].second.end(), group_match[l].second[n]);
                        if(it_k != group_match[k].second.end())
                            continue;
                        group_match[k].second.push_back(group_match[l].second[n]);
                    }
                    group_match.erase(group_match.begin() + l);
                    break; //don't forget to break out the loop;
                }
                //erase the current match group element
            }
        }
    }

    //print to screen to check the results
    std::cout<<"after 2nd step: element match as follows: "<< std::endl;
    for(int i = 0; i < group_match.size(); i++)
    {
        for (int j = 0; j < group_match[i].first.size(); j++)
        {
            std::cout << group_match[i].first[j] << " ";
        }
        std::cout << "-- ";
        for (int j = 0; j < group_match[i].second.size(); j++)
        {
            std::cout << group_match[i].second[j] << " ";
        }
        std::cout << std::endl;
    }

    //see results
    std::vector<Eigen::Vector3i> colorSet = pseudocolor(group_match.size());

    std::ofstream group_match_g;
    std::ofstream group_match_curr;
    group_match_g.open("group_match_g_level0.txt");
    group_match_curr.open("group_match_curr_level0.txt");
    //map to the image
    //convert to plane match vector with membership
    std::vector<unsigned short> pl_mat_gv(rows * cols / 4, 0);
    std::vector<unsigned short> pl_mat_fv(rows * cols / 4, 0);
    for(int i = 0; i < group_match.size(); i++)
    {
        for(int j = 0; j < group_match[i].first.size(); j++)
        {
            int planeID = group_match[i].first[j];
            for(int pID = plEx_g->shape2pointAssocSet[planeID].start; pID < plEx_g->shape2pointAssocSet[planeID].start + plEx_g->shape2pointAssocSet[planeID].interval; pID++)
            {
                int pixID = plEx_g->pc.at(pID).index;
                pl_mat_gv[pixID] = i + 1;
//                group_match_g << plEx_g->pc.at(pID).pos[0] <<" " << plEx_g->pc.at(pID).pos[1] <<" "<< plEx_g->pc.at(pID).pos[2] << " "
//                              << -plEx_g->pc.at(pID).normal[0] << " " << -plEx_g->pc.at(pID).normal[1] << " " << -plEx_g->pc.at(pID).normal[2] << " "
//                              << colorSet[i](0) <<" "<< colorSet[i](1) << " "<< colorSet[i](2) <<"\n";
            }

        }
        for(int j = 0; j < group_match[i].second.size(); j++)
        {
            int planeID = group_match[i].second[j];
            for(int pID = plEx_curr->shape2pointAssocSet[planeID].start; pID < plEx_curr->shape2pointAssocSet[planeID].start + plEx_curr->shape2pointAssocSet[planeID].interval; pID++)
            {
                int pixID = plEx_curr->pc.at(pID).index;
                pl_mat_fv[pixID] = i + 1;
//                group_match_curr << plEx_curr->pc.at(pID).pos[0] <<" " << plEx_curr->pc.at(pID).pos[1] <<" "<< plEx_curr->pc.at(pID).pos[2] << " "
//                              << -plEx_curr->pc.at(pID).normal[0] << " " << -plEx_curr->pc.at(pID).normal[1] << " " << -plEx_curr->pc.at(pID).normal[2] << " "
//                              << colorSet[i](0) <<" "<< colorSet[i](1) << " "<< colorSet[i](2) <<"\n";
            }

        }
    }
    group_match_g.close();
    group_match_curr.close();
    std::cout <<"demension of local match before: "<<plane_match_map_curr_[0].cols() <<" "<< plane_match_map_curr_[0].rows()<< std::endl;

    plane_match_map_g_[0].upload(pl_mat_gv, /*sizeof(unsigned short) **/ cols);
    plane_match_map_curr_[0].upload(pl_mat_fv,/*sizeof(unsigned short) **/ cols);

    for(int pyr = 1; pyr < NUM_PYRS; pyr++)
    {
        resizePlaneMap(plane_match_map_curr_[pyr - 1], plane_match_map_curr_[pyr]);
        std::cout <<"demension of local match: "<<plane_match_map_curr_[pyr-1].cols() <<" "<< plane_match_map_curr_[pyr-1].rows()<< std::endl;
        std::cout <<"demension of vmap: "<<vmaps_curr_[pyr-1].cols() <<" "<< vmaps_curr_[pyr-1].rows()<< std::endl;
        resizePlaneMap(plane_match_map_g_[pyr - 1], plane_match_map_g_[pyr]);
        std::cout <<"demension of global match: "<<plane_match_map_g_[pyr-1].cols() <<" "<< plane_match_map_g_[pyr-1].rows()<< std::endl;
    }

    delete [] vmap_host_g_;
    delete [] nmap_host_g_;
    delete [] vmap_host_curr_;
    delete [] nmap_host_curr_;
    delete plEx_g;
    delete plEx_curr;
}

void RGBDOdometry::populateRGBDData(GPUTexture * rgb,
                                    DeviceArray2D<float> * destDepths,
                                    DeviceArray2D<unsigned char> * destImages)
{
    verticesToDepth(vmaps_tmp, destDepths[0], maxDepthRGB);   //project vertices to depth image

    for(int i = 0; i + 1 < NUM_PYRS; i++)                     //down sample depth map to form a hierarchical structure
    {
        pyrDownGaussF(destDepths[i], destDepths[i + 1]);
    }

    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    imageBGRToIntensity(textPtr, destImages[0]);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);    //asign rgb image to cuda resource destImage[]

    for(int i = 0; i + 1 < NUM_PYRS; i++)            //downsample destImage for form a corresponding hierarchical structure
    {
        pyrDownUcharGauss(destImages[i], destImages[i + 1]);
    }

    cudaDeviceSynchronize();
}

void RGBDOdometry::initRGBModel(GPUTexture * rgb)
{
    //NOTE: This depends on vmaps_tmp containing the corresponding depth from initICPModel
    populateRGBDData(rgb, &lastDepth[0], &lastImage[0]);
}

void RGBDOdometry::initRGB(GPUTexture * rgb)
{
    //NOTE: This depends on vmaps_tmp containing the corresponding depth from initICP
    populateRGBDData(rgb, &nextDepth[0], &nextImage[0]);
}

void RGBDOdometry::initCurvature(GPUTexture* curvk1, GPUTexture* curvk2)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &curvk1->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, curvk1->cudaRes, 0, 0);
    cudaMemcpyFromArray(ck1maps_tmp.ptr(), textPtr, 0, 0, ck1maps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &curvk1->cudaRes);
    copyCurvatureMap(ck1maps_tmp, ck1maps_curr_[0], GlobalStateParam::get().preprocessingCurvValidThreshold);

    cudaGraphicsMapResources(1, &curvk2->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, curvk2->cudaRes, 0, 0);
    cudaMemcpyFromArray(ck2maps_tmp.ptr(), textPtr, 0, 0, ck2maps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &curvk2->cudaRes);
    copyCurvatureMap(ck2maps_tmp, ck2maps_curr_[0], GlobalStateParam::get().preprocessingCurvValidThreshold);

    for(int i = 1; i < NUM_PYRS; i++)
    {
        resizeCMap(ck1maps_curr_[i - 1], ck1maps_curr_[i]);
        resizeCMap(ck2maps_curr_[i - 1], ck2maps_curr_[i]);
    }
    cudaDeviceSynchronize();
}

void RGBDOdometry::initCurvatureModel(GPUTexture* curvk1Model, GPUTexture* curvk2Model,
                                      const Eigen::Matrix4f & modelPose)
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &curvk1Model->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, curvk1Model->cudaRes, 0, 0);
    cudaMemcpyFromArray(ck1maps_tmp.ptr(), textPtr, 0, 0, ck1maps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &curvk1Model->cudaRes);
    copyCurvatureMap(ck1maps_tmp, ck1maps_g_prev_[0], GlobalStateParam::get().preprocessingCurvValidThreshold);

    cudaGraphicsMapResources(1, &curvk2Model->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, curvk2Model->cudaRes, 0, 0);
    cudaMemcpyFromArray(ck2maps_tmp.ptr(), textPtr, 0, 0, ck2maps_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &curvk2Model->cudaRes);
    copyCurvatureMap(ck2maps_tmp, ck2maps_g_prev_[0], GlobalStateParam::get().preprocessingCurvValidThreshold);

    for(int i = 1; i < NUM_PYRS; i++)
    {
        resizeCMap(ck1maps_g_prev_[i - 1], ck1maps_g_prev_[i]);
        resizeCMap(ck2maps_g_prev_[i - 1], ck2maps_g_prev_[i]);
    }
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcam = modelPose.topLeftCorner(3, 3);
    Eigen::Vector3f tcam = modelPose.topRightCorner(3, 1);

    mat33 device_Rcam = Rcam;
    float3 device_tcam = *reinterpret_cast<float3*>(tcam.data());

    //transform previous curvature map to global model frame (reference)
    for(int i = 0; i < NUM_PYRS; ++i)
    {
        transformCurvMaps(ck1maps_g_prev_[i], ck2maps_g_prev_[i], device_Rcam, device_tcam, ck1maps_g_prev_[i], ck2maps_g_prev_[i]);
    }
    cudaDeviceSynchronize();
}

void RGBDOdometry::initICPweight(GPUTexture* icpWeight){

    cudaArray* textPtr;
    cudaGraphicsMapResources(1, &icpWeight->cudaRes);
    cudaGraphicsSubResourceGetMappedArray(&textPtr, icpWeight->cudaRes, 0, 0);
    cudaMemcpyFromArray(icpweight_tmp.ptr(), textPtr, 0, 0, icpweight_tmp.sizeBytes(), cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1, &icpWeight->cudaRes);
    copyicpWeightMap(icpweight_tmp, icpWeightMap_[0]);

    for(int i = 1; i < NUM_PYRS; i++)
    {
        resizeicpWeightMap(icpWeightMap_[i - 1], icpWeightMap_[i]);
    }
    cudaDeviceSynchronize();
}

void RGBDOdometry::initFirstRGB(GPUTexture * rgb)  //for so3???
{
    cudaArray * textPtr;

    cudaGraphicsMapResources(1, &rgb->cudaRes);

    cudaGraphicsSubResourceGetMappedArray(&textPtr, rgb->cudaRes, 0, 0);

    //color is from BGR
    imageBGRToIntensity(textPtr, lastNextImage[0]);

    cudaGraphicsUnmapResources(1, &rgb->cudaRes);

    for(int i = 0; i + 1 < NUM_PYRS; i++)
    {
        pyrDownUcharGauss(lastNextImage[i], lastNextImage[i + 1]);
    }
}

void RGBDOdometry::getIncrementalTransformation(Eigen::Vector3f & trans,
                                                Eigen::Matrix<float, 3, 3, Eigen::RowMajor> & rot,
                                                const bool & rgbOnly,
                                                const float & icpWeight,
                                                const bool & pyramid,
                                                const bool & fastOdom,
                                                const bool & so3,
                                                const bool & if_curvature_info,
                                                const int index_frame)
{
    bool icp = !rgbOnly && icpWeight > 0;      //if we only use icp
    bool rgb = rgbOnly || icpWeight < 100;     //RGB information is applied

    //initialize previous pose
    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev = rot;
    Eigen::Vector3f tprev = trans;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rcurr = Rprev;
    Eigen::Vector3f tcurr = tprev;

    if(rgb)

    {
        for(int i = 0; i < NUM_PYRS; i++)
        {
            computeDerivativeImages(nextImage[i], nextdIdx[i], nextdIdy[i]);
        }
    }

    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> resultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

    if(so3)
    {
        int pyramidLevel = 2;

        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> R_lr = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>::Identity();
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();

        K(0, 0) = intr(pyramidLevel).fx;
        K(1, 1) = intr(pyramidLevel).fy;
        K(0, 2) = intr(pyramidLevel).cx;
        K(1, 2) = intr(pyramidLevel).cy;
        K(2, 2) = 1;

        float lastError = std::numeric_limits<float>::max() / 2;
        float lastCount = std::numeric_limits<float>::max() / 2;

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> lastResultR = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Identity();

        for(int i = 0; i < 10; i++)
        {
            Eigen::Matrix<float, 3, 3, Eigen::RowMajor> jtj;
            Eigen::Matrix<float, 3, 1> jtr;

            //transfer from on image space to another image space
            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> homography = K * resultR * K.inverse();

            mat33 imageBasis;
            memcpy(&imageBasis.data[0], homography.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_inv = K.inverse();
            mat33 kinv;
            memcpy(&kinv.data[0], K_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K_R_lr = K * resultR;
            mat33 krlr;
            memcpy(&krlr.data[0], K_R_lr.cast<float>().eval().data(), sizeof(mat33));

            float residual[2];

            so3Step(lastNextImage[pyramidLevel],
                    nextImage[pyramidLevel],
                    imageBasis,
                    kinv,
                    krlr,
                    sumDataSO3,
                    outDataSO3,
                    jtj.data(),
                    jtr.data(),
                    &residual[0],
                    GPUConfig::getInstance().so3StepThreads,
                    GPUConfig::getInstance().so3StepBlocks);

            //lastSO3Error , lastSO3Count
            lastSO3Error = sqrt(residual[0]) / residual[1];
            lastSO3Count = residual[1];

            //Converged
            if(lastSO3Error < lastError && lastCount == lastSO3Count)
            {
                break;
            }
            else if(lastSO3Error > lastError + 0.001)
            {
                lastSO3Error = lastError;
                lastSO3Count = lastCount;
                resultR = lastResultR;
                break;
            }

            lastError = lastSO3Error;
            lastCount = lastSO3Count;
            lastResultR = resultR;

            Eigen::Vector3f delta = jtj.ldlt().solve(jtr);

            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> rotUpdate = OdometryProvider::rodrigues(delta.cast<double>());

            R_lr = rotUpdate.cast<float>() * R_lr;

            for(int x = 0; x < 3; x++)
            {
                for(int y = 0; y < 3; y++)
                {
                    resultR(x, y) = R_lr(x, y);
                }
            }
        }
    }

    iterations[0] = fastOdom ? 3 : 10;
    iterations[1] = pyramid ? 5 : 0;
    iterations[2] = pyramid ? 4 : 0;

    Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Rprev_inv = Rprev.inverse();
    mat33 device_Rprev_inv = Rprev_inv;
    float3 device_tprev = *reinterpret_cast<float3*>(tprev.data());

    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> resultRt = Eigen::Matrix<double, 4, 4, Eigen::RowMajor>::Identity();

    if(so3)
    {
        for(int x = 0; x < 3; x++)
        {
            for(int y = 0; y < 3; y++)
            {
                resultRt(x, y) = resultR(x, y);
            }
        }
    }

//    std::ofstream ICPERROR_Export;
//    char ex_name[256], file_n[50];
//    strcpy(ex_name, "/home/robin/PycharmProjects/Plot_tool/wall/");
//    snprintf(file_n,sizeof(file_n),"icp_error_%d.txt",index_frame);
//    strcat(ex_name, file_n);
//    ICPERROR_Export.open(ex_name);--------------------------------------

    bool use_sparseICP = GlobalStateParam::get().registrationICPUseSparseICP;

    for(int i = NUM_PYRS - 1; i >= 0; i--)
    {
        if(rgb)
        {
            projectToPointCloud(lastDepth[i], pointClouds[i], intr, i);
        }

        Eigen::Matrix<double, 3, 3, Eigen::RowMajor> K = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>::Zero();    //camera intrinsic projection

        //for each hierachies, intrinsic parameters is different
        K(0, 0) = intr(i).fx;
        K(1, 1) = intr(i).fy;
        K(0, 2) = intr(i).cx;
        K(1, 2) = intr(i).cy;
        K(2, 2) = 1;

        lastRGBError = std::numeric_limits<float>::max();  //to represents the max float

        //--------lambdamap initilization with all zeros------------------//
        if(use_sparseICP)
        {
            float3* init_lambda = new float3[lambdaMap[i].rows() * lambdaMap[i].cols()];
            for (int init = 0; init <lambdaMap[i].rows() * lambdaMap[i].cols(); init++) {
                float3 init_zeros;
                init_zeros.x = 0.0f;
                init_zeros.y = 0.0f;
                init_zeros.z = 0.0f;
                init_lambda[init] = init_zeros;
            }
            lambdaMap[i].upload(init_lambda, sizeof(float3) * lambdaMap[i].cols(), lambdaMap[i].rows(), lambdaMap[i].cols());
            delete[] init_lambda;
        }
        //--------lambdamap initilization------------------//

        for(int j = 0; j < iterations[i]; j++)
        {

            Eigen::Matrix<double, 4, 4, Eigen::RowMajor> Rt = resultRt.inverse();
            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R = Rt.topLeftCorner(3, 3);
            Eigen::Matrix<double, 3, 3, Eigen::RowMajor> KRK_inv = K * R * K.inverse();

            mat33 krkInv;
            memcpy(&krkInv.data[0], KRK_inv.cast<float>().eval().data(), sizeof(mat33));

            Eigen::Vector3d Kt = Rt.topRightCorner(3, 1);
            Kt = K * Kt;
            float3 kt = {(float)Kt(0), (float)Kt(1), (float)Kt(2)};

            int sigma = 0;
            int rgbSize = 0;

            if(rgb)
            {
                computeRgbResidual(pow(minimumGradientMagnitudes[i], 2.0) / pow(sobelScale, 2.0),
                                   nextdIdx[i],
                                   nextdIdy[i],
                                   lastDepth[i],
                                   nextDepth[i],
                                   lastImage[i],
                                   nextImage[i],
                                   corresImg[i],
                                   sumResidualRGB,
                                   maxDepthDeltaRGB,
                                   kt,
                                   krkInv,
                                   sigma,
                                   rgbSize,
                                   GPUConfig::getInstance().rgbResThreads,
                                   GPUConfig::getInstance().rgbResBlocks);
            }

            float sigmaVal = std::sqrt((float)sigma / rgbSize == 0 ? 1 : rgbSize);
            float rgbError = std::sqrt(sigma) / (rgbSize == 0 ? 1 : rgbSize);

            if(rgbOnly && rgbError > lastRGBError)
            {
                break;
            }

            lastRGBError = rgbError;
            lastRGBCount = rgbSize;

            if(rgbOnly)
            {
                //Signals the internal optimisation to weight evenly
                sigmaVal = -1;   
            }

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_icp;
            Eigen::Matrix<float, 6, 1> b_icp;

            mat33 device_Rcurr = Rcurr;
            float3 device_tcurr = *reinterpret_cast<float3*>(tcurr.data());

            DeviceArray2D<float>& vmap_curr = vmaps_curr_[i];
            DeviceArray2D<float>& nmap_curr = nmaps_curr_[i];
            DeviceArray2D<float>& ck1maps_curr = ck1maps_curr_[i];
            DeviceArray2D<float>& ck2maps_curr = ck2maps_curr_[i];

            DeviceArray2D<float>& vmap_g_prev = vmaps_g_prev_[i];
            DeviceArray2D<float>& nmap_g_prev = nmaps_g_prev_[i];
            DeviceArray2D<float>& ck1maps_g_prev = ck1maps_g_prev_[i];
            DeviceArray2D<float>& ck2maps_g_prev = ck2maps_g_prev_[i];

            DeviceArray2D<float>& icpWeightMap = icpWeightMap_[i];

            DeviceArray2D<unsigned short>& plane_match_map_curr = plane_match_map_curr_[i];
            DeviceArray2D<unsigned short>& plane_match_map_g = plane_match_map_g_[i];

            float residual[2];
            if(icp)
            {
                icpStep(device_Rcurr,
                        device_tcurr,
                        vmap_curr,
                        nmap_curr,
                        ck1maps_curr,
                        ck2maps_curr,
                        plane_match_map_curr,
                        i,
                        device_Rprev_inv,
                        device_tprev,
                        intr(i),
                        vmap_g_prev,
                        nmap_g_prev,
                        ck1maps_g_prev,
                        ck2maps_g_prev,
                        icpWeightMap,
                        plane_match_map_g,
                        corresICP[i],
                        cuda_out_[i],
                        z_thrinkMap[i],
                        lambdaMap[i],
                        distThres_,
                        angleThres_,
                        curvatureThres_,
                        GlobalStateParam::get().registrationICPUseCoorespondenceSearch,
                        GlobalStateParam::get().registrationICPNeighborSearchRadius,
                        if_curvature_info,
                        use_sparseICP,
                        sumDataSE3,
                        outDataSE3,
                        A_icp.data(),
                        b_icp.data(),
                        &residual[0],
                        GPUConfig::getInstance().icpStepThreads,
                        GPUConfig::getInstance().icpStepBlocks);
                }

            //download corresICP to cpu
            //save to file for visualization
//           if(GlobalStateParam::get().if_make_fragments && i == 0 && j == iterations[i] - 1)
//           {
//               size_t step = sizeof(int2);
//               size_t pixel_N = corresICP[i].cols() * corresICP[i].rows();
//               const size_t buffersize = step * corresICP[i].cols() * corresICP[i].rows();
//               int2* host_corresICP = new int2[pixel_N];
//               memset(&host_corresICP[0], 0, buffersize);
//               corresICP[i].download(host_corresICP, step * corresICP[i].cols());
//               getLastCorrespondence(host_corresICP, corresICP[i].rows(), corresICP[i].cols());
//               //saveCorrepICPsave(host_corresICP, pixel_N, index_frame , i , j);
//               delete [] host_corresICP;
//           }
           //download to txt file

//           if(i == 0 && j == iterations[i] - 1)
//           {
//               size_t step = sizeof(float4);
//               size_t pixel_N = cuda_out_[i].cols() * cuda_out_[i].rows();
//               const size_t buffersize = step * cuda_out_[i].cols() * cuda_out_[i].rows();
//               float4* host_out = new float4[pixel_N];
//               memset(&host_out[0], 0, buffersize);
//               cuda_out_[i].download(host_out, step * cuda_out_[i].cols());
////               saveCudaAttrib(host_out, cuda_out_[i].cols(), cuda_out_[i].rows(),index_frame, i , j);
//               float sum = 0.0;
//               int count = 0;
//               for(int tt = 0; tt < cuda_out_[i].cols() * cuda_out_[i].rows(); tt++)
//               {
//                   if(host_out[tt].x > 0)
//                   {
//                       sum = sum + host_out[tt].x;
//                       count++;
//                   }
//               }
//               float average_distance_error = sum / count;
////               std::cout << sum << " " << count << " " <<  average_distance_error << std::endl;
//               dist_error_points.push_back(average_distance_error);
//               delete [] host_out;
//            }
            lastICPError = sqrt(residual[0]) / residual[1];
            lastICPCount = residual[1];

            //ICPERROR_Export << lastICPError << "\n";

            Eigen::Matrix<float, 6, 6, Eigen::RowMajor> A_rgbd;
            Eigen::Matrix<float, 6, 1> b_rgbd;

            if(rgb)
            {
                rgbStep(corresImg[i],
                        sigmaVal,
                        pointClouds[i],
                        intr(i).fx,
                        intr(i).fy,
                        nextdIdx[i],
                        nextdIdy[i],
                        GlobalStateParam::get().registrationColorUseRGBGrad,
                        sobelScale,
                        sumDataSE3,
                        outDataSE3,
                        A_rgbd.data(),
                        b_rgbd.data(),
                        GPUConfig::getInstance().rgbStepThreads,
                        GPUConfig::getInstance().rgbStepBlocks);
            }

            Eigen::Matrix<double, 6, 1> result;
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_rgbd = A_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> dA_icp = A_icp.cast<double>();
            Eigen::Matrix<double, 6, 1> db_rgbd = b_rgbd.cast<double>();
            Eigen::Matrix<double, 6, 1> db_icp = b_icp.cast<double>();

            if(icp && rgb)
            {
                double w = icpWeight;
                lastA = dA_rgbd + w * w * dA_icp;
                lastb = db_rgbd + w * db_icp;
                result = lastA.ldlt().solve(lastb);
            }
            else if(icp)
            {
                lastA = dA_icp;
                lastb = db_icp;
                result = lastA.ldlt().solve(lastb);
            }
            else if(rgb)
            {
                lastA = dA_rgbd;
                lastb = db_rgbd;
                result = lastA.ldlt().solve(lastb);
            }
            else
            {
                assert(false && "Control shouldn't reach here");
            }

            Eigen::Isometry3f rgbOdom;

            OdometryProvider::computeUpdateSE3(resultRt, result, rgbOdom);

            Eigen::Isometry3f currentT;
            currentT.setIdentity();
            currentT.rotate(Rprev);
            currentT.translation() = tprev;

            currentT = currentT * rgbOdom.inverse();

            tcurr = currentT.translation();
            Rcurr = currentT.rotation();

            //update lambdaMap(for sparse ICP)
            mat33 device_Rcurr_update = Rcurr;
            float3 device_tcurr_update = *reinterpret_cast<float3*>(tcurr.data());
            if(use_sparseICP)
            {
                updateLambdaMap(device_Rcurr_update,
                                device_tcurr_update,
                                vmap_curr,
                                nmap_curr,
                                device_Rprev_inv,
                                device_tprev,
                                intr(i),
                                vmap_g_prev,
                                nmap_g_prev,
                                corresICP[i],
                                z_thrinkMap[i],
                                lambdaMap[i],
                                distThres_,
                                angleThres_,
                                GPUConfig::getInstance().icpStepThreads,
                                GPUConfig::getInstance().icpStepBlocks);
            }
        }
    }

   // ICPERROR_Export.close();
    if(rgb && (tcurr - tprev).norm() > 0.3)
    {
        Rcurr = Rprev;
        tcurr = tprev;
    }

    //swap NextImage to lastNextImage
    if(so3)
    {
        for(int i = 0; i < NUM_PYRS; i++)
        {
            std::swap(lastNextImage[i], nextImage[i]);
        }
    }

    trans = tcurr;
    rot = Rcurr;
}

void RGBDOdometry::getLastCorrespondence(int2* host_corresp, int rows, int cols)
{
    //get correspondence map; from the last iteration and in detailed texture
    int stride = rows * cols;
    for(int i = 0; i < stride; i++)
    {
        int u_s, v_s, u_t, v_t;
        u_s = host_corresp[i].x;
        if(u_s == -1)
            continue;
        v_s = host_corresp[i].y;
        v_t = i / cols;
        u_t = i - v_t * cols;

        Eigen::Vector4i cp(u_s, v_s, u_t, v_t);
        correspondence.push_back(cp);
    }
}

//Eigen::Matrix6d RGBDOdometry::CreateInformationMatrix(){

//    //get correspondence;  //use smoothed texture
//    std::cout << "we get " << correspondence.size() << " correspondence pairs" << std::endl;
//    //get target map from gpu, rows = 4 * rows
//    int rows = vmaps_curr_[0].rows();
//    int cols = vmaps_curr_[0].cols();

//    size_t step = sizeof(float);
//    const size_t buffersize = step * rows * cols;
//    float* vmap_host_curr = new float[rows * cols];
//    memset(&vmap_host_curr[0], 0, buffersize);
//    vmaps_curr_[0].download(vmap_host_curr, step * cols);

//    int stride = rows / 4 * cols;
//    //compute GTG here
//    Eigen::Matrix6d GTG = Eigen::Matrix6d::Identity();

//    Eigen::Matrix6d GTG_private = Eigen::Matrix6d::Identity();
//    Eigen::Vector6d G_r_private = Eigen::Vector6d::Zero();

//    for(int i = 0; i < correspondence.size(); i++){
//        int u_t = correspondence[i](2);
//        int v_t = correspondence[i](3);
//        int index_point = u_t + v_t * cols;

//        //target point position
//        double x = vmap_host_curr[index_point];
//        double y = vmap_host_curr[index_point + stride];
//        double z = vmap_host_curr[index_point + 2 * stride];

//        G_r_private.setZero();
//        G_r_private(1) = z;
//        G_r_private(2) = -y;
//        G_r_private(3) = 1.0;
//        GTG_private.noalias() += G_r_private * G_r_private.transpose();
//        G_r_private.setZero();
//        G_r_private(0) = -z;
//        G_r_private(2) = x;
//        G_r_private(4) = 1.0;
//        GTG_private.noalias() += G_r_private * G_r_private.transpose();
//        G_r_private.setZero();
//        G_r_private(0) = y;
//        G_r_private(1) = -x;
//        G_r_private(5) = 1.0;
//        GTG_private.noalias() += G_r_private * G_r_private.transpose();

//    }

//    delete [] vmap_host_curr;

//    //clean correspondence vector
//    correspondence.clear();

//    std::cout << "information matrix is:\n" << GTG_private << std::endl;

//    return GTG_private;
//}

void RGBDOdometry::addToPoseGraph(Eigen::Matrix3f Rprevious, Eigen::Vector3f tprevious, Eigen::Matrix3f Rcurrent, Eigen::Vector3f tcurrent,
                                  int index_frame, Eigen::Matrix4d Trans){
    //Note that the first node of the posegraph is initialized in the class constructor
    //Note that in odometry the transformation matrix is from current to previous, be care of that
    Eigen::Matrix4d pose_i = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d pose_j = Eigen::Matrix4d::Identity();
    pose_i.topRightCorner(3, 1) = tprevious.cast<double>();
    pose_i.topLeftCorner(3, 3) = Rprevious.cast<double>();

    pose_j.topRightCorner(3, 1) = tcurrent.cast<double>();
    pose_j.topLeftCorner(3, 3) = Rcurrent.cast<double>();

    std::cout <<"pose_i:\n" << pose_i << std::endl;
    std::cout <<"pose_j:\n" << pose_j << std::endl;

    Eigen::Matrix4d Trans_i_to_j = pose_j.inverse() * pose_i;
    //get information Matrix

//    Eigen::Matrix6d info = CreateInformationMatrix();

//    open3d::PoseGraphEdge e(index_frame - 1, index_frame, Trans_i_to_j, info);
    //pose_graph_.nodes_.push_back(pose_i);
//    pose_graph_.nodes_.push_back(pose_j);
//    pose_graph_.edges_.push_back(e);
    //
}

void RGBDOdometry::savePoseGraph(int index_fragment)
{
    char fragment_template[128];
    strcpy(fragment_template, "fragments/fragment_hrbf_");

    char suffix[64];
    snprintf(suffix, sizeof(suffix), "%03d.json", index_fragment);

    strcat(fragment_template, suffix);
//    std::cout <<"The node size: "<< pose_graph_.nodes_.size() << std::endl;
//    open3d::WritePoseGraph(fragment_template, pose_graph_);

}
Eigen::MatrixXd RGBDOdometry::getCovariance()
{
    return lastA.cast<double>().lu().inverse();
}

void RGBDOdometry::DownloadGPUMaps()
{
    for(int i = 0; i < 3; i++)
    {
        //rows  = 4 * rows in GPU
        int rows = vmaps_g_prev_[i].rows();
        int cols = vmaps_g_prev_[i].cols();
        //buffer size
        size_t step = sizeof(float);
        const size_t buffersize = step * rows * cols;

        float* vmap_host_g = new float[rows * cols];
        memset(&vmap_host_g[0], 0, buffersize);
        vmaps_g_prev_[i].download(vmap_host_g, step * cols);
        
        float* nmap_host_g = new float[rows * cols];
        memset(&nmap_host_g[0], 0, buffersize);
        nmaps_g_prev_[i].download(nmap_host_g, step * cols);

        float* curv_max = new float[rows * cols];
        memset(&curv_max[0], 0, buffersize);
        ck1maps_g_prev_[i].download(curv_max, step * cols);

        float* curv_min = new float[rows * cols];
        memset(&curv_min[0], 0, buffersize);
        ck2maps_g_prev_[i].download(curv_min, step * cols);

        short* lastimage_dx = new short[rows * cols / 4];
        memset(&lastimage_dx[0], 0, sizeof(short) * rows * cols / 4);
        nextdIdx[i].download(lastimage_dx, sizeof(short) * cols);

        short* lastimage_dy = new short[rows * cols / 4];
        memset(&lastimage_dy[0], 0, sizeof(short) * rows * cols / 4);
        nextdIdy[i].download(lastimage_dy, sizeof(short) * cols);

        float* icp_weight = new float[rows * cols / 4];
        memset(&icp_weight[0], 0, sizeof(float) * rows * cols / 4);
        icpWeightMap_[i].download(icp_weight, sizeof(float) * cols);


        savefilePLY(vmap_host_g, nmap_host_g, curv_max, curv_min,
                    lastimage_dx, lastimage_dy, icp_weight, rows / 4, cols, "prev", i);

        // float* vmap_host_curr = new float[rows * cols];
        // memset(&vmap_host_curr[0], 0, buffersize);
        // vmaps_curr_[i].download(vmap_host_curr, step * cols);
        // float* nmap_host_curr = new float[rows * cols];
        // memset(&nmap_host_curr[0], 0, buffersize);
        // nmaps_curr_[i].download(nmap_host_curr, step * cols);
        // savefilePLY(vmap_host_curr, nmap_host_curr, rows / 4, cols , "curr", i);

        delete [] vmap_host_g;
        delete [] nmap_host_g;
        delete [] lastimage_dx;
        delete [] lastimage_dy;
        delete [] curv_max;
        delete [] curv_min;
        delete [] icp_weight;


        // delete [] vmap_host_curr;
        // delete [] nmap_host_curr;
    }
}

void RGBDOdometry::savefilePLY(float* vertices, float* normals,float* curv_max, float* curv_min,
                               short* image_dx, short* image_dy, float* icp_weight, int rows, int cols, std::string type, int paramid){

    char file_name_full[256];
    strcpy(file_name_full, type.c_str());

    char filename[100];
    snprintf(filename, sizeof(filename), "_map_%d.ply", paramid);

    strcat(file_name_full, filename);

    std::ofstream fs;
    fs.open(file_name_full);

    int stride = rows * cols;

    //Write header
    fs<<"ply\n";
    fs<<"format"<<" ascii 1.0\n"<<"comment Created by Robin\n";

    //Vertices
    fs<<"element vertex "<<stride<<"\n";
    fs<<"property float x\n"
        "property float y\n"
        "property float z\n";

    //set point cloud color as white
    // fs<<"property uchar red\n"
    //     "property uchar green\n"
    //     "property uchar blue\n";

    fs<<"property float nx\n"
        "property float ny\n"
        "property float nz\n";

    fs<<"property float curv_max\n"
        "property float curv_min\n";

    fs<<"property float image_dx\n"
        "property float image_dy\n";

    fs<<"property float rgb_gradient_mag\n";

    fs<<"property float icp_weight\n";

    fs<<"end_header\n";

    float nan = std::numeric_limits<float>::quiet_NaN ();

    //PCL Load can not load NAN value so set invalid point 0 0 0;
    for(int i = 0; i < stride; i++)
    {
        if(std::isnan(vertices[i]) || std::isnan(vertices[i + stride])|| std::isnan(vertices[i +2 * stride]) || vertices[i + 2 * stride] == 0.0f ||
                std::isnan(normals[i]) || std::isnan(normals[i + stride])|| std::isnan(normals[i +2 * stride]))
        {
            fs << 0 << " " << 0 << " " << 0 << " "
            << 0 << " " << 0 << " " << 0 <<" " 
            << 0 << " " << 0 << " "
            << 0 << " " << 0 << " " << 0 << " " << 0
            //<< 1.0 << " " << 0 << " " << 0 <<
            <<"\n";
            continue;
        }
        fs << vertices[i] << " " << vertices[i + stride] << " " << vertices[i + 2 * stride] <<" "
           //<< 255 <<" "<< 255 <<" "<< 255 << " "
           << -normals[i] << " " << -normals[i + stride] << " " << -normals[i + 2 * stride] <<" "
           << curv_max[i + 3 * stride] << " " << curv_min[i + 3 * stride] << " "
           << image_dx[i] << " " << image_dy[i] << " " << sqrt(image_dx[i] * image_dx[i] + image_dy[i] * image_dy[i]) << " " << icp_weight[i]
           << "\n";
    }
    fs.close();

}

void RGBDOdometry::saveCorrepICPsave(int2* corresp, int stride,int frameID, int paramid, int iterative)
{
    char file_name_full[256];
    strcpy(file_name_full, "./corres/cor");

    char fn[100];
    snprintf(fn, sizeof(fn), "%d_level%d_iter%d.corres", frameID, paramid, iterative);

    strcat(file_name_full, fn);

    std::ofstream saveFile;

    saveFile.open(file_name_full);

    std::cout<<"stride is: "<< stride << std::endl;
    saveFile << stride << "\n";

    for(int i = 0; i < stride; i++)
    {
        saveFile << corresp[i].x << " " << corresp[i].y <<"\n";
    }
    saveFile.close();
}

float RGBDOdometry::saveCudaAttrib(float4* curv_out, int cols, int rows,int frameID, int paramid, int iterative)
{
    char file_name_full[256];
    strcpy(file_name_full, "./cuda_out/cur");

    char fn[100];
    snprintf(fn, sizeof(fn), "%d_level%d_iter%d.txt", frameID , paramid, iterative);

    strcat(file_name_full, fn);

    std::ofstream saveFile;

    saveFile.open(file_name_full);

    std::cout<<"stride is: "<< cols * rows << std::endl;
    saveFile << cols * rows << "\n";

    for(int i = 0; i < cols * rows; i++)
    {
        saveFile << curv_out[i].x << " " << curv_out[i].y <<" "<< curv_out[i].z<<" "<< curv_out[i].w <<"\n";
    }
    saveFile.close();

    return 0.0;
}

void RGBDOdometry::saveCudaAttrib(float3* curv_out, int cols, int rows,int frameID, int paramid, int iterative)
{
    char file_name_full[256];
    strcpy(file_name_full, "./cuda_out/cur");

    char fn[100];
    snprintf(fn, sizeof(fn), "%d_level%d_iter%d_3.txt", frameID , paramid, iterative);

    strcat(file_name_full, fn);

    std::ofstream saveFile;

    saveFile.open(file_name_full);

    std::cout<<"stride is: "<< cols * rows << std::endl;
    saveFile << cols * rows << "\n";

    for(int i = 0; i < cols * rows; i++)
    {
        saveFile << curv_out[i].x << " " << curv_out[i].y <<" "<< curv_out[i].z<<"\n";
    }
    saveFile.close();
}

