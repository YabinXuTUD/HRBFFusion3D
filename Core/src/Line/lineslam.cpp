#include "lineslam.h"
#include "utils.h"

#ifdef QTPROJECT
#include <QtGui>
#include <QtOpenGL>
#include <gl/GLU.h>
#endif

#define UNDISTORT
#define WATCH_SINGLE
#define WATCH_MATCH
//#define EXTRACTLINE_USE_MAHDIST
#define USE_FIXED_FRAMERATE
//#define	SLAM_LBA

void SystemParameters::init()
{
    // ----- 2d-line -----
    line_segment_len_thresh		= 30;		// pixels, min lenght of image line segment to use

    // ----- 3d-line measurement ----
    ratio_of_collinear_pts		= 0.6;		// ratio, decide if a frameline has enough collinear pts
    pt2line_dist_extractline	= 0.02;		// meter, threshold pt to line distance when detect lines from pts
    pt2line_mahdist_extractline	= 1.5;		// NA,	  as above
    ransac_iters_extract_line	= 100;		// 1, max ransac iters when detect lines from pts
    num_cells_lineseg_range		= 10;		// 1,
    ratio_support_pts_on_line	= 0.7;		// ratio, when verifying a detected 3d line
    line3d_length_thresh		= 0.02;		// in meter, ignore too short 3d line segments

    // ----- camera model -----
    stdev_sample_pt_imgline		= 1;		// in pixel, std dev of sample point from an image line
    depth_stdev_coeff_c1		= 0.00273;	// c1,c2,c3: coefficients of depth noise quadratic function
    depth_stdev_coeff_c2		= 0.00074;
    depth_stdev_coeff_c3		= -0.00058;

    // ----- key frame -----
    num_raw_frame_skip			= 1;
    window_length_keyframe		= 10;
    num_2dlinematch_keyframe	= 30;		// detect keyframe, minmum number of 2d line matches left
    num_3dlinematch_keyframe	= 40;

    // ----- relative motion -----
    pt2line3d_dist_relmotion	= 0.05;		// in meter,
    line3d_angle_relmotion		= 10;
    fast_motion					= 1;
    inlier_ratio_constvel		= 0.4;
    dark_ligthing				= false;
    max_img_brightness			= 0;

    // ----- lba -----   //for line matching
    num_pos_lba	= 5;
    num_frm_lba	= 7;
    g2o_BA_use_kernel			= true;
    g2o_BA_kernel_delta			= 10;

    // ----- loop closing -----
    loopclose_interval			= 50;  // frames, check loop closure
    loopclose_min_3dmatch		= 30;  // min_num for 3d line matches between two frames

    // ----- lsd setting -----
    lsd_angle_th = 22.5;						// default: 22.5 deg
    lsd_density_th = 0.7;					    // default: 0.7
}

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

extern cv::Mat K, distCoeffs;
extern SystemParameters sysPara;

namespace Line3D{

// void Map3d::slam()
// {
// 	for(int pathIdx=0; pathIdx < paths.size(); ++pathIdx) {
// 		datapath = paths[pathIdx];
//         frames.clear();
// 		keyframeIdx.clear();
// 		lmklines.clear();
// 		// get synced file list
// 		vector<vector<string> > list;
// 		string syncname = datapath+"syncidx.txt";
// 		cout<<syncname<<endl;
// 		ifstream fsync(syncname.c_str());
// 		int i=0;
// 		if (fsync.is_open())  {
// 			string tmp;
// 			vector<string> step;
// 			while(fsync>>tmp)	{
// 				step.push_back(tmp);
// 				if(i%4 == 3) {
// 					list.push_back(step);
// 					step.clear();
// 				}
// 				++i;
// 			}
// 		}
// 		fsync.close();

// 		// write back for matlab use
// 		string ofname = datapath+"syncidx_m.txt";
// 		ofstream of(ofname.c_str());
// 		for(int i=0; i<list.size();++i){
// 			of<<list[i][0]<<'\t'<<list[i][2]<<endl;

// 		}
// 		of.close();

// 		vector<FrameLine> prevLines;
// 		int minMatchNum = sysPara.num_2dlinematch_keyframe;
// 		int minMatch3dNum = sysPara.num_3dlinematch_keyframe;
// 		int numFrameSkip = sysPara.num_raw_frame_skip;
// 		cout<<list.size()<<endl;
// 		for (int i=0; i<list.size(); ++i) {
// 			MyTimer t; 	t.start();
// 			// read rgb-d images and initialize Frames		
// 			if(frames.size()<i+1) { //
// 				Frame frm (datapath+list[i*numFrameSkip][1], datapath+list[i*numFrameSkip][3], K, distCoeffs); // 5M
// 				frm.id = i;
// 				frm.timestamp = atof(list[i*numFrameSkip][2].c_str()); // use depth image timestamp, original	
// 			//	frm.timestamp = atof(list[i*numFrameSkip][0].c_str()); // use rgb image timestamp
// 				frames.push_back(frm);
// 			}
// 			if(i==0) {
// 				frames[i].isKeyFrame = true;
// 				keyframeIdx.push_back(i);
// 				frames[i].R = cv::Mat::eye(3,3,CV_64F);
// 				frames[i].t = cv::Mat::zeros(3,1,CV_64F);
// 				vel = cv::Mat::zeros(3,1,CV_64F);
// 			}
// 			else
// 				frames[i].isKeyFrame = false;

// 			//compute gt motion
// 			cv::Mat gtR, gtt;
// 			if (compute_motion_given_ptpair_file (datapath+"ptpair/"+list[i*numFrameSkip][0]+".txt", frames[i].depth, gtR, gtt))
// 			{
// 				string fnam = datapath+"ptpair/relmot_"+list[i*numFrameSkip][0]+".txt";
// 				ofstream of(fnam.c_str());
// 				of<<gtR.at<double>(0,0)<<'\t'<<gtR.at<double>(0,1)<<'\t'<<gtR.at<double>(0,2)<<'\n';
// 				of<<gtR.at<double>(1,0)<<'\t'<<gtR.at<double>(1,1)<<'\t'<<gtR.at<double>(1,2)<<'\n';
// 				of<<gtR.at<double>(2,0)<<'\t'<<gtR.at<double>(2,1)<<'\t'<<gtR.at<double>(2,2)<<'\n';
// 				of<<gtt.at<double>(0)<<'\t'<<gtt.at<double>(1)<<'\t'<<gtt.at<double>(2)<<'\n';
// 				of.close();
// 			}


// 			if(i>0) {
// 				if(frames[i-1].isKeyFrame) {
// 					prevLines = frames[i-1].lines;				
// 				}

// 				int prevkf = keyframeIdx.back();
// 				vector<vector<int> > match;	
// 				bool trackline = true;
// 				int match3d = 0;
// #ifdef USE_FIXED_FRAMERATE
// 				if(i-prevkf > sysPara.window_length_keyframe)
// 					trackline = false;			
// #endif		
//                 //parameters depend on the bright
// 				double imbright = ave_img_bright(frames[i].gray);
//                 cout << "ave_img_bright: " << imbright <<"\n";
// 				if(sysPara.max_img_brightness < imbright)
// 					sysPara.max_img_brightness = imbright;
// 				//cout<<"light = "<<imbright<<endl;
// 				if ( imbright/sysPara.max_img_brightness < 0.3 || imbright < 10) {
// 					sysPara.dark_ligthing = true;
// 					sysPara.pt2line3d_dist_relmotion	= 0.025;		// in meter, 
// 					sysPara.line3d_angle_relmotion		= 5;
// 					sysPara.fast_motion = 0;

// 				} else {
// 					sysPara.dark_ligthing = false;
// 					sysPara.pt2line3d_dist_relmotion	= 0.05;		// in meter, 
// 					sysPara.line3d_angle_relmotion		= 10;
// 					sysPara.fast_motion = 1;
// 				}
// 				if(trackline) {
//                     //line matching
// 					trackLine(prevLines, frames[i].lines, match);
// 					// ---- count how many 3d matches in present frame ---- 
// 					for(int j=0; j<match.size(); ++j) {
// 						FrameLine a;
// 						if(prevkf == i-1)
// 							a = prevLines[match[j][0]];
// 						else
// 							a = frames[prevkf].lines[prevLines[match[j][0]].lid_prvKfrm];
// 						FrameLine b = frames[i].lines[match[j][1]];
// 						if (a.haveDepth && b.haveDepth) {
// 							match3d++;
// 						}
// 					}			
					
// 				}
// 				// ---- detect key frames ----	
// 				if( i-keyframeIdx.back() > 1 && 
// 					(match.size() < minMatchNum || match3d<minMatch3dNum
// 					|| i-keyframeIdx.back() > sysPara.window_length_keyframe )) {
						
// 						cout<<"keyframe "<<i-1<<"\t"<<frames[i-1].lines.size()<<"\t";
						
// #ifdef WATCH_MATCH
// 						cv::Mat canv1 = frames[prevkf].rgb.clone(), canv2 = frames[i-1].rgb.clone();							
// 						for(int j=0; j<prevLines.size(); ++j) {
// 							FrameLine a = frames[prevkf].lines[prevLines[j].lid_prvKfrm];
// 							FrameLine b = prevLines[j];
// 							cv::Scalar color(rand()%200,rand()%200,rand()%200,0);
// 							if(a.haveDepth && b.haveDepth) {
// 								cv::line(canv1, a.p, a.q, color, 2);
// 								cv::line(canv2, b.p, b.q, color, 2);								
// 							}
// 						}
						
// #endif
						
// 						// ---- compute relative pose between last keyframe and frame i-1 (new keyframe)
// 						vector<RandomLine3d> kfl_a, kfl_b;
// 						vector<int> pair3didx;
// 						for(int j=0; j<prevLines.size(); ++j) {
// 							FrameLine& a = frames[prevkf].lines[prevLines[j].lid_prvKfrm];
// 							FrameLine& b = frames[i-1].lines[prevLines[j].lid];					
// 							if (a.haveDepth && b.haveDepth) {
// 								if(a.line3d.covA.rows < 3) {// line not optimized yet									
//                                     MLEstimateLine3d_compact (a.line3d, 100);
// 								}
//                                 if(b.line3d.covA.rows < 3) {// line not optimized yet
// 									MLEstimateLine3d_compact (b.line3d, 100);									
// #ifdef SLAM_LBA
// 									frames[i-1].lines[b.lid] = b;
// #endif
// 								}
// 								kfl_a.push_back(a.line3d);
// 								kfl_b.push_back(b.line3d);
// 								pair3didx.push_back(j);
// 							}
// 						}
// 						cv::Mat R,t;
// 						vector<int> inliers = computeRelativeMotion_Ransac (kfl_a, kfl_b, R,t);	
						
// 						double inratio = 0;
// 						if(kfl_a.size()>0)
// 							inratio = inliers.size()/(double)kfl_a.size();
// 						cout<<kfl_a.size()<<"\t"<<inliers.size()<<"\t"<<inratio<<endl;
// #ifdef WATCH_MATCH
// 						putText(canv2, num2str(inliers.size())+" / "+num2str(kfl_a.size()), cv::Point(10,50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,200,0), 2);
// 						showImage("im1", &canv1);
// 						showImage("im2", &canv2);
// 					//	cv::imwrite("tmpimg/"+num2str(i-1)+"_a.jpg",canv1);
// 					//	cv::imwrite("tmpimg/"+num2str(i-1)+"_b.jpg",canv1);
// 						cv::waitKey(5);
// #endif

// 						if ( inliers.size() < 3 || (inliers.size() < 15 && inratio < sysPara.inlier_ratio_constvel) 
// 							||(cv::norm(vel) > 0.1  && cv::norm(t) > 8 * cv::norm(vel*(frames[i-1].timestamp - frames[prevkf].timestamp)))
// 							) {
// 								cout<<"::::::::constant velocity used :::::::: "<< inratio <<", "<<inliers.size()<<endl;								
// 								frames[i-1].R = frames[prevkf].R;
// 								cv::Mat pos0 = -frames[prevkf].R.t()*frames[prevkf].t;
// 								cv::Mat pos1 = pos0 + vel*(frames[i-1].timestamp - frames[prevkf].timestamp);
// 								frames[i-1].t = -frames[i-1].R*pos1;
// 								useConstVel = true;
// 						} else {
// 							frames[i-1].R = R*frames[prevkf].R;
// 							frames[i-1].t = R*frames[prevkf].t + t;
// 							//--- compute velocity ---
// 							int old = keyframeIdx[max(0,(int)keyframeIdx.size()-5)];
// 							double dt = frames[i-1].timestamp - frames[old].timestamp; // sec
// 							vel = (frames[old].R.t()*frames[old].t - frames[i-1].R.t()*frames[i-1].t)/dt;// in WCS, m/s							
// 							useConstVel = false;
// 						}
// 						keyframeIdx.push_back(i-1);
// //                        write_linepairs_tofile(kfl_a, kfl_b, "linepair_"+num2str(keyframeIdx.size()-1)+".txt", frames[i-1].timestamp);
// 						frames[i-1].isKeyFrame = true;
							
						
// #ifdef SLAM_LBA
// 						if(useConstVel)
// 							inliers.clear();
// 						//--- setup/update landmark lines ---
// 						for(int ii=0; ii<inliers.size();++ii) {
// 							int j = pair3didx[inliers[ii]];
// 					//	for(int j=0; j<prevLines.size(); ++j) {
// 							FrameLine& a = frames[prevkf].lines[prevLines[j].lid_prvKfrm];
// 							FrameLine& b = frames[i-1].lines[prevLines[j].lid];					
// 							if (a.haveDepth && b.haveDepth) {  // only consider 3d observations
// 								if(a.gid <0) { // set up new landmark							
// 									LmkLine ll;
// 									ll.A = mat2cvpt3d(
// 										frames[prevkf].R.t()*(cvpt2mat(a.line3d.A,0) - frames[prevkf].t));
// 									ll.B = mat2cvpt3d(
// 										frames[prevkf].R.t()*(cvpt2mat(a.line3d.B,0) - frames[prevkf].t));
// 									ll.gid = lmklines.size();
// 									vector<int> tmp;
// 									tmp.push_back(frames[prevkf].id);
// 									tmp.push_back(a.lid);
// 									ll.frmId_lnLid.push_back(tmp);
// 									tmp.clear();
// 									tmp.push_back(i-1);
// 									tmp.push_back(b.lid);
// 									ll.frmId_lnLid.push_back(tmp);
// 									a.gid = ll.gid;
// 									b.gid = ll.gid;
// 									lmklines.push_back(ll);
// 								} else { // update exsiting 3d landmark lines
// 									b.gid = a.gid;
// 									vector<int> tmp;
// 									tmp.push_back(i-1);
// 									tmp.push_back(b.lid);
// 									lmklines[a.gid].frmId_lnLid.push_back(tmp);
// 								}
// 							}
// 						}	
	
// 						lba_g2o(sysPara.num_pos_lba,sysPara.num_frm_lba);
// 						write2file (*this, "_lba");
// 						loopclose();
// #else	
// 						//error here
// 						write2file (*this, "");
// 						frames[prevkf].lines.clear();  // cost lot of mem
// #endif //SLAM_LBA
// 						i = i - 1;
// 						continue;
// 				}

// 					for(int j=i-sysPara.num_frm_lba; j<=i-2; ++j) {
// 						if (j < 0) continue; 
// 						frames[j].clear();
// 					}
				
// 				// display
// 				/*		for(int j=0; j<match.size(); ++j) {
// 				cv::Scalar color(0,0,255,0);
// 				cv::Mat canv1 = frames[i-1].rgb.clone(), canv2 = frames[i].rgb.clone();
// 				cv::line(canv1, prevLines[match[j][0]].p, prevLines[match[j][0]].q,
// 				color,3);
// 				cv::line(canv2, frames[i].lines[match[j][1]].p, frames[i].lines[match[j][1]].q,
// 				color, 2);
// 				showImage("im1", &canv1);
// 				showImage("im2", &canv2);
// 				cv::waitKey();
// 				}
// 				*/
// 				vector<FrameLine> tmpPrevLines;
// 				for(int j=0; j<match.size(); ++j)  {
// 					if(frames[i-1].isKeyFrame) {
// 						frames[i].lines[match[j][1]].lid_prvKfrm = match[j][0];
// 					} else {
// 						frames[i].lines[match[j][1]].lid_prvKfrm = prevLines[match[j][0]].lid_prvKfrm;
// 					}
// 					tmpPrevLines.push_back(frames[i].lines[match[j][1]]);
// 				}
// 				prevLines = tmpPrevLines;	

// 			}	
// 		//	t.end(); cout<<"per frame "<<t.time_ms<<" ms"<<endl;
// 		}
// 	}
// #ifdef SLAM_LBA
// 	write2file (*this, "_lba");
// #endif
// }

Frame::Frame (string rgbName, string depName, cv::Mat K, cv::Mat dc, float sf, int fID): scaleFactor(sf), id(fID),
    drawLinesProgram(loadProgramFromFile("draw_lines.vert", "draw_lines.frag"))
{
//	MyTimer t;	t.start();
    cv::Mat oriRgb = cv::imread(rgbName, cv::IMREAD_COLOR);

    //undistort RGB image if needed
	rgb = oriRgb.clone();
#ifdef UNDISTORT            // 10 ms
	if(cv::norm(dc)>1e-5)	
		cv::undistort(oriRgb, rgb, K, dc);
#endif
	oriRgb.release();

    //convert it to gray image
	if(rgb.channels() ==3)
        cv::cvtColor(rgb, gray, CV_RGB2GRAY);
	else
		gray = rgb;

    //equalize image to see if it can improve the performance of LSD detection
//	cv::equalizeHist(gray, gray);

    oriDepth = cv::imread(depName, CV_LOAD_IMAGE_ANYDEPTH);
	oriDepth.convertTo(depth, CV_64F);

    //minimun length of used 2D lines
	lineLenThresh = sysPara.line_segment_len_thresh;

    //detect 2D lines
    detectFrameLines();
    //extract 3D lines
	extractLineDepth();

	int n3 = 0;
	for(int i=0; i<lines.size(); ++i) {		
		if(lines[i].haveDepth) {
			n3++;
		}
	}
	oriDepth.release();
	oriRgb.release();

    glGenBuffers(1, &vbo); //for line visualization

#ifdef WATCH_SINGLE
    cv::Mat canv = rgb.clone();
    std::vector<Eigen::Vector3i> colorSet = pseudocolor(lines.size());
    cv::Scalar color;
    int x=0;
    for(int i=0; i<lines.size(); ++i) {
        double w=2;
        color = cv::Scalar(colorSet[i][0], colorSet[i][1], colorSet[i][2], 0);
        if(lines[i].haveDepth) {
            color = cv::Scalar(colorSet[i][0], colorSet[i][1], colorSet[i][2], 0);
            w=2;
            x++;
        }
        cv::line(canv, lines[i].p, lines[i].q, color, w);
    }
    if(GlobalStateParam::get().lineDetectionMergeDetected2DLines)
        cv::imwrite("lines/after_" + std::to_string(id) + ".png", canv);
    else
        cv::imwrite("lines/before_" + std::to_string(id) + ".png", canv);
#endif
//    t.end(); cout<<"building a frame: "<<t.time_ms <<endl;
}

void Frame::detectFrameLines()
{
    static int cn=0; static double at1=0, at2=0; MyTimer t1,t2;
	IplImage pImg = gray;
	ntuple_list  lsdOut;	
    t1.start();
    lsdOut = callLsd(&pImg);
    t1.end();
    std::cout <<"time of lsd detection: "<< t1.time_ms << std::endl;
	int dim = lsdOut->dim;

    //the output of LSD is represented by 2 endpoints
	double a,b,c,d;
	lines.reserve(lsdOut->size);
    for(int i=0; i<lsdOut->size; i++) {         // store LSD output to lineSegments
		a = lsdOut->values[i*dim];
		b = lsdOut->values[i*dim+1];
		c = lsdOut->values[i*dim+2];
		d = lsdOut->values[i*dim+3];
		if ( sqrt((a-c)*(a-c)+(b-d)*(b-d)) > lineLenThresh) {
            lines.push_back(FrameLine(cv::Point2d(a,b), cv::Point2d(c,d), id));
		}
	}

    //show detected 2D lines
    if(GlobalStateParam::get().lineDetectionShowDetected2DLines)
    {
        unsigned int w = pImg.width;
        unsigned int h = pImg.height;
        IplImage *canvas = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 3);
        for(int i=0;i<lines.size();i++){
            cvLine(&pImg, lines[i].p, lines[i].q, cvScalar(255, 255, 255, 0), 1.5, 8, 0);
        }
        cvNamedWindow("LSD Result",1);
        cvShowImage("LSD Result",&pImg);
        cvWaitKey(3000);
        cvDestroyWindow("LSD Result");
        cvReleaseImage(&canvas);
    }

    //2D line merging for fragmetary lines (to be improved)
    if(GlobalStateParam::get().lineDetectionMergeDetected2DLines)
    {
        mergeColinearLines();
    }

	// assign id and equation
	for (int i=0; i<lines.size(); ++i){
		lines[i].lid = i;
		lines[i].complineEq2d();
	}

	// compute msld line descriptors
	cv::Mat xGradImg, yGradImg;
	int ddepth = CV_64F;	
	cv::Sobel(gray, xGradImg, ddepth, 1, 0, 5); // Gradient X
	cv::Sobel(gray, yGradImg, ddepth, 0, 1, 5); // Gradient Y
	#pragma omp  parallel for
    for (int i=0; i<lines.size(); ++i){
		computeMSLD(lines[i], &xGradImg, &yGradImg);
	}
}

void Frame::mergeColinearLines()
{
    std::vector<std::vector<int>>  matches;
    for(int i = 0; i < lines.size(); i++){
        cv::Point2d p1 = lines[i].p;
        cv::Point2d q1 = lines[i].q;

        vector<int> onePairIdx;
        onePairIdx.push_back(i);
        for(int j = i + 1; j < lines.size(); j++){
             cv::Point2d p2 = lines[j].p;
             cv::Point2d q2 = lines[j].q;

             //minimum distance of end points
             double min_dist_1 = min(cv::norm(p1 - p2), cv::norm(q1 - q2));
             double min_dist_2 = min(cv::norm(p1 - q2), cv::norm(q1 - p2));
             double min_dist = min(min_dist_1, min_dist_2);

             //distance between the midpoint of one segment to the other segment
             cv::Point2d mid_1 = (p1 + q1) / 2;
             cv::Point2d q2_p2_normalized = (q2 - p2) / cv::norm(q2 - p2);
             double mid_dist_1 = cv::norm((mid_1 - p2).cross(q2_p2_normalized));
             cv::Point2d mid_2 = (p2 + q2) / 2;
             cv::Point2d q1_p1_normalized = (q1 - p1) / cv::norm(q1 - p1);
             double mid_dist_2 = cv::norm((mid_2 - p1).cross(q1_p1_normalized));

             //overlap should not be larger than a certain threshold (remain to be updated)
             double over_lap;
             cv::Point2d v_pq = q1 - p1;
             double proj_p2 = (p2 - p1).dot(v_pq) / cv::norm(v_pq);
             double proj_q2 = (q2 - p1).dot(v_pq) / cv::norm(v_pq);
             if(proj_p2 > proj_q2)
             {
                 double tmp = proj_p2;
                 proj_p2 = proj_q2;
                 proj_q2 = tmp;
             }
             if(proj_q2 < 0)
                 over_lap = 0;
             if(proj_p2 < 0 && proj_q2 > 0 && proj_q2 < cv::norm(v_pq))
                 over_lap = proj_q2;
             if(proj_p2 < 0 && proj_q2 > cv::norm(v_pq))
                 over_lap = cv::norm(v_pq);
             if(proj_p2 > 0 && proj_p2 < cv::norm(v_pq) && proj_q2 < cv::norm(v_pq))
                 over_lap = cv::norm(proj_q2 - proj_p2);
             if(proj_p2 > 0 && proj_p2 < cv::norm(v_pq) && proj_q2 > cv::norm(v_pq))
                 over_lap = cv::norm(v_pq) - proj_p2;
             if(proj_p2 > cv::norm(v_pq))
                 over_lap = 0;
             double overlap_radio = min(over_lap / cv::norm(q1 - p1), over_lap / cv::norm(q2 - p2));

             if(min_dist < 10 && mid_dist_1 < 5 && mid_dist_2 < 5 && overlap_radio < 0.1)
             {
                 lines[i].haveMatched = true;
                 lines[j].haveMatched = true;
                 onePairIdx.push_back(j);
//                     std::cout << "matched: " << j << " ";
             }
        }
        //std::cout << "\n";
        if(onePairIdx.size() > 1){
            bool add_new = true;
            for(int k = 0; k < onePairIdx.size(); k++){
                for(int n = 0; n < matches.size(); n++){
                    vector<int>::iterator it = find(matches[n].begin(), matches[n].end(), onePairIdx[k]);
                    if(it!=matches[n].end()){
                        matches[n].insert(matches[n].end(), onePairIdx.begin(), onePairIdx.end());
                        //delete duplicated elements
                        sort(matches[n].begin(), matches[n].end());
                        matches[n].erase(unique(matches[n].begin(), matches[n].end()), matches[n].end());
                        add_new = false;
                        break;
                    }
                }
            }
            if(add_new)
                matches.push_back(onePairIdx);
        }
    }
    //std::cout << "find " << matches.size() << " 2D line features to be merged!\n";
    //merge matched 2D line segments
    vector<FrameLine> lines_merged;
    for(int i = 0; i < matches.size(); i++)
    {
        double sum_dist = 0;
        vector<cv::Point2d> points;
        cv::Point2d p_merged, q_merged;
        for(int j = 0; j < matches[i].size(); j++)
        {
            points.push_back(lines[matches[i][j]].p);
            points.push_back(lines[matches[i][j]].q);
            sum_dist = sum_dist + cv::norm(lines[matches[i][j]].p - lines[matches[i][j]].q);
            //std::cout << matches[i][j] << " ";
        }
        //std::cout << std::endl;
        double min_dist = 0;
        for(int m = 0; m < points.size(); m++)
        {
            for(int n = m + 1; n < points.size(); n++)
            {
                double d = cv::norm(points[m] - points[n]);
                if(d > min_dist)
                {
                    p_merged = points[m];
                    q_merged = points[n];
                    //std::cout << "end points of merged lines: " << m << " " << n << "\n";
                    min_dist = d;
                }
            }
        }
        lines_merged.push_back(FrameLine(p_merged, q_merged, id));
      }
      for(int i = 0; i < lines.size(); i++)
      {
        //push back unmatched lines
        if(!lines[i].haveMatched)
          lines_merged.push_back(lines[i]);
      }
      swap(lines, lines_merged);
}

void Frame::extractLineDepth()
	// extract the 3d info of an frame line if availabe from the depth image
	// input: depth, lines
	// output: lines with 3d info
{	
	int n_3dln = 0;

//    ofstream file;
//    file.open("3Dline_samples.txt");
//    std::vector<Eigen::Vector3i> colorSet = pseudocolor(lines.size());

    #pragma omp  parallel for
    for(int i=0; i<lines.size(); ++i)	{       // each line
		double len = cv::norm(lines[i].p - lines[i].q);		
		vector<cv::Point3d> pts3d;
		// iterate through a line
        double numSmp = len;   //(double) min((int)len, 200);     // number of line points sampled, related to accuracy and efficiency
        //3D point set
        pts3d.reserve(numSmp);
		for(int j=0; j<=numSmp; ++j) {
			// use nearest neighbor to querry depth value
			// assuming position (0,0) is the top-left corner of image, then the
			// top-left pixel's center would be (0.5,0.5)
			cv::Point2d pt = lines[i].p * (1-j/numSmp) + lines[i].q * (j/numSmp);
			if(pt.x<0 || pt.y<0 || pt.x >= depth.cols || pt.y >= depth.rows ) continue;
			int row, col; // nearest pixel for pt
            if((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y)) {    // boundary issue
                col = max(int(pt.x-1), 0);
                row = max(int(pt.y-1), 0);
			} else {
				col = int(pt.x);
				row = int(pt.y);
			}

            double zval = -1;
            if(depth.at<double>(row,col) < EPS) {     // no depth info

            } else {
                zval = depth.at<double>(row,col) * static_cast<double>(scaleFactor);

                if(GlobalStateParam::get().lineDetectionPruneOccludedPoints){
                    //search on neighbors to check if it is occluded edge point
                    int r = 1;
                    int D = r * 2 + 1;
                    int t_col = min(col - D / 2 + D, int(depth.cols));
                    int t_row = min(row - D / 2 + D, int(depth.rows));
                    double dist_max_signed;
                    double dist_abs = 0;
                    double zval_max;
                    int col_max, row_max;
                    for(int c_row = max(row - D / 2, 0); c_row < t_row; ++c_row){
                        for(int c_col = max(col - D / 2, 0); c_col < t_col; ++c_col){
                            double c_zval = depth.at<double>(c_row,c_col) * static_cast<double>(scaleFactor);
                            if(c_zval < EPS)
                                continue;
                            double dist = zval - c_zval;
                            if(fabs(dist) > dist_abs)
                            {
                                dist_max_signed = dist;
                                dist_abs = fabs(dist);
                                zval_max = c_zval;
                                col_max = c_col;
                                row_max = c_row;
                            }
                        }
                    }
                    if(dist_abs > 0.02 && dist_max_signed > 0)
                    {
                        zval = zval_max;
                        pt.x = col_max;
                        pt.y = row_max;
                    }
                }
            }
#ifdef DEBUGGING
			/*		cout<<"line "<<i<<": "<<pt<<": "<<zval <<", "<<oriDepth.at<ushort>(row,col)<<" m"<<endl;
			cv::Mat canv = rgb.clone(), canv2 = depth.clone()/65500;
			cv::Scalar color = cv::Scalar(rand()%255,rand()%255,rand()%255,0);
			cv::line(canv, lines[i].p, lines[i].q,color, 2);
			cv::circle(canv, pt, 2, color, 3);
			cv::line(canv2, lines[i].p, lines[i].q,color, 1);
			cv::circle(canv2, pt, 2, color, 2);
			showImage("lines", &canv);
			showImage("depth", &canv2);
			cv::waitKey();
			*/	         
#endif
			// export 3d points to file
            // sample_3D for color
            if (zval > 0) {
				cv::Point2d xy3d = mat2cvpt(K.inv()*cvpt2mat(pt))*zval;	
				pts3d.push_back(cv::Point3d(xy3d.x, xy3d.y, zval));
                //file << xy3d.x << " " << xy3d.y << " " << zval << " " << colorSet[i][0] << " " << colorSet[i][1] << " " << colorSet[i][2] << "\n";
			}
		}

        if (pts3d.size() < max(10.0,len*sysPara.ratio_of_collinear_pts))
            continue;

		RandomLine3d tmpLine;		

#ifdef EXTRACTLINE_USE_MAHDIST
		vector<RandomPoint3d> rndpts3d;
		rndpts3d.reserve(pts3d.size());
		// compute uncertainty of 3d points
		for(int j=0; j<pts3d.size();++j) {
			rndpts3d.push_back(compPt3dCov(pts3d[j], K));
		}
		// using ransac to extract a 3d line from 3d pts
		tmpLine = extract3dline_mahdist(rndpts3d);
#else
        //Euclidean Distance
		tmpLine = extract3dline(pts3d);
#endif
        if( tmpLine.pts.size()/len > sysPara.ratio_of_collinear_pts	&&
            cv::norm(tmpLine.A - tmpLine.B) > sysPara.line3d_length_thresh) {
                //MLEstimateLine3d (tmpLine, 100);
				lines[i].haveDepth = true;
				lines[i].line3d = tmpLine;
				n_3dln++;
        }
    }

    //file.close();
    //std::cout << "extrated " << n_3dln << " 3D lines\n";
}

void Frame::update3DLinePose(cv::Mat Rot, cv::Mat trans)
{
    Rot.copyTo(R);
    trans.copyTo(t);

    //update endline points and inlier poses
    for(int i = 0; i < lines.size(); i++)
    {
        if(lines[i].haveDepth)
        {
           lines[i].line3d.A = mat2cvpt3d(R * cvpt2mat(lines[i].line3d.A, 0) + trans);
           lines[i].line3d.B = mat2cvpt3d(R * cvpt2mat(lines[i].line3d.B, 0) + trans);

           for(int j = 0; j < lines[i].line3d.pts.size(); j++)
           {
                lines[i].line3d.pts[j].pos = mat2cvpt3d(R * cvpt2mat(lines[i].line3d.pts[j].pos, 0) + trans);
                lines[i].line3d.pts[j].xyz[0] = lines[i].line3d.pts[j].pos.x;
                lines[i].line3d.pts[j].xyz[1] = lines[i].line3d.pts[j].pos.y;
                lines[i].line3d.pts[j].xyz[1] = lines[i].line3d.pts[j].pos.z;
           }
        }
    }
}

void Frame::drawLines(pangolin::OpenGlMatrix mvp, const Eigen::Matrix4f & pose)
{
    std::vector<float> lb;
    int N_lines = lines.size();
    lb.reserve(6 * N_lines);    //2 point (3 floats) for a line;
    for(int i = 0; i < N_lines; i++){
        if(lines[i].haveDepth)
        {
            lb.push_back(static_cast<float>(lines[i].line3d.A.x));
            lb.push_back(static_cast<float>(lines[i].line3d.A.y));
            lb.push_back(static_cast<float>(lines[i].line3d.A.z));

            lb.push_back(static_cast<float>(lines[i].line3d.B.x));
            lb.push_back(static_cast<float>(lines[i].line3d.B.y));
            lb.push_back(static_cast<float>(lines[i].line3d.B.z));
        }
    }
    //draw lines
    drawLinesProgram->Bind();
    drawLinesProgram->setUniform(Uniform("MVP", mvp));
    drawLinesProgram->setUniform(Uniform("pose", pose));

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, N_lines * 6 * sizeof(float), &lb[0], GL_STREAM_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), 0);
    glDrawArrays(GL_LINES, 0, N_lines);

    glDisableVertexAttribArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    drawLinesProgram->Unbind();

}

void Frame::write3DLines2file(string fname)
{
    fname = fname + "3d_" + num2str(id) + ".txt";
    ofstream file(fname.c_str());

    float num_seg = 2000;
    std::vector<Eigen::Vector3i> colors = pseudocolor(lines.size());
    std::cout << "number of 3D lines: " << lines.size() << "\n";
    for(int i = 0; i< lines.size(); ++i) {
        if(lines[i].haveDepth) {
            //sample the line to points;
            cv::Point3d p;
            cv::Point3d d = lines[i].line3d.B -  lines[i].line3d.A;
            for(int j = 0; j < num_seg; ++j){
                p = lines[i].line3d.A + (j / num_seg) * d;
                file<<p.x <<" "<<p.y <<" "<<p.z <<" "<<colors[i](0) <<" "<<colors[i](1) <<" "<<colors[i](2) <<"\n";
            }
        }
    }
    file.close();
}


FrameLine::FrameLine(cv::Point2d p_, cv::Point2d q_, int fId): frameId(fId)
{
	p = p_;
	q = q_;
	l = cvpt2mat(p).cross(cvpt2mat(q));
	haveDepth = false;
    haveMatched = false;
	gid = -1;
}

cv::Point2d FrameLine::getGradient(cv::Mat* xGradient, cv::Mat* yGradient)
{	
	cv::LineIterator iter(*xGradient, p, q, 8);
	double xSum=0, ySum=0;
	for (int i=0; i<iter.count; ++i, ++iter) {
		xSum += xGradient->at<double>(iter.pos());
		ySum += yGradient->at<double>(iter.pos());
	}
	double len = sqrt(xSum*xSum+ySum*ySum);
	return cv::Point2d(xSum/len, ySum/len);
}

/*void Frame::clear()
{
lines.clear();	
rgb.release();
gray.release();
depth.release();
oriDepth.release();
}*/

void Frame::clear()
{
	if(this->isKeyFrame) {
		rgb.release();
		gray.release();
		depth.release();
		oriDepth.release();
		for(int i=0; i<lines.size(); ++i) {
			if(lines[i].haveDepth && lines[i].line3d.covA.rows != 0) {
				lines[i].line3d.pts.clear();
			}
		/*	lines[i].line3d.covA.release();
			lines[i].line3d.covB.release();
			lines[i].line3d.rndA.cov.release();
			lines[i].line3d.rndB.cov.release();
			lines[i].line3d.rndA.U.release();
			lines[i].line3d.rndB.U.release();
			lines[i].line3d.rndA.W.release();
			lines[i].line3d.rndB.W.release();
		*/}
	} else {
		lines.clear();	
		rgb.release();
		gray.release();
		depth.release();
		oriDepth.release();
	}
//	cout<<" <<<<<< frame "<<this->id << " cleared >>>>>>> \n ";
}

#ifdef SLAM_LBA
void Map3d::loopclose()
{
	
	// ======= 1. detect loop closure =======
	int numRcntFrm = sysPara.loopclose_interval; // no. of recent (raw) frames not considered for LC
	double thresLocDist = 0.5; // in meter, only consider camera located within this
	double thresOriAngl = 60 * PI/180; // in radian

	if (frames.back().id - lastLoopCloseFrame < numRcntFrm) 
		return;

	Frame& curKf = frames[keyframeIdx.back()];
	vector<PoseConstraint> pcs;
	for(int ikf = 0; ikf < keyframeIdx.size(); ++ikf) {
		Frame& kf = frames[keyframeIdx[ikf]];
		if(abs(kf.id - curKf.id) < numRcntFrm) continue; // don't consider recent frames
		if(rotAngle(kf.R.t()*curKf.R) > thresOriAngl) continue;
		if(cv::norm(-kf.R.t()*kf.t+curKf.R.t()*curKf.t) > thresLocDist) continue;
		// ===== start frame line matching =====
		vector<vector<int> > matches;
		cout<<"potential loop closure "<<curKf.id <<"<->"<<kf.id<<endl;		
		matchLine(curKf.lines, kf.lines, matches);		
		vector<RandomLine3d> l3a, l3b;
		vector<vector<int> > lmkGidPairs;
		for(int i =0; i<matches.size(); ++i) {
			FrameLine& a = curKf.lines[matches[i][0]];
			FrameLine& b = kf.lines[matches[i][1]];
			if (a.haveDepth && b.haveDepth) {
				if(a.line3d.covA.rows < 3) {// line not optimized yet
					MLEstimateLine3d_compact (a.line3d, 100);									
				}
				if(b.line3d.covA.rows < 3) { // line not optimized yet
					MLEstimateLine3d_compact (b.line3d, 100);					
				}
				l3a.push_back(a.line3d);
				l3b.push_back(b.line3d);
				vector<int> pair(2);
				pair[0] = b.gid; pair[1] = a.gid; lmkGidPairs.push_back(pair);
			}
		}
		
		if(l3a.size() < sysPara.loopclose_min_3dmatch) continue; 
		cv::Mat R, t;
		vector<int> inliers = computeRelativeMotion_Ransac (l3a,l3b, R,t);
		cout<<"found 3d matches "<<l3a.size()<<", after ransac "<<inliers.size()<<endl;
		if(inliers.size() < 10 || inliers.size()*2 < l3a.size()) {
			
		} else {
			cout << -R.t()*t<<endl;
			PoseConstraint pc;
			pc.from = keyframeIdx.size()-1;
			pc.to = ikf;
			pc.R = R.clone();
			pc.t = t.clone();
			pc.numMatches = inliers.size();
			pcs.push_back(pc);
			vector<vector<int> > goodLmkGidPairs;
			cv::Mat canv1 = curKf.rgb.clone(), canv2 = kf.rgb.clone();
			
			for(int i=0; i < inliers.size(); ++i) {				
				cv::Scalar color(rand()%200,rand()%200,rand()%200);
				cv::line(canv1, curKf.lines[matches[inliers[i]][0]].p, curKf.lines[matches[inliers[i]][0]].q, color,2);
				cv::putText(canv1, num2str(i), curKf.lines[matches[inliers[i]][0]].p, 1, 1, color);
				cv::line(canv2, kf.lines[matches[inliers[i]][1]].p, kf.lines[matches[inliers[i]][1]].q, color,2); 
				cv::putText(canv2, num2str(i), kf.lines[matches[inliers[i]][1]].p,1, 1, color);
				if(lmkGidPairs[inliers[i]][0] >=0 && lmkGidPairs[inliers[i]][1] >=0)
					goodLmkGidPairs.push_back(lmkGidPairs[inliers[i]]);
			}
		//	showImage("curkf", &canv1);
		//	showImage("prvkf", &canv2);
		//	cv::waitKey();
			correctAll(goodLmkGidPairs);
		//	correctPose(pcs);
			lastLoopCloseFrame = frames.back().id;
			break;
		}
		
	}
	// ======= 2. correct map/pose graph =======
	
}
#endif

#ifdef QTPROJECT
void SlamThread::run()
{	
	map.slam();
}

void Map3d::draw3d()
{
	float scale = 0.10;
	// plot first camera, small
	glLineWidth(1*scale);
	glBegin(GL_LINES);
	glColor3f(1,0,0); // x-axis
	glVertex3f(0,0,0);
	glVertex3f(1*scale,0,0);
	glColor3f(0,1,0);
	glVertex3f(0,0,0);
	glVertex3f(0,1*scale,0);
	glColor3f(0,0,1);// z axis
	glVertex3f(0,0,0);
	glVertex3f(0,0,1*scale);
	glEnd();

	cv::Mat xw = (cv::Mat_<double>(3,1)<< 0.5,0,0)*scale,
		yw = (cv::Mat_<double>(3,1)<< 0,0.5,0)*scale,
		zw = (cv::Mat_<double>(3,1)<< 0,0,0.5)*scale;

	for (int i=1; i<frames.size(); ++i) {
		if(!frames[i].isKeyFrame) continue;
		if(!(frames[i].R.dims==2)) continue; // handle in-process view 
		cv::Mat c = -frames[i].R.t()*frames[i].t;	
		cv::Mat x_ = frames[i].R.t() * (xw-frames[i].t),
			y_ = frames[i].R.t() * (yw-frames[i].t),
			z_ = frames[i].R.t() * (zw-frames[i].t);
		glBegin(GL_LINES);

		glColor3f(1,0,0);
		glVertex3f(c.at<double>(0),c.at<double>(1),c.at<double>(2));
		glVertex3f(x_.at<double>(0),x_.at<double>(1),x_.at<double>(2));
		glColor3f(0,1,0);
		glVertex3f(c.at<double>(0),c.at<double>(1),c.at<double>(2));
		glVertex3f(y_.at<double>(0),y_.at<double>(1),y_.at<double>(2));
		glColor3f(0,0,1);
		glVertex3f(c.at<double>(0),c.at<double>(1),c.at<double>(2));
		glVertex3f(z_.at<double>(0),z_.at<double>(1),z_.at<double>(2));
		glEnd();
	}

}

#endif
}
