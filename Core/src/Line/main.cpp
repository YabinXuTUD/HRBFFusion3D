#include "lineslam.h"
#include <iostream>
#include "utils.h"
#include "define.h"
#ifdef QTPROJECT
#include <QApplication>
#include <QDesktopWidget>
#include "window.h"
#endif

//-------- global variables --------
cv::Mat K, distCoeffs;
SystemParameters sysPara;

void SystemParameters::init()
{
	// ----- 2d-line -----
	line_segment_len_thresh		= 10;		// pixels, min lenght of image line segment to use

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
    lsd_density_th = 0.7;					   // default: 0.7

}


int main(int argc, char *argv[])
{

	MyTimer timer;
	timer.start();
	srand(1); // fix random seed for debugging
	sysPara.init();
	vector<string> paths;
    getConfigration  (paths, K, distCoeffs, 100) ;
	Map3d map(paths);

#ifdef QTPROJECT
	SlamThread compThread(map);
	compThread.start();
	QApplication app(argc, argv);
	Window win;
	win.resize(win.sizeHint());
	win.setScene(&map);

	// ------- plot ---------
	int desktopArea = QApplication::desktop()->width() *
		QApplication::desktop()->height();
	int widgetArea = win.width() * win.height();
	if (((float)widgetArea / (float)desktopArea) < 0.75f)
		win.show();
	else
		win.showMaximized();
	return app.exec();
#else
	map.slam();
	timer.end();
	std::cout<<"total time is "<<timer.time_s<<" sec."<<std::endl;
	return 0;
#endif
}
