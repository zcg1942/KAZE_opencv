//=============================================================================
//
// kaze_match.cpp
// Author: Pablo F. Alcantarilla
// Date: 11/12/2012
// Email: pablofdezalc@gmail.com
//
// KAZE Features Copyright 2014, Pablo F. Alcantarilla
// All Rights Reserved
// See LICENSE for the license information
//=============================================================================

/**
 * @file kaze_match.cpp
 * @brief Main program for matching two images with KAZE features
 * The two images can have different resolutions
 * @date Dec 11, 2014
 * @author Pablo F. Alcantarilla
 */

#include "KAZE.h"
#include<imgproc.hpp>
//#include <transpose.hpp>//转置操作,找不到
using namespace std;

/* ************************************************************************* */
// Some image matching options
const float MAX_H_ERROR = 2.50;	// Maximum error in pixels to accept an inlier
const float DRATIO = .80;		// NNDR Matching value
void PSNR_count(cv::Mat, cv::Mat);
/* ************************************************************************* */
/**
 * @brief This function parses the command line arguments for setting KAZE parameters
 * and image matching between two input images
 * @param options Structure that contains KAZE settings
 * @param img_path1 Name of the first input image
 * @param img_path2 Name of the second input image
 * @param homography_path Name of the file that contains a ground truth homography
 */
int parse_input_options(KAZEOptions& options, std::string& img_path1, std::string& img_path2,
                        std::string& homography_path, int argc, char *argv[]);

/* ************************************************************************* */
/** Main Function 																	 */
int main( int argc, char *argv[] ) {

  KAZEOptions options;
  cv::Mat img1, img1_32, img2, img2_32, img1_rgb, img2_rgb, img_com, img_r,img_xform;
  cv::Mat desc1, desc2, HG;
  string img_path1, img_path2, homography_path;
  float ratio = 0.0, rfactor = .90;
  vector<cv::KeyPoint> kpts1, kpts2;
  vector<vector<cv::DMatch> > dmatches;

  int nkpts1 = 0, nkpts2 = 0, nmatches = 0, ninliers = 0, noutliers = 0;

  // Variables for measuring computation times
  double t1 = 0.0, t2 = 0.0, tkaze = 0.0, tmatch = 0.0;

  // Parse the input command line options
  if (parse_input_options(options, img_path1, img_path2, homography_path, argc, argv))
    return -1;//出错就返回-1

  // Read the image, force to be grey scale
  img1 = cv::imread(img_path1, 0);

  if (img1.data == NULL) {
    cerr << "Error loading image: " << img_path1 << endl;
    return -1;
  }

  // Read the image, force to be grey scale
  img2 = cv::imread(img_path2, 0);//读进去的时候就是灰度图像

  if (img2.data == NULL) {
    cout << "Error loading image: " << img_path2 << endl;
    return -1;
  }

  // Convert the images to float
  img1.convertTo(img1_32, CV_32F, 1.0/255.0, 0);
  img2.convertTo(img2_32, CV_32F, 1.0/255.0, 0);

  // Color images for results visualization
  img1_rgb = cv::Mat(cv::Size(img1.cols,img1.rows), CV_8UC3);
  img2_rgb = cv::Mat(cv::Size(img2.cols,img2.rows), CV_8UC3);
  img_com = cv::Mat(cv::Size(img1.cols*2,img1.rows), CV_8UC3);//长度是图1的2倍
  //img_com = cv::Mat(MAX(img1.cols,img2.cols), img1.rows+img2.rows, CV_8UC3); //cvSize(MAX(img1->width, img2->width),
	 
  img_r = cv::Mat(cv::Size(img_com.cols*rfactor, img_com.rows*rfactor), CV_8UC3);//乘一个比例因子0.9
  img_xform = cv::Mat(cv::Size(img2.cols, img2.rows), CV_8UC3);

  // Read ground truth homography file
  bool use_ransac = false;
  if (read_homography(homography_path, HG) == false)//确保内存没有被写入
    use_ransac = true;//默认用RANSAC

  // Create the first KAZE object
  options.img_width = img1.cols;
  options.img_height = img1.rows;
  libKAZE::KAZE evolution1(options);

  // Create the second KAZE object
  options.img_width = img2.cols;
  options.img_height = img2.rows;
  libKAZE::KAZE evolution2(options);

  t1 = cv::getTickCount();

  // Create the nonlinear scale space
  // and perform feature detection and description for image 1
  evolution1.Create_Nonlinear_Scale_Space(img1_32);
  evolution1.Feature_Detection(kpts1);
  evolution1.Compute_Descriptors(kpts1, desc1);

  // Create the nonlinear scale space
  // and perform feature detection and description for image 2
  evolution2.Create_Nonlinear_Scale_Space(img2_32);
  evolution2.Feature_Detection(kpts2);
  evolution2.Compute_Descriptors(kpts2, desc2);

  t2 = cv::getTickCount();
  tkaze = 1000.0*(t2-t1) / cv::getTickFrequency();//时间，以毫秒为单位

  nkpts1 = kpts1.size();
  nkpts2 = kpts2.size();

  // Matching Descriptors!!
  vector<cv::Point2f> matches, inliers;
  cv::Ptr<cv::DescriptorMatcher> matcher_l2 = cv::DescriptorMatcher::create("BruteForce");//初始匹配对

  t1 = cv::getTickCount();

  matcher_l2->knnMatch(desc1, desc2, dmatches,2);//KNN Find k best matches for each query descriptor，k=2，找到距离最近的前两个点，好计算最近邻比
  //最近邻比
  matches2points_nndr(kpts1, kpts2, dmatches, matches, DRATIO);//matches是Point2f的点；dmatches是DMatch类型，最近邻前两个

  t2 = cv::getTickCount();
  tmatch = 1000.0*(t2-t1) / cv::getTickFrequency();

  // Compute Inliers!!
  cv::Mat H_warp= cv::Mat::zeros(3, 3, CV_32FC1);
  if (use_ransac == false)
    compute_inliers_homography(matches, inliers, HG, MAX_H_ERROR);//两点距离小于容错，就push_back进inline,这一步实际没有执行
  else
	  //将matches分别push back进points1,points2
    H_warp= compute_inliers_ransac(matches, inliers, MAX_H_ERROR, false);//默认用RANSAC,
  //函数改为return,得到66'B',ASCAll码,但是Mat不能直接输出，调试也只是ASCAll码
  //warpPerspective(img1_rgb, img_xform, H_warp, cv::Size(img_xform.cols, img_xform.rows));
  cout << "H_warp = " << endl << " " << H_warp<< endl << endl;//打印Mat


  // Compute the inliers statistics
  nmatches = matches.size()/2;//计算匹配对对数,matches包含最近的两个配对
  ninliers = inliers.size()/2;
  noutliers = nmatches - ninliers;
  ratio = 100.0*((float) ninliers / (float) nmatches);

  //计算均方根误差RMSE 丁南南
  double x1, y1, x11, y11, x2, y2,x22,y22, distance, RMSE;
  /*vector<cv::Point2f> inliers_Hwarp;
  warpPerspective(inliers, img_xform, H_warp, img_xform.size());*/
  double h11 = 0.0, h12 = 0.0, h13 = 0.0;
  double h21 = 0.0, h22 = 0.0, h23 = 0.0;
  double h31 = 0.0, h32 = 0.0, h33 = 0.0;
  double sum = 0.0;
  //cv::Mat H1 = cv::Mat::zeros(3, 3, CV_32FC1);
  //cv::transpose(H_warp, H_warp);转置
  h11 = H_warp.at<double>(0, 0);
  h12 = H_warp.at<double>(0, 1);
  h13 = H_warp.at<double>(0, 2);
  h21 = H_warp.at<double>(1, 0);
  h22 = H_warp.at<double>(1, 1);
  h23 = H_warp.at<double>(1, 2);
  h31 = H_warp.at<double>(2, 0);
  h32 = H_warp.at<double>(2, 1);
  h33 = H_warp.at<double>(2, 2);//float时一直溢出
  /*float*data = (float*)H_warp.data;
  h11 = data[1];Mat中data成员解析*/
  for (int i = 0; i < inliers.size(); i += 2)//不是i+2，是i=i+2
  {
	  x1 = inliers[i].x;
	  y1 = inliers[i].y;
	/*  x11 = ((h22 - h32*y1)*(h33*x1 - h13) - (h12 - h32*x1)*(h33*y1 - h23)) / ((h22 - h32*y1)*(h11 - h31*x1) - (h12 - h32*x1)*(h21 - h31*y1));
	  y11 = ((h21 - h31*y1)*(h33*x1 - h13) - (h11 - h31*x1)*(h33*y1 - h23)) / ((h21 - h31*y1)*(h12 - h32*x1) - (h11 - h31*x1)*(h22 - h32*y1));*/
	  x11 = -((h22 - h32*y1)*(h13 - h33*x1) - (h12 - h32*x1)*(h23 - h33*y1)) / ((h22 - h32*y1)*(h11 - h31*x1) - (h12 - h32*x1)*(h21 - h31*y1));
	  y11 = ((h21 - h31*y1)*(h13 - h33*x1) - (h11 - h31*x1)*(h23 - h33*y1)) / ((h22 - h32*y1)*(h11 - h31*x1) - (h12 - h32*x1)*(h21 - h31*y1));
	  x2 = inliers[i + 1].x;
	  y2 = inliers[i + 1].y;
	/*  x22 = (h11*x2 + h12*y2 + h13) / (h31*x2 + h32*y2 + h33);
	  y22 = (h21*x2 + h22*y2 + h23) / (h31*x2 + h32*y2 + h33);*/
	  distance = powf(x11 - x2, 2) + powf(y11 - y2, 2);//powf求平方，pow
	  //distance = powf(x22 - x1, 2) + powf(y22 - y1, 2);
	  sum = sum + distance;
	  
  }
  RMSE = sqrt(sum/ ninliers);
  cout << "均方根误差=" << RMSE << endl;

  //计算信噪比
  warpPerspective(img1, img_xform, H_warp, img_xform.size());
  PSNR_count(img_xform, img2);
  
  

  // Prepare the visualization
  cvtColor(img1, img1_rgb, cv::COLOR_GRAY2BGR);
  cvtColor(img2, img2_rgb, cv::COLOR_GRAY2BGR);
  warpPerspective(img1_rgb, img_xform, H_warp, img_xform.size(), cv::INTER_LINEAR, 0, cv::Scalar(0));
  cv::imshow("xform", img_xform);
  cv::imshow("img1", img1_rgb);
  cv::imshow("img2", img2_rgb);

 

  // Draw the list of detected points
  draw_keypoints(img1_rgb,kpts1);//在彩图上画特征点
  draw_keypoints(img2_rgb,kpts2);

  // Create the new image with a line showing the correspondences
  draw_inliers(img1_rgb, img2_rgb, img_com, inliers);
  cv::resize(img_com, img_r, cv::Size(img_r.cols, img_r.rows), 0, 0, cv::INTER_LINEAR);

  // Show matching statistics
  cout << "Number of Keypoints Image 1: " << nkpts1 << endl;
  cout << "Number of Keypoints Image 2: " << nkpts2 << endl;
  cout << "KAZE Features Extraction Time (ms): " << tkaze << endl;
  cout << "Matching Descriptors Time (ms): " << tmatch << endl;
  cout << "Number of Matches: " << nmatches << endl;
  cout << "Number of Inliers: " << ninliers << endl;
  cout << "Number of Outliers: " << noutliers << endl;
  cout << "Inliers Ratio: " << ratio << endl << endl;

  // Show the images in OpenCV windows
  cv::imshow("Image 1",img1_rgb);
  cv::imshow("Image 2",img2_rgb);
  cv::imshow("Matches",img_com);
 
  cv::waitKey(0);
}


void PSNR_count(cv::Mat x, cv::Mat y)
{
	double MSE, PSNR0;
	double s = 0;
	int PSNR_dB;
	cv::Mat gray1, gray2;
	/*gray1 = cv::Mat(cv::Size(x.cols, x.rows), CV_8UC3);
	gray2 = cv::Mat(cv::Size(y.cols, y.rows), CV_8UC3);*/

	/*cvtColor(x, gray1,cv::COLOR_BGR2GRAY);
	cvtColor(y, gray2, cv::COLOR_BGR2GRAY);*/
	gray1 = x.clone();
	gray2 = y.clone();
	for (int i = 0; i < gray1.rows; i++)//不能是小于等于，否则溢出！
	{
		for (int j = 0; j < gray1.cols; j++)
		{
			if (gray1.at<uchar>(i, j) != 0)
			{
				s = s + int(powf(gray1.at<uchar>(i, j) - gray2.at<uchar>(i, j), 2));//注意循环顺序和坐标顺序，先进行行循环，对应y坐标，at时放在前面
			}
			else s = s + 0;
		}
	}
	MSE = s / ((gray1.rows)*(gray1.cols));
	PSNR0 = powf(255 , 2) / MSE;//不能用^求平方，c++中^是异或
	PSNR_dB = int(10 * log10(PSNR0));
	cout << "复原前后峰值信噪比是：" << PSNR_dB << "dB" << endl;

}


/* ************************************************************************* */
int parse_input_options(KAZEOptions& options, std::string& img_path1, std::string& img_path2,
                        std::string& homography_path, int argc, char *argv[]) {

  // If there is only one argument return
  if (argc == 1) {
    show_input_options_help(1);
    return -1;
  }
  // Set the options from the command line
  else if (argc >= 2) {

    if (!strcmp(argv[1],"--help")) {
      show_input_options_help(1);
      return -1;
    }

    img_path1 = argv[1];
    img_path2 = argv[2];

    if (argc >= 4)
     homography_path = argv[3];

    for (int i = 3; i < argc; i++) {
      if (!strcmp(argv[i],"--soffset")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.soffset = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--omax")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.omax = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--dthreshold")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.dthreshold = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--sderivatives")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.sderivatives = atof(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--nsublevels")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.nsublevels = atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--diffusivity")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.diffusivity = DIFFUSIVITY_TYPE(atoi(argv[i]));
        }
      }
      else if (!strcmp(argv[i],"--descriptor")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.descriptor = DESCRIPTOR_TYPE(atoi(argv[i]));
          if (options.descriptor > GSURF_EXTENDED || options.descriptor < SURF_UPRIGHT) {
            options.descriptor = MSURF;
          }
        }
      }
      else if (!strcmp(argv[i],"--save_scale_space")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.save_scale_space = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--use_fed")) {
        i = i+1;
        if (i >= argc) {
          cout << "Error introducing input options!!" << endl;
          return -1;
        }
        else {
          options.use_fed = (bool)atoi(argv[i]);
        }
      }
      else if (!strcmp(argv[i],"--verbose")) {
        options.verbosity = true;
      }
      else if (!strcmp(argv[i],"--help")) {
        show_input_options_help(1);
        return -1;
      }
    }
  }
  else {
    cout << "Error introducing input options!!" << endl;
    show_input_options_help(1);
    return -1;
  }

  return 0;
}
