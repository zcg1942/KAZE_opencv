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

#include "./lib/KAZE.h"

using namespace std;

/* ************************************************************************* */
// Some image matching options
const float MAX_H_ERROR = 2.50;	// Maximum error in pixels to accept an inlier
const float DRATIO = .80;		// NNDR Matching value

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
  cv::Mat img1, img1_32, img2, img2_32, img1_rgb, img2_rgb, img_com, img_r;
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
    return -1;

  // Read the image, force to be grey scale
  img1 = cv::imread(img_path1, 0);

  if (img1.data == NULL) {
    cerr << "Error loading image: " << img_path1 << endl;
    return -1;
  }

  // Read the image, force to be grey scale
  img2 = cv::imread(img_path2, 0);

  if (img2.data == NULL) {
    cout << "Error loading image: " << img_path2 << endl;
    return -1;
  }

  // Convert the images to float
  img1.convertTo(img1_32, CV_32F, 1.0/255.0, 0);
  img2.convertTo(img2_32, CV_32F, 1.0/255.0, 0);

  // Color images for results visualization
  img1_rgb = cv::Mat(cv::Size(img1.cols,img1.rows), CV_8UC3);
  img2_rgb = cv::Mat(cv::Size(img2.cols,img1.rows), CV_8UC3);
  img_com = cv::Mat(cv::Size(img1.cols*2,img1.rows), CV_8UC3);
  img_r = cv::Mat(cv::Size(img_com.cols*rfactor, img_com.rows*rfactor), CV_8UC3);

  // Read ground truth homography file
  bool use_ransac = false;
  if (read_homography(homography_path, HG) == false)
    use_ransac = true;

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
  tkaze = 1000.0*(t2-t1) / cv::getTickFrequency();

  nkpts1 = kpts1.size();
  nkpts2 = kpts2.size();

  // Matching Descriptors!!
  vector<cv::Point2f> matches, inliers;
  cv::Ptr<cv::DescriptorMatcher> matcher_l2 = cv::DescriptorMatcher::create("BruteForce");

  t1 = cv::getTickCount();

  matcher_l2->knnMatch(desc1, desc2, dmatches,2);
  matches2points_nndr(kpts1, kpts2, dmatches, matches, DRATIO);

  t2 = cv::getTickCount();
  tmatch = 1000.0*(t2-t1) / cv::getTickFrequency();

  // Compute Inliers!!
  if (use_ransac == false)
    compute_inliers_homography(matches, inliers, HG, MAX_H_ERROR);
  else
    compute_inliers_ransac(matches, inliers, MAX_H_ERROR, false);

  // Compute the inliers statistics
  nmatches = matches.size()/2;
  ninliers = inliers.size()/2;
  noutliers = nmatches - ninliers;
  ratio = 100.0*((float) ninliers / (float) nmatches);

  // Prepare the visualization
  cvtColor(img1, img1_rgb, cv::COLOR_GRAY2BGR);
  cvtColor(img2, img2_rgb, cv::COLOR_GRAY2BGR);

  // Draw the list of detected points
  draw_keypoints(img1_rgb,kpts1);
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
