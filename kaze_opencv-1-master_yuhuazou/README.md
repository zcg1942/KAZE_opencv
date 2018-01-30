﻿kaze_opencv
===========

wrap KAZE features implementation to cv::Feature2D API without rebuilding OpenCV (minimum version 2.4)

Files:  
		
		|  .gitignore
		|  README.md
		|  box.png				// Test image 1
		|  box_in_scene.png		// Test image 2
		|  KazeOpenCV.cpp		// Sample code of using KAZE to match images
		|  predep.h				// Head file for recognizing OpenCV version
		|  targetver.h			// Windows system level dependences (auto-generated by Visual Studio)
		|   
		|--KAZE
			|   kaze_features.cpp				// Class that warps KAZE to cv::Feature2D
			|   kaze_features.h
			|   kaze.cpp						// Implementation of KAZE
			|   kaze.h
			|   kaze_config.cpp					// Configuration variables and options
			|   kaze_config.h
			|   kaze_ipoint.cpp					// Class that defines a point of interest
			|   kaze_ipoint.h
			|   kaze_nldiffusion_functions.cpp	// Functions for non-linear diffusion applications
			|   kaze_nldiffusion_functions.h
			|   kaze_utils.cpp					// Some useful functions
			|   kaze_utils.h