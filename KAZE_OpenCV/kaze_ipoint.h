
/**
 * @file Ipoint.h
 * @brief Class that defines a point of interest
 * @date Jan 21, 2012
 * @author Pablo F. Alcantarilla
 * @update: 2013-03-28 by Yuhua Zou
 */

#ifndef _IPOINT_H_
#define _IPOINT_H_

//*************************************************************************************
//*************************************************************************************

// Includes
#include <vector>
#include <algorithm>
#include <math.h>
#include "opencv2/core/core.hpp"

// Ipoint Class Declaration
class Ipoint
{

public:

    // ������ĸ���������������꣨Coordinates of the detected interest point��
    float xf,yf;    // Float coordinates
    int x,y;        // Integer coordinates

    // ������ĳ߶ȼ��𣬦�Ϊ��λ��Detected scale of the interest point (sigma units)��
    float scale;

    // ͼ��߶Ȳ���������ֵ��Size of the image derivatives (pixel units)��
    int sigma_size;

    // ���������Ӧֵ��Feature detector response��
    float dresponse;

    // ����ʱ�䣨Evolution time��
    float tevolution;

    // ������������Octave�飨Octave of the keypoint��
    float octave;

    // �����������Ĳ㼶��Sublevel in each octave of the keypoint��
    float sublevel;

    // �����������������Descriptor vector and size��
    std::vector<float> descriptor;
    int descriptor_size;

    // �������������Main orientation��
    float angle;

    // �����������ͣ�Descriptor mode��
    int descriptor_mode;

    // ������˹��־ֵ��Sign of the laplacian (for faster matching)��
    int laplacian;

    // ��������Evolution Level��
    unsigned int level;

    // Constructor
    Ipoint(void);


    // ������Ӧֵ����������� ��Sort Ipoint by response value��
    bool operator < (const Ipoint& rhs ) const   
    {   
        return dresponse < rhs.dresponse; 
    }
    bool operator > (const Ipoint& rhs ) const   
    {   
        return dresponse > rhs.dresponse; 
    }

};

//*************************************************************************************
//*************************************************************************************

/**
 * Filters for KAZE::Ipoint
 */
void filterDuplicated( std::vector<Ipoint>& keypoints );

void filterRetainBest(std::vector<Ipoint>& keypoints, int n_points);

void filterUnvalidKeypoints( std::vector<Ipoint>& keypoints );

void filterByPixelsMask( std::vector<Ipoint>& keypoints, const cv::Mat& mask );

#endif

