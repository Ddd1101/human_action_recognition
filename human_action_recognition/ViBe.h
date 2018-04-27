#pragma once
#include <iostream>
//#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#define NUM_SAMPLES 20		//ÿ�����ص����������
#define MIN_MATCHES 2		//#minָ��
#define RADIUS 20		//Sqthere�뾶
#define SUBSAMPLE_FACTOR 16	//�Ӳ�������


class ViBe_BGS
{
public:
	ViBe_BGS(void);
	~ViBe_BGS(void);

	void init(const Mat _image);   //��ʼ��
	void processFirstFrame(const Mat _image);
	void testAndUpdate(const Mat _image);  //����
	Mat getMask(void) { return m_mask; };

private:
	Mat m_samples[NUM_SAMPLES];
	Mat m_foregroundMatchCount;
	Mat m_mask;
};