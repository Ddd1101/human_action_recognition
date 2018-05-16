#pragma once
//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect/objdetect.hpp>    
//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>

using namespace std;
using namespace cv;

struct Wicket {
	int x = 0;
	int y = 0;
	int height = 0;
	int width = 0;
	double core = 0;
	bool isEx = 0;
};

void ContrastAndBright(Mat &src, Mat &dst, double alpha, double beta);//��ǿ�ԱȶȺ�����  alpha�Աȶ� beta����

void srcAmend(Mat &src);

void bgAmend(Mat &frame);

bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2);

Mat getApartFrame(Mat &src, Mat &mask);

void humanRecognition(Mat img);//������

void outRect(Mat &src);//��Ӿ���

//ȥ�����������������
void filterBg(Mat &mask);

//ȥ����������������ֲ��õ���Ӿ���
Wicket filterBg_boundRect(Mat &mask);

//��������
Wicket core(Mat mask);

Mat mergeRow(Mat A, Mat B);

Mat RegionGrow(Mat MatIn, int iGrowPoint, int iGrowJudge);//iGrowPointΪ���ӵ���ж�������iGrowJudgeΪ��������

void lightTrait();//ϡ���������

