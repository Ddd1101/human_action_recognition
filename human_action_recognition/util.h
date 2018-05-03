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

void ContrastAndBright(Mat &src, Mat &dst, double alpha, double beta);//增强对比度和亮度  alpha对比度 beta亮度

void srcAmend(Mat &src);

Wicket bgAmend(Mat &frame);

bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2);

Mat getApartFrame(Mat &src, Mat &mask);

cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1);

void filterOver(cv::Mat thinSrc);

std::vector<cv::Point> getPoints(const cv::Mat &thinSrc, unsigned int raudis = 4, unsigned int thresholdMax = 6, unsigned int thresholdMin = 4);

void humanRecognition(Mat img);//人体检测

void outRect(Mat &src);//外接矩形

