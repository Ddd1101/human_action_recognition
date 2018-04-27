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

using namespace std;
using namespace cv;

void ContrastAndBright(Mat &src, Mat &dst, double alpha, double beta);//增强对比度和亮度  alpha对比度 beta亮度

void srcAmend(Mat &src);

void bgAmend(Mat &frame);

Mat getApartFrame(Mat &src, Mat &mask);

cv::Mat thinImage(const cv::Mat & src, const int maxIterations = -1);

void filterOver(cv::Mat thinSrc);

std::vector<cv::Point> getPoints(const cv::Mat &thinSrc, unsigned int raudis = 4, unsigned int thresholdMax = 6, unsigned int thresholdMin = 4);

void humanRecognition(Mat img);//人体检测

void outRect(Mat &src);//外接矩形

