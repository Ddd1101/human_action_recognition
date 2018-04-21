#pragma once
//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>
#include <string>

using namespace std;
using namespace cv;

void ContrastAndBright(Mat &src, Mat &dst, double alpha, double beta);//增强对比度和亮度  alpha对比度 beta亮度