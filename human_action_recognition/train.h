#pragma once
//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//C
#include <stdio.h>
//C++
#include <iostream>
#include <sstream>
#include <string>
#include <windows.h>
#include <vector>
#include <queue>
#include <iomanip>
#include <fstream>  
#include <sstream>  
#include <string>

using namespace std;
using namespace cv;

void getTrainData();
Mat mergeRows(cv::Mat A, cv::Mat B);
void myTrain();