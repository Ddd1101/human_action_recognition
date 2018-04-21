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

#include "util.h"

using namespace cv;
using namespace std;

typedef vector<Point> contour_t;

int main() {
	//视频路径
	string FilePath = "C:\\9.avi";
	//读取视频
	VideoCapture capture(FilePath);
	if (!capture.isOpened()) {
		cout << "Can't open the video!" << endl;
		return  1;
	}
	//读取帧率
	double rate = capture.get(CV_CAP_PROP_FPS);
	int delay = 1000 / rate;

	Mat src;
	Mat foreground;
	Mat mask;
	Mat fgimg;
	int it = 0;

	Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 16, true); //混合高斯建模算法
	bgsubstractor.dynamicCast<BackgroundSubtractor>();
	bgsubstractor->setVarThreshold(20);

	while (1) {
		if (!capture.read(src))
		{
			break;
		}

		//增加对比度
		srcAmend(src);

		//图像大小变化
		//resize(src, src, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);

		bgsubstractor->apply(src, mask, 0.01);

		//形态学处理
		bgAmend(mask);

		///////////////////////////////////////切割人体图像


		Mat dst = getApartFrame(src,mask);
		imshow("foreground", dst);
		imshow("src", src);
		waitKey(10);
		/*if (waitKey(delay) > 0)
		stop = true;*/
	}
	capture.release();
	waitKey();
	return 0;

}
