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
//本工程
#include "util.h"
//#include "ImageSegmentation.h"
//#include "BodyDetect.h"
//#include "Body.h"

using namespace cv;
using namespace std;

typedef vector<Point> contour_t;

int main() {
	//视频路径
	string FilePath = "C:\\kth\\walking\\16.avi";
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
	Mat mask;
	int it = 0;

	Mat videoDisplay;
	//CJcCalBody test;
	int cutTop = 30, cutBottom = 10;


	Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 16, false); //混合高斯建模算法
	bgsubstractor.dynamicCast<BackgroundSubtractor>();
	bgsubstractor->setVarThreshold(20);

	while (1) {
		if (!capture.read(src))
		{
			break;
		}
		DWORD startTime = GetCurrentTime();//开始时间

		//Rect cutRect = Rect(0, cutTop, src.size().width, src.size().height - cutTop - cutBottom);

		//src(cutRect).copyTo(src);//提取矩形子阵

		//图像前景提取处理
		srcAmend(src);//增加对比度
		//resize(src, src, Size(480, 270));
		resize(src, src, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);//图像大小变化
		bgsubstractor->apply(src, mask, 0.01);//得到前景灰度图
		bgAmend(mask);//灰度图形态学处理
		imshow("test", mask);

		//图像细化，骨骼化    
		cv::Mat dst = thinImage(mask);
		filterOver(dst);//过滤细化后的图像    
		std::vector<cv::Point> points = getPoints(dst, 6, 9, 6);//查找端点和交叉点										  
		dst = dst * 255;//二值图转化成灰度图，并绘制找到的点
		mask = mask * 255;
		vector<cv::Point>::iterator it = points.begin();
		for (; it != points.end(); it++)
		{
			circle(dst, *it, 4, 255, 1);
		}
		imshow("test2", dst);

		//骨骼提取


		/*Mat cutFrame = getApartFrame(src, mask);//切割人体图像
		cutFrame.copyTo(videoDisplay);

		cvtColor(cutFrame, cutFrame, CV_BGR2GRAY);

		test.recognizeImage(cutFrame);//识别图像中的人体骨骼

		std::vector<BodyData> BodyArr;
		test.GetBodyData(BodyArr);//得到身体特征点的数据

		for (int i = 0; i < BodyArr.size(); i++)//画出特征点的位置
		{

			for (int j = 0; j < 7; j++)
			{
				if (BodyArr[i]._keyBodyDts[j][0] != NULL)
					circle(videoDisplay, BodyArr[i]._keyBodyDts[j][0]->pos, 4, Scalar(0, 255, 0), -1);

			}

			stringstream s;
			s << BodyArr[i]._index;


			putText(videoDisplay, s.str(), Point(BodyArr[i]._heart.x + 5, BodyArr[i]._heart.y), CV_FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 0, 255));
			cout << BodyArr[i].m_fTimes << endl;
		}

		std::vector<TornadoData> TornadoArr;
		test.GetTornadoData(TornadoArr);

		for (int i = 0; i < TornadoArr.size(); i++)
		{
			circle(videoDisplay, TornadoArr[i]._pos, 4, Scalar(0, 0, 255), -1);
		}

		imshow("foreground", videoDisplay);*/
		imshow("src", src);
		waitKey(50);
		/*if (waitKey(delay) > 0)
		stop = true;*/
	}
	capture.release();
	waitKey();
	return 0;
}
