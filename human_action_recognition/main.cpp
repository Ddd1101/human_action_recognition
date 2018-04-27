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
//本工程
#include "util.h"
#include "ViBe.h"
//#include "ImageSegmentation.h"
//#include "BodyDetect.h"
//#include "Body.h"

using namespace cv;
using namespace std;

typedef vector<Point> contour_t;

int main() {
	//视频路径
	string FilePath = "C:\\kth\\running\\1.avi";
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
	Mat tmp;
	Mat mask;
	Mat outline;
	Mat bone;
	int it = 0;

	Mat videoDisplay;
	int cutTop = 30, cutBottom = 10;

	//自适应混合高斯背景建模的背景减除法
	Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 16, false);
	bgsubstractor.dynamicCast<BackgroundSubtractor>();
	bgsubstractor->setVarThreshold(20);
	//ViBe
	ViBe_BGS Vibe_Bgs;
	int count = 0;
	Mat element3 = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat element5 = getStructuringElement(MORPH_RECT, Size(5, 5));
	int maxX = 0, maxY = 0;
	while (1) {
		count++;
		if (!capture.read(src))
		{
			break;
		}
		DWORD startTime = GetCurrentTime();//开始时间

		resize(src, src, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);//图像大小变化

		//自适应混合高斯背景建模的背景减除法
		//图像前景提取处理
		/*srcAmend(src);//增加对比度
		//resize(src, src, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);//图像大小变化

		bgsubstractor->apply(src, mask, 0.01);//得到前景灰度图
		bgAmend(mask);//灰度图形态学处理
		imshow("mask",mask);
		imshow("src",src);*/
		//humanRecognition(src);

		//绘制外接矩形
		/*vector<vector<cv::Point> > contours;
		findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		cout << contours.size() << endl;
		for (int i = 0; i < contours.size(); i++) {
			vector<cv::Mat> PersonMat;	//每个人对应的Mat
			cv::Rect PersonRect = boundingRect(contours[i]);
			Mat cutFrameRectCopy;
			mask(PersonRect).copyTo(cutFrameRectCopy);
			resize(cutFrameRectCopy, cutFrameRectCopy, cv::Size(150.0*PersonRect.width / PersonRect.height, 150.0));
			imshow("outline", cutFrameRectCopy);
		}
		imshow("outline", outline);*/

		//vibe
		cvtColor(src, tmp, CV_BGR2GRAY);
		if (count == 1) {
			Vibe_Bgs.init(tmp);
			Vibe_Bgs.processFirstFrame(tmp);
			cout << "training gmm compelete" << endl;
		}
		else {
			Vibe_Bgs.testAndUpdate(tmp);
			mask = Vibe_Bgs.getMask();
			morphologyEx(mask, mask, MORPH_OPEN, element3);
			//bgAmend(mask);
			threshold(mask, mask, 0, 255, CV_THRESH_OTSU);//最大类间差
			morphologyEx(mask, mask, MORPH_OPEN, element3);//开运算=腐蚀+膨胀
			//morphologyEx(mask, mask, MORPH_CLOSE, element3);//闭运算=膨胀+腐蚀
			//medianBlur(mask, mask, 3);//中值滤波
			//normalize(mask, mask, 0, 255, CV_MINMAX);//归一化

			//绘制外接矩形
			vector<vector<cv::Point> > contours;
			findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			//findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//最外层轮廓  

			int max = 0;
			vector<vector<Point>> ::iterator it;
			for (it = contours.begin(); it != contours.end();)
			{
				if (max < it->size())
				{
					max = it->size();
				}
				it++;
			}
			if (contours.size() > 1) {
				for (it = contours.begin(); it != contours.end();)
				{
					if (it->size() != max)
					{
						it = contours.erase(it);
					}
					else
					{
						it++;
					}
				}
			}
			//cout << contours.size() << "  " << max << endl;
			Mat mask2(mask.size(), CV_8U, Scalar(0));
			drawContours(mask2, contours, -1, Scalar(255), CV_FILLED);

			medianBlur(mask2, mask2, 5);//中值滤波*/
			mask2.copyTo(mask);

			//2.由轮廓确定正外接矩形  
			int width = 0;
			int height = 0;
			int x = 0;
			int y = 0;
			//2.1 定义Rect类型的vector容器boundRect存放正外接矩形，初始化大小为contours.size()即轮廓个数  
			Rect boundRect;
			//2.2 遍历每个轮廓  
				//2.3由轮廓（点集）确定出正外接矩形  
			if (contours.size()==1) {
				boundRect = boundingRect(Mat(contours[0]));
				//2.4获得正外接矩形的左上角坐标及宽高  
				width = boundRect.width;
				height = boundRect.height;
				x = boundRect.x;
				y = boundRect.y;
				if (x>maxX) {
					maxX = x;
				}
				if (y>maxY) {
					maxY = y;
				}

				//2.5用画矩形方法绘制正外接矩形  
				//rectangle(mask, Rect(x, y, width, height), Scalar(255, 0, 0), 2, 8);

				//图像细化，骨骼化    
				bone=thinImage(mask);
				filterOver(bone);//过滤细化后的图像
				std::vector<cv::Point> points = getPoints(bone, 6, 9, 6);//查找端点和交叉点
				bone = bone * 255;//二值图转化成灰度图，并绘制找到的点
				mask = mask * 255;
				vector<cv::Point>::iterator it2 = points.begin();
				for (; it2 != points.end(); it2++)
				{
					circle(bone, *it2, 4, 255, 1);
				}
				imshow("bone", bone);
			}
			else {
				bone = mask;
			}
			

			imshow("mask", mask);
			imshow("src", src);
		}

		

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
		//Mat cutImg = getApartFrame(src, mask);
		//imshow("cutImg", cutImg);
		waitKey(5);
		/*if (waitKey(delay) > 0)
		stop = true;*/
	}
	cout << maxX <<" "<< maxY << endl;
	capture.release();
	waitKey();
	return 0;
}
