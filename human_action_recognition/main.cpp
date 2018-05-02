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
//本工程
#include "util.h"
#include "ViBe.h"

using namespace cv;
using namespace std;

typedef vector<Point> contour_t;

int main() {
	//读取图像
	//string FilePath = "C:\\dataset\\pos\\video\\3.avi";//视频路径
	string FilePath = "C:\\dataset\\dataset\\pos\\8";
	vector< String > files;
	glob(FilePath, files);
	queue<double> points[80];
	//Mat traits;
	//读取视频
	VideoCapture capture(FilePath);
	/*if (!capture.isOpened()) {
		cout << "Can't open the video!" << endl;
		return  1;
	}
	//读取帧率
	double rate = capture.get(CV_CAP_PROP_FPS);
	int delay = 1000 / rate;*/

	Mat src;
	Mat tmp;
	Mat mask;
	Mat outline;
	Mat bone;
	int it = 0;

	Mat videoDisplay;
	int cutTop = 30, cutBottom = 10;

	//自适应混合高斯背景建模的背景减除法
	Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 25, false);
	//bgsubstractor.dynamicCast<BackgroundSubtractor>();
	bgsubstractor->setVarThreshold(20);
	//ViBe
	ViBe_BGS Vibe_Bgs;
	int count = 0;
	Mat element3 = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat element5 = getStructuringElement(MORPH_RECT, Size(5, 5));
	int maxX = 0, maxY = 0;
	while (count < files.size() - 1) {
	//while(1){
		count++;
		/*if (!capture.read(src))
		{
			break;
		}*/
		src = imread(files[count]);
		//cvtColor(src, src, COLOR_RGB2GRAY);
		if (src.empty() == 1) {
			break;
		}
		DWORD startTime = GetCurrentTime();//开始时间

		//resize(src, src, Size(src.cols / 2, src.rows / 2), 0, 0, INTER_LINEAR);//图像大小变化

		//自适应混合高斯背景建模的背景减除法
		//图像前景提取处理
		//srcAmend(src);//增加对比度
		//resize(src, src, Size(src.cols / 2, src.rows / 2), 0, 0, INTER_LINEAR);//图像大小变化
		bgsubstractor->apply(src, mask, -1);//得到前景灰度图
		double point_tmp=bgAmend(mask);//灰度图形态学处理
		if (point_tmp>=1) {
			point_tmp = 0;
		}
		if (points->size() < 80) {
			points->push(point_tmp*100+50);
		}
		else {
			points->pop();
			points->push(point_tmp * 100+50);
		}
		Point tmp;
		double tmpIn;
		//cout << points->size() << endl;
		//traits.create(480, 640, CV_8UC3, Scalar(0, 0, 0));
		Mat traits(480, 640, CV_8UC3, Scalar(0, 0, 0));//创建一个全黑的图片
		for (int i = 0; i < points->size(); i++) {
			tmp.x = (i + 1) * 8;
			//double y = points->front();
			tmp.y = points->front();
			points->pop();
			points->push(tmp.y);
			cout << tmp.x << " -- " << tmp.y << endl;
			circle(traits, tmp, 3, cv::Scalar(0, 255, 0));
		}
		cout << "===========================" << endl;
		imshow("traits",traits);
		imshow("mask", mask);
		imshow("src", src);


		//vibe
		/*cvtColor(src, tmp, CV_BGR2GRAY);
		if (count == 1) {
			Vibe_Bgs.init(tmp);
			Vibe_Bgs.processFirstFrame(tmp);
			cout << "training gmm compelete" << endl;
		}
		else {
			Vibe_Bgs.testAndUpdate(tmp);
			mask = Vibe_Bgs.getMask();
			morphologyEx(mask, mask, MORPH_OPEN, element3);
			threshold(mask, mask, 0, 255, CV_THRESH_OTSU);//最大类间差
			morphologyEx(mask, mask, MORPH_OPEN, element5);//开运算=腐蚀+膨胀
			morphologyEx(mask, mask, MORPH_CLOSE, element5);//闭运算=膨胀+腐蚀
			medianBlur(mask, mask, 3);//中值滤波

			//重心
			Rect boundRect;
			vector<vector<Point>> contours;
			Mat contours_src = mask.clone();
			findContours(contours_src, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);//查找轮廓
			int max = 0;
			double top, bottom;
			CvPoint center;

			vector<vector<Point>> ::iterator it;
			if (contours.size()>0) {
				for (it = contours.begin(); it != contours.end();)
				{
					boundRect = boundingRect(Mat(*it));
					if ((boundRect.y + boundRect.height / 2) < (mask.rows / 6)) {
						it = contours.erase(it);
					}
					else {
						if (max < it->size())
						{
							max = it->size();
						}
						it++;
					}
				}
			}
			if (contours.size()>0) {
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

			Mat mask2(tmp.size(), CV_8U, Scalar(0));
			drawContours(mask2, contours, -1, Scalar(255), CV_FILLED);
			mask2.copyTo(mask);

			if (contours.size() >= 1) {
				top = bottom = 0;
				IplImage tmp = IplImage(mask);
				double m00, m10, m01;
				CvMoments moment;
				cvMoments((CvArr*)&tmp, &moment, 1);
				m00 = cvGetSpatialMoment(&moment, 0, 0);
				m10 = cvGetSpatialMoment(&moment, 1, 0);
				m01 = cvGetSpatialMoment(&moment, 0, 1);
				center.x = (int)(m10 / m00);
				center.y = (int)(m01 / m00);
				boundRect = boundingRect(Mat(contours[0]));
				top = boundRect.y;
				bottom = boundRect.y + boundRect.height;
				cout << (bottom - center.y) / boundRect.height << endl;
			}
			else {
				center.x = center.y = 0;
			}

			double point_tmp = (bottom - center.y) / boundRect.height;//灰度图形态学处理
			if (point_tmp >= 1) {
				point_tmp = 0;
			}
			if (points->size() < 80) {
				points->push(point_tmp * 100 + 50);
			}
			else {
				points->pop();
				points->push(point_tmp * 100 + 50);
			}

			Point tmp;
			double tmpIn;
			Mat traits(480, 640, CV_8UC3, Scalar(0, 0, 0));//创建一个全黑的图片
			for (int i = 0; i < points->size(); i++) {
				tmp.x = (i + 1) * 8;
				tmp.y = points->front();
				points->pop();
				points->push(tmp.y);
				//cout << tmp.x << " -- " << tmp.y << endl;
				circle(traits, tmp, 3, cv::Scalar(0, 255, 0));
			}

			imshow("traits", traits);
			imshow("mask", mask);
			imshow("src", src);
		}*/
		waitKey(10);
	}
	//capture.release();
	waitKey();
	return 0;
}
