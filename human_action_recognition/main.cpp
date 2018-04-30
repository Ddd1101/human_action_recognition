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
//������
#include "util.h"
#include "ViBe.h"
//#include "ImageSegmentation.h"
//#include "BodyDetect.h"
//#include "Body.h"

using namespace cv;
using namespace std;

typedef vector<Point> contour_t;

int main() {
	//��ȡͼ��
	//string FilePath = "C:\\dataset\\pos\\video\\3.avi";//��Ƶ·��
	string FilePath = "C:\\dataset\\pos\\30";
	vector< String > files;
	glob(FilePath, files);
	queue<double> points[80];
	//Mat traits;
	//��ȡ��Ƶ
	/*VideoCapture capture(FilePath);
	if (!capture.isOpened()) {
		cout << "Can't open the video!" << endl;
		return  1;
	}
	//��ȡ֡��
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

	//����Ӧ��ϸ�˹������ģ�ı���������
	Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 16, false);
	bgsubstractor.dynamicCast<BackgroundSubtractor>();
	bgsubstractor->setVarThreshold(20);
	//ViBe
	ViBe_BGS Vibe_Bgs;
	int count = 0;
	Mat element3 = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat element5 = getStructuringElement(MORPH_RECT, Size(5, 5));
	int maxX = 0, maxY = 0;
	while (count < files.size() - 1) {
		count++;
		/*if (!capture.read(src))
		{
			break;
		}*/
		src = imread(files[count]);
		cvtColor(src, src, COLOR_RGB2GRAY);
		if (src.empty() == 1) {
			break;
		}
		DWORD startTime = GetCurrentTime();//��ʼʱ��

		//resize(src, src, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);//ͼ���С�仯

		//����Ӧ��ϸ�˹������ģ�ı���������
		//ͼ��ǰ����ȡ����
		//srcAmend(src);//���ӶԱȶ�
		resize(src, src, Size(src.cols / 2, src.rows / 2), 0, 0, INTER_LINEAR);//ͼ���С�仯
		bgsubstractor->apply(src, mask, -1);//�õ�ǰ���Ҷ�ͼ
		double point_tmp=bgAmend(mask);//�Ҷ�ͼ��̬ѧ����
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
		Mat traits(480, 640, CV_8UC3, Scalar(0, 0, 0));//����һ��ȫ�ڵ�ͼƬ
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
		waitKey(10);


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
			//bgAmend(mask);
			threshold(mask, mask, 0, 255, CV_THRESH_OTSU);//�������
			morphologyEx(mask, mask, MORPH_OPEN, element5);//������=��ʴ+����
			morphologyEx(mask, mask, MORPH_CLOSE, element5);//������=����+��ʴ
			medianBlur(mask, mask, 3);//��ֵ�˲�

			//����
			Rect boundRect;
			double top, bottom;
			if (contours.size() == 1) {
				top = bottom = 0;
				IplImage tmp = IplImage(mask);
				CvPoint center;
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
				cout << center.x << " -- " << center.y << " -- " << top << " -- " << bottom << " -- " << (bottom - center.y) / boundRect.height << endl;
			}

			imshow("mask", mask);
			imshow("src", src);
		}*/
		waitKey(10);
	}
	//capture.release();
	waitKey();
	return 0;
}
