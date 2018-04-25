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
	//��Ƶ·��
	string FilePath = "C:\\kth\\running\\2.avi";
	//��ȡ��Ƶ
	VideoCapture capture(FilePath);
	if (!capture.isOpened()) {
		cout << "Can't open the video!" << endl;
		return  1;
	}
	//��ȡ֡��
	double rate = capture.get(CV_CAP_PROP_FPS);
	int delay = 1000 / rate;

	Mat src;
	Mat tmp;
	Mat mask;
	int it = 0;

	Mat videoDisplay;
	//CJcCalBody test;
	int cutTop = 30, cutBottom = 10;

	//����Ӧ��ϸ�˹������ģ�ı���������
	/*Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 16, false);
	bgsubstractor.dynamicCast<BackgroundSubtractor>();
	bgsubstractor->setVarThreshold(20);*/
	//ViBe
	ViBe_BGS Vibe_Bgs;
	int count = 0;
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	while (1) {
		count++;
		if (!capture.read(src))
		{
			break;
		}
		DWORD startTime = GetCurrentTime();//��ʼʱ��

		resize(src, src, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);//ͼ���С�仯

		//����Ӧ��ϸ�˹������ģ�ı���������
		//ͼ��ǰ����ȡ����
		/*srcAmend(src);//���ӶԱȶ�
		//resize(src, src, Size(480, 270));
		resize(src, src, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);//ͼ���С�仯


		bgsubstractor->apply(src, mask, 0.01);//�õ�ǰ���Ҷ�ͼ
		bgAmend(mask);//�Ҷ�ͼ��̬ѧ����*/

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
			morphologyEx(mask, mask, MORPH_OPEN, element);
			//bgAmend(mask);
			morphologyEx(mask, mask, MORPH_OPEN, element);//������=��ʴ+����
			morphologyEx(mask, mask, MORPH_CLOSE, element);//������=����+��ʴ
			//medianBlur(mask, mask, 7);//��ֵ�˲�
			imshow("mask", mask);
		}


		//imshow("gray",tmp);


		//imshow("test", mask);

		//ͼ��ϸ����������    
		/*cv::Mat dst = thinImage(mask);
		filterOver(dst);//����ϸ�����ͼ��
		std::vector<cv::Point> points = getPoints(dst, 6, 9, 6);//���Ҷ˵�ͽ����
		dst = dst * 255;//��ֵͼת���ɻҶ�ͼ���������ҵ��ĵ�
		mask = mask * 255;
		vector<cv::Point>::iterator it = points.begin();
		for (; it != points.end(); it++)
		{
			circle(dst, *it, 4, 255, 1);
		}
		imshow("test2", dst);*/

		//������ȡ


		/*Mat cutFrame = getApartFrame(src, mask);//�и�����ͼ��
		cutFrame.copyTo(videoDisplay);

		cvtColor(cutFrame, cutFrame, CV_BGR2GRAY);

		test.recognizeImage(cutFrame);//ʶ��ͼ���е��������

		std::vector<BodyData> BodyArr;
		test.GetBodyData(BodyArr);//�õ����������������

		for (int i = 0; i < BodyArr.size(); i++)//�����������λ��
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
		waitKey(50);
		/*if (waitKey(delay) > 0)
		stop = true;*/
	}
	capture.release();
	waitKey();
	return 0;
}
