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
	string FilePath = "C:\\kth\\running\\1.avi";
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
		//resize(src, src, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);//ͼ���С�仯

		bgsubstractor->apply(src, mask, 0.01);//�õ�ǰ���Ҷ�ͼ
		bgAmend(mask);//�Ҷ�ͼ��̬ѧ����
		imshow("mask",mask);
		imshow("src",src);*/
		//humanRecognition(src);

		//������Ӿ���
		/*vector<vector<cv::Point> > contours;
		findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
		cout << contours.size() << endl;
		for (int i = 0; i < contours.size(); i++) {
			vector<cv::Mat> PersonMat;	//ÿ���˶�Ӧ��Mat
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
			threshold(mask, mask, 0, 255, CV_THRESH_OTSU);//�������
			morphologyEx(mask, mask, MORPH_OPEN, element3);//������=��ʴ+����
			//morphologyEx(mask, mask, MORPH_CLOSE, element3);//������=����+��ʴ
			//medianBlur(mask, mask, 3);//��ֵ�˲�
			//normalize(mask, mask, 0, 255, CV_MINMAX);//��һ��

			//������Ӿ���
			vector<vector<cv::Point> > contours;
			findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
			//findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//���������  

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

			medianBlur(mask2, mask2, 5);//��ֵ�˲�*/
			mask2.copyTo(mask);

			//2.������ȷ������Ӿ���  
			int width = 0;
			int height = 0;
			int x = 0;
			int y = 0;
			//2.1 ����Rect���͵�vector����boundRect�������Ӿ��Σ���ʼ����СΪcontours.size()����������  
			Rect boundRect;
			//2.2 ����ÿ������  
				//2.3���������㼯��ȷ��������Ӿ���  
			if (contours.size()==1) {
				boundRect = boundingRect(Mat(contours[0]));
				//2.4�������Ӿ��ε����Ͻ����꼰���  
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

				//2.5�û����η�����������Ӿ���  
				//rectangle(mask, Rect(x, y, width, height), Scalar(255, 0, 0), 2, 8);

				//ͼ��ϸ����������    
				bone=thinImage(mask);
				filterOver(bone);//����ϸ�����ͼ��
				std::vector<cv::Point> points = getPoints(bone, 6, 9, 6);//���Ҷ˵�ͽ����
				bone = bone * 255;//��ֵͼת���ɻҶ�ͼ���������ҵ��ĵ�
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
		waitKey(5);
		/*if (waitKey(delay) > 0)
		stop = true;*/
	}
	cout << maxX <<" "<< maxY << endl;
	capture.release();
	waitKey();
	return 0;
}
