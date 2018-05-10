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
//������
#include "util.h"
#include "ViBe.h"

using namespace cv;
using namespace std;


int main() {
	Mat elem3 = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat elem5 = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat elem7 = getStructuringElement(MORPH_RECT, Size(7, 7));
	//��ȡͼ��
	string FilePath = "C:\\dataset\\dataset\\neg\\1";
	string FilePath2 = "C:\\dataset\\neg\\1";

	VideoCapture capture(FilePath);

	vector< String > files;
	vector< String > files2;
	bool isVideo = 0;
	bool isGauss = 0;
	if (isVideo == 1) {
		//��ȡ��Ƶ
		if (!capture.isOpened()) {
			cout << "Can't open the video!" << endl;
			return  1;
		}
		//��ȡ֡��
		double rate = capture.get(CV_CAP_PROP_FPS);
		int delay = 1000 / rate;
	}
	else {
		glob(FilePath, files);
		glob(FilePath2, files2);
	}

	queue<double> points[80];

	Mat src;
	Mat tmp;
	Mat mask;
	Mat bone;

	int cutTop = 30, cutBottom = 10;

	int corner_count = 5000;

	vector<Point2f> corners0;
	vector<Point2f> corners1;

	//����Ӧ��ϸ�˹������ģ�ı���������
	Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 25, false);
	//bgsubstractor.dynamicCast<BackgroundSubtractor>();
	bgsubstractor->setVarThreshold(20);
	//ViBe
	ViBe_BGS Vibe_Bgs;

	int count = 0;
	while (isVideo == 1 || count < files.size() - 1) {
		if (isVideo == 1) {
			if (!capture.read(src))
			{
				break;
			}
		}
		else {
			src = imread(files[count]);
		}
		cvtColor(src, src, COLOR_RGB2GRAY);
		if (src.empty() == 1) {
			break;
		}

		if (isGauss == 1) {//��˹
			//����Ӧ��ϸ�˹������ģ�ı���������
			//ͼ��ǰ����ȡ����
			//srcAmend(src);//���ӶԱȶ�
			//resize(src, src, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);//ͼ���С�仯
			bgsubstractor->apply(src, mask, -1);//�õ�ǰ���Ҷ�ͼ
			bgAmend(mask);
		}
		else {//vibe
			if (count == 0) {
				Vibe_Bgs.init(src);
				Vibe_Bgs.processFirstFrame(src);
				cout << "training gmm compelete" << endl;
			}
			else {
				Vibe_Bgs.testAndUpdate(src);
				mask = Vibe_Bgs.getMask();
				//morphologyEx(mask, mask, MORPH_OPEN, elem3);
				//threshold(mask, mask, 0, 255, CV_THRESH_OTSU);//�������
				morphologyEx(mask, mask, MORPH_OPEN, elem5);//������=��ʴ+����
				morphologyEx(mask, mask, MORPH_CLOSE, elem5);//������=����+��ʴ
				medianBlur(mask, mask, 3);//��ֵ�˲�
			}
		}

		Wicket bound = filterBg_boundRect(mask);

		cout << fixed << setw(4) << double(bound.width) / double(bound.height) << " -- ";
		cout << bound.width * bound.height << " -- ";
		cout << bound.width + bound.height << " -- ";
		cout << bound.height << endl;
		//double point_tmp = wicket.core;//�Ҷ�ͼ��̬ѧ����
		/*if (point_tmp >= 1) {
			point_tmp = 0;
		}
		if (points->size() < 80) {
			points->push(point_tmp * 100 + 50);
		}
		else {
			points->pop();
			points->push(point_tmp * 100 + 50);
		}*/
		//Point tmp;
		//double tmpIn;
		//traits.create(480, 640, CV_8UC3, Scalar(0, 0, 0));
		//Mat traits(480, 640, CV_8UC3, Scalar(0, 0, 0));//����һ��ȫ�ڵ�ͼƬ
		/*for (int i = 0; i < points->size(); i++) {
			tmp.x = (i + 1) * 8;
			tmp.y = points->front();
			points->pop();
			points->push(tmp.y);
			//cout << tmp.x << " -- " << tmp.y << endl;
			circle(traits, tmp, 3, cv::Scalar(0, 255, 0));
		}*/

		//ϡ���������
			/*Mat src0;
			Mat src1;
			Mat gray0;
			Mat gray1;*/
			/*Mat src0 = imread(files2[count++], CV_LOAD_IMAGE_COLOR);
			Mat src1 = imread(files2[count], CV_LOAD_IMAGE_COLOR);
			cvtColor(src0, gray0, CV_BGR2GRAY);
			goodFeaturesToTrack(gray0, corners0, corner_count, 0.01, 5, Mat(), 3, false, 0.04);
			cornerSubPix(gray0, corners0, Size(10, 10), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
			cvtColor(src1, gray1, CV_BGR2GRAY);
			goodFeaturesToTrack(gray1, corners1, corner_count, 0.01, 5, Mat(), 3, false, 0.04);
			cornerSubPix(gray1, corners1, Size(10, 10), Size(-1, -1), TermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 20, 0.03));
			vector<Point2f>::iterator it;

			for (it = corners0.begin(); it != corners0.end(); it++)
			{
				circle(mask, *it, 1, Scalar(255, 255, 255), 1, 8, 0);
			}
			for (it = corners1.begin(); it != corners1.end(); it++)
			{
				circle(mask, *it, 1, Scalar(255, 255, 255), 1, 8, 0);
			}*/


			//�����и�
			/*Mat trait2;
			if (wicket.isEx == 1) {
				Rect rect(wicket.x, wicket.y, wicket.width, wicket.height);
				trait2 = mask(rect);
				copyMakeBorder(trait2, trait2, 480 - wicket.y, 480 - (wicket.y + wicket.height), 640 - wicket.x, 640 - (wicket.x + wicket.width), BORDER_CONSTANT, Scalar(0, 0, 0));
				Mat traitTmp = Mat(wicket.height, wicket.width, CV_8U, Scalar(255, 255, 255));
				copyMakeBorder(traitTmp, traitTmp, 480 - wicket.y, 480 - (wicket.y + wicket.height), 640 - wicket.x, 640 - (wicket.x + wicket.width), BORDER_CONSTANT, Scalar(0, 0, 0));
				Mat traitTmp2;
				traitTmp2.create(mask.size(), mask.type());
				traitTmp2 = Scalar::all(0);
				trait2.copyTo(traitTmp2, traitTmp);
				imshow("tmp", traitTmp2);
			}
			else {
				trait2 = Mat(480, 640, CV_8U, Scalar(0, 0, 0));
			}*/
		if (mask.empty() != 1) {
			imshow("mask", mask);
		}
		imshow("src", src);
		waitKey(10);
		count++;
	}
	//capture.release();
	return 0;
}