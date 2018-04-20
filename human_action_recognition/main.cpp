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

using namespace cv;
using namespace std;

typedef vector<Point> contour_t;

void lsn5_Foreground(void);

int main() {
	//��Ƶ·��
	string FilePath = "C:/7.mp4";
	//��ȡ��Ƶ
	VideoCapture capture(FilePath);
	if (!capture.isOpened()) {
		cout << "Can't open the video!" << endl;
		return  1;
	}
	//��ȡ֡��
	double rate = capture.get(CV_CAP_PROP_FPS);
	int delay = 1000 / rate;

	Mat frame;
	Mat foreground;
	Mat mask, f1, f2, f3, f4;
	/*string s1 = "C:\\final\\part1\\background\\10\\";
	string s2 = "C:\\final\\part1\\background\\19\\";
	string s3 = "C:\\final\\part1\\background\\111\\";
	string s4 = "C:\\final\\part1\\background\\113\\";*/
	int it = 0;

	//namedWindow("Extracted Foreground");
	Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 16, true); //��ϸ�˹��ģ�㷨
	bgsubstractor.dynamicCast<BackgroundSubtractor>();
	bgsubstractor->setVarThreshold(20);
	bool stop(false);
	while (!stop) {
		if (!capture.read(frame))
		{
			break;
		}
		resize(frame, frame, Size(frame.cols / 2, frame.rows / 2), 0, 0, INTER_LINEAR);

		bgsubstractor->apply(frame, mask, 0.01);

		//��̬ѧ����
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		//����������������ȥ��㣬�ٰѿն��������
		morphologyEx(mask, f1, MORPH_OPEN, element);//������=��ʴ+����

													/*medianBlur(f1, f2, 9);
													medianBlur(f1, f3, 11);*/
		medianBlur(f1, f1, 13);

		/*string name = to_string(it);
		name += ".jpg";
		if (it % 5 == 0) {
		imwrite(s2 + name, f2);
		imwrite(s3 + name, f3);
		imwrite(s4 + name, f4);
		}*/
		it++;

		imshow("mask", f1);
		waitKey(50);
		/*if (waitKey(delay) > 0)
		stop = true;*/
	}
	capture.release();
	waitKey();
	return 0;

}
