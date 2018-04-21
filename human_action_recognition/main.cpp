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
	//��Ƶ·��
	string FilePath = "C:\\9.avi";
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
	Mat foreground;
	Mat mask;
	Mat fgimg;
	int it = 0;

	Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 16, true); //��ϸ�˹��ģ�㷨
	bgsubstractor.dynamicCast<BackgroundSubtractor>();
	bgsubstractor->setVarThreshold(20);

	while (1) {
		if (!capture.read(src))
		{
			break;
		}

		//���ӶԱȶ�
		srcAmend(src);

		//ͼ���С�仯
		//resize(src, src, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);

		bgsubstractor->apply(src, mask, 0.01);

		//��̬ѧ����
		bgAmend(mask);

		///////////////////////////////////////�и�����ͼ��


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
