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

void ContrastAndBright(Mat &src, Mat &dst, double alpha, double beta);//��ǿ�ԱȶȺ�����  alpha�Աȶ� beta����

int main() {
	//��Ƶ·��
	string FilePath = "C:/8.avi";
	//��ȡ��Ƶ
	VideoCapture capture(FilePath);
	if (!capture.isOpened()) {
		cout << "Can't open the video!" << endl;
		return  1;
	}
	//��ȡ֡��
	double rate = capture.get(CV_CAP_PROP_FPS);
	int delay = 1000 / rate;

	Mat src, frame;
	Mat foreground;
	Mat mask, f1, f2, f3, f4;
	Mat fgimg;
	int it = 0;

	Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 16, true); //��ϸ�˹��ģ�㷨
	bgsubstractor.dynamicCast<BackgroundSubtractor>();
	bgsubstractor->setVarThreshold(20);
	bool stop(false);
	while (!stop) {
		if (!capture.read(src))
		{
			break;
		}

		//���ӶԱȶ�
		//frame = src;
		frame = Mat::zeros(src.size(), src.type());
		ContrastAndBright(src, frame, 1.5, 50);

		//ͼ���С�仯
		//resize(frame, frame, Size(frame.cols * 2, frame.rows * 2), 0, 0, INTER_LINEAR);

		if (fgimg.empty())
		{
			fgimg.create(frame.size(), frame.type());
		}

		bgsubstractor->apply(frame, mask, 0.01);

		//��̬ѧ����
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		//����������������ȥ��㣬�ٰѿն��������
		morphologyEx(mask, f1, MORPH_OPEN, element);//������=��ʴ+����

		medianBlur(f1, f1, 13);

		//GaussianBlur(mask, mask, Size(5, 5), 3.5, 3.5);//��˹ƽ��
		it++;

		///////////////////////////////////////�и�����ͼ��
		//===== ��ǰ�������������������
		vector<vector<Point>> contours;
		Mat contours_src = f1.clone();

		//��������̬����
		Mat ker = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));

		//��������
		findContours(contours_src, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		//�����ҵ����������Ƶ�����
		Mat mask2(fgimg.size(), CV_8U, Scalar(0));
		drawContours(mask2, contours, -1, Scalar(255), CV_FILLED);

		//��ǰ��������ٴ����봦��
		Mat fgmask1;
		f1.copyTo(fgmask1, mask2);
		//===== ǰ�����봦�����

		//��ȡǰ��ͼ��
		fgimg = Scalar::all(0);
		frame.copyTo(fgimg, fgmask1);

		imshow("foreground", fgimg);
		//imshow("mask", f1);
		imshow("src", frame);
		waitKey(100);
		/*if (waitKey(delay) > 0)
		stop = true;*/
	}
	capture.release();
	waitKey();
	return 0;

}

void ContrastAndBright(Mat &src, Mat &dst, double alpha, double beta) {
	// ִ�б任 new_image(i,j) = alpha    * image(i,j) + beta
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			for (int c = 0; c < 3; c++)
			{
				dst.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(alpha * (src.at<Vec3b>(y, x)[c]) + beta);
			}
		}
	}
}
