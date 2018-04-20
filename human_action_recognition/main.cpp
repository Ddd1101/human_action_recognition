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

void ContrastAndBright(Mat &src, Mat &dst, double alpha, double beta);//增强对比度和亮度  alpha对比度 beta亮度

int main() {
	//视频路径
	string FilePath = "C:/8.avi";
	//读取视频
	VideoCapture capture(FilePath);
	if (!capture.isOpened()) {
		cout << "Can't open the video!" << endl;
		return  1;
	}
	//读取帧率
	double rate = capture.get(CV_CAP_PROP_FPS);
	int delay = 1000 / rate;

	Mat src, frame;
	Mat foreground;
	Mat mask, f1, f2, f3, f4;
	Mat fgimg;
	int it = 0;

	Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 16, true); //混合高斯建模算法
	bgsubstractor.dynamicCast<BackgroundSubtractor>();
	bgsubstractor->setVarThreshold(20);
	bool stop(false);
	while (!stop) {
		if (!capture.read(src))
		{
			break;
		}

		//增加对比度
		//frame = src;
		frame = Mat::zeros(src.size(), src.type());
		ContrastAndBright(src, frame, 1.5, 50);

		//图像大小变化
		//resize(frame, frame, Size(frame.cols * 2, frame.rows * 2), 0, 0, INTER_LINEAR);

		if (fgimg.empty())
		{
			fgimg.create(frame.size(), frame.type());
		}

		bgsubstractor->apply(frame, mask, 0.01);

		//形态学处理
		Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		//这两个操作就是先去噪点，再把空洞填充起来
		morphologyEx(mask, f1, MORPH_OPEN, element);//开运算=腐蚀+膨胀

		medianBlur(f1, f1, 13);

		//GaussianBlur(mask, mask, Size(5, 5), 3.5, 3.5);//高斯平滑
		it++;

		///////////////////////////////////////切割人体图像
		//===== 对前景掩码做处理，过滤噪点
		vector<vector<Point>> contours;
		Mat contours_src = f1.clone();

		//对轮廓形态处理
		Mat ker = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));

		//查找轮廓
		findContours(contours_src, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		//将查找到的轮廓绘制到掩码
		Mat mask2(fgimg.size(), CV_8U, Scalar(0));
		drawContours(mask2, contours, -1, Scalar(255), CV_FILLED);

		//对前景掩码的再次掩码处理
		Mat fgmask1;
		f1.copyTo(fgmask1, mask2);
		//===== 前景掩码处理完毕

		//获取前景图像
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
	// 执行变换 new_image(i,j) = alpha    * image(i,j) + beta
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
