#include "util.h"

Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

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

void srcAmend(Mat &src) {
	Mat tmp;

	//增加对比度
	tmp = Mat::zeros(src.size(), src.type());
	ContrastAndBright(src, tmp, 1.5, 50);
	src = tmp;
}

void bgAmend(Mat &mask) {
	Mat tmp;
	//这两个操作就是先去噪点，再把空洞填充起来
	morphologyEx(mask, tmp, MORPH_OPEN, element);//开运算=腐蚀+膨胀
	dilate(tmp, tmp, element);
	medianBlur(tmp, tmp, 13);
	mask = tmp;
}

Mat getApartFrame(Mat &src, Mat &mask) {
	//===== 对前景掩码做处理，过滤噪点
	vector<vector<Point>> contours;

	//查找轮廓
	findContours(mask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	//将查找到的轮廓绘制到掩码
	Mat dst;
	dst.create(mask.size(), mask.type());
	Mat mask2(dst.size(), CV_8U, Scalar(0));
	drawContours(mask2, contours, -1, Scalar(255), CV_FILLED);

	//对前景掩码的再次掩码处理
	Mat fgmask1;
	mask.copyTo(fgmask1, mask2);
	//===== 前景掩码处理完毕

	//获取前景图像
	dst = Scalar::all(0);
	src.copyTo(dst, fgmask1);
	return dst;
}