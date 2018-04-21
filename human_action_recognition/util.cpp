#include "util.h"

Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));

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

void srcAmend(Mat &src) {
	Mat tmp;

	//���ӶԱȶ�
	tmp = Mat::zeros(src.size(), src.type());
	ContrastAndBright(src, tmp, 1.5, 50);
	src = tmp;
}

void bgAmend(Mat &mask) {
	Mat tmp;
	//����������������ȥ��㣬�ٰѿն��������
	morphologyEx(mask, tmp, MORPH_OPEN, element);//������=��ʴ+����
	dilate(tmp, tmp, element);
	medianBlur(tmp, tmp, 13);
	mask = tmp;
}

Mat getApartFrame(Mat &src, Mat &mask) {
	//===== ��ǰ�������������������
	vector<vector<Point>> contours;

	//��������
	findContours(mask, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	//�����ҵ����������Ƶ�����
	Mat dst;
	dst.create(mask.size(), mask.type());
	Mat mask2(dst.size(), CV_8U, Scalar(0));
	drawContours(mask2, contours, -1, Scalar(255), CV_FILLED);

	//��ǰ��������ٴ����봦��
	Mat fgmask1;
	mask.copyTo(fgmask1, mask2);
	//===== ǰ�����봦�����

	//��ȡǰ��ͼ��
	dst = Scalar::all(0);
	src.copyTo(dst, fgmask1);
	return dst;
}