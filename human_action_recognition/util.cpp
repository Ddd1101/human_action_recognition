#include "util.h"
//#include "ImageSegmentation.h"

Mat element3 = getStructuringElement(MORPH_RECT, Size(3, 3));
Mat element5 = getStructuringElement(MORPH_RECT, Size(5, 5));
Mat element7 = getStructuringElement(MORPH_RECT, Size(7, 7));

//对比度
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
	ContrastAndBright(src, tmp, 1.3, 0);
	src = tmp;
}

Wicket bgAmend(Mat &mask) {
	Rect boundRect;
	Wicket result ;
	//这两个操作就是先去噪点，再把空洞填充起来
	morphologyEx(mask, mask, MORPH_OPEN, element5);//开运算=腐蚀+膨胀
	//dilate(mask, mask, element3);
	morphologyEx(mask, mask, MORPH_CLOSE, element5);//闭运算=膨胀+腐蚀
	medianBlur(mask, mask, 5);//中值滤波
	//去掉人体外的其他部分
	vector<vector<Point>> contours;
	Mat contours_src = mask.clone();
	findContours(contours_src, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);//查找轮廓
	sort(contours.begin(), contours.end(), ContoursSortFun);
	Mat toMask(480, 640, CV_8UC3, Scalar(0, 0, 0));//创建一个全黑的图片
	vector<vector<Point>> ::iterator it;

	//重心
	double top, bottom;
	double s;
	/*if (contours.size() > 0) {
		s = (double)contourArea(contours[0], true)*(-1);
		if (s < 3000) {
			cvtColor(toMask, mask, CV_BGR2GRAY);
			return result;
		}
		result.isEx = 1;
		drawContours(toMask, contours, 0, Scalar(255, 255, 255), -1);
		cvtColor(toMask, mask, CV_BGR2GRAY);
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
		result.x = boundRect.x;
		result.y = boundRect.y;
		result.height = boundRect.height;
		result.width = boundRect.width;
		//cout << center.x << " -- " << center.y << " -- " << top << " -- " << bottom << " -- " << (bottom - center.y) / boundRect.height ;
		result.core = (bottom - center.y) / boundRect.height;
		//rectangle(mask, Rect(boundRect.x, boundRect.y, boundRect.width, boundRect.height), Scalar(255, 255, 255), 1, 8);
	}*/
	return result;
}

bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2) {
	return (contour1.size() > contour2.size());
}

double centerPoint(Mat mask, vector<vector<Point>> contours) {
	double result;
	Rect boundRect;
	double top, bottom;
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
	result = (bottom - center.y) / boundRect.height;
	return result;
}

//前景分割
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

//行人跟踪
void humanRecognition(Mat img) {
	HOGDescriptor hog;
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());//opencv中默认的训练数据

	vector< Rect > detections;
	vector< double > foundWeights;

	hog.detectMultiScale(img, detections, foundWeights);//设置检测的相关参数及返回数据
														//hog.detectMultiScale(img, detections, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	for (size_t j = 0; j < detections.size(); j++)
	{
		if (foundWeights[j] < 1) continue; //清楚权值较小的检测窗口  
		Scalar color = Scalar(0, foundWeights[j] * foundWeights[j] * 200, 0);
		rectangle(img, detections[j], color, img.cols / 400 + 1);
	}

	imshow("human", img);
}

//Vibe

