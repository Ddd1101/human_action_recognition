#include "util.h"
//#include "ImageSegmentation.h"

Mat element3 = getStructuringElement(MORPH_RECT, Size(3, 3));
Mat element5 = getStructuringElement(MORPH_RECT, Size(5, 5));
Mat element7 = getStructuringElement(MORPH_RECT, Size(7, 7));

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
	ContrastAndBright(src, tmp, 1.3, 0);
	src = tmp;
}

double bgAmend(Mat &mask) {
	Rect boundRect;
	Mat tmp;
	double result=0;
	//����������������ȥ��㣬�ٰѿն��������
	morphologyEx(mask, tmp, MORPH_OPEN, element7);//������=��ʴ+����
	//dilate(tmp, tmp, element5);
	morphologyEx(tmp, tmp, MORPH_CLOSE, element7);//������=����+��ʴ
	//dilate(tmp, tmp, element5);
	medianBlur(tmp, tmp, 5);//��ֵ�˲�
	mask.copyTo(tmp);
	//ȥ�����������������
	vector<vector<Point>> contours;
	Mat contours_src = mask.clone();
	findContours(contours_src, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);//��������
	int max = 0;
	vector<vector<Point>> ::iterator it;
	if (contours.size()>0) {
		for (it = contours.begin(); it != contours.end();)
		{
			boundRect = boundingRect(Mat(*it));
			if ((boundRect.y + boundRect.height / 2) < (mask.rows / 6)) {
				it=contours.erase(it);
			}
			else {
				if (max < it->size())
				{
					max = it->size();
				}
				it++;
			}
		}
	}
	if (contours.size()>0) {
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
	//cout << contours.size() << endl;
	//�����ҵ����������Ƶ�����
	Mat mask2(tmp.size(), CV_8U, Scalar(0));
	drawContours(mask2, contours, -1, Scalar(255), CV_FILLED);
	medianBlur(mask2, mask2, 5);//��ֵ�˲�

    //����
	double top, bottom;
	if (contours.size() == 1) {
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
		//cout << center.x << " -- " << center.y << " -- " << top << " -- " << bottom << " -- " << (bottom - center.y) / boundRect.height ;
		result = (bottom - center.y) / boundRect.height;
		//cout << " -- " << result << endl;
	}

	mask2.copyTo(mask);

	threshold(mask, mask, 130, 255, cv::THRESH_BINARY);//��ֵ������
	return result;
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

/**
* @brief ������ͼ�����ϸ��,������
* @param srcΪ����ͼ��,��cvThreshold�����������8λ�Ҷ�ͼ���ʽ��Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
* @param maxIterations���Ƶ���������������������ƣ�Ĭ��Ϊ-1���������Ƶ���������ֱ��������ս��
* @return Ϊ��srcϸ��������ͼ��,��ʽ��src��ʽ��ͬ��Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
*/
cv::Mat thinImage(const cv::Mat & src, const int maxIterations)
{
	assert(src.type() == CV_8UC1);

	src /= 255;

	cv::Mat dst;
	int width = src.cols;
	int height = src.rows;
	src.copyTo(dst);
	//imshow("test3", src);
	int count = 0;  //��¼��������  
	while (true)
	{
		count++;
		if (maxIterations != -1 && count > maxIterations) //���ƴ������ҵ�����������  
			break;
		std::vector<uchar *> mFlag; //���ڱ����Ҫɾ���ĵ�  
									//�Ե���  
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);
				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p6 == 0 && p4 * p6 * p8 == 0)
					{
						//���  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���  
		}

		//�Ե���  
		for (int i = 0; i < height; ++i)
		{
			uchar * p = dst.ptr<uchar>(i);
			for (int j = 0; j < width; ++j)
			{
				//��������ĸ����������б��  
				//  p9 p2 p3  
				//  p8 p1 p4  
				//  p7 p6 p5  
				uchar p1 = p[j];
				if (p1 != 1) continue;
				uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
				uchar p8 = (j == 0) ? 0 : *(p + j - 1);
				uchar p2 = (i == 0) ? 0 : *(p - dst.step + j);
				uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - dst.step + j + 1);
				uchar p9 = (i == 0 || j == 0) ? 0 : *(p - dst.step + j - 1);
				uchar p6 = (i == height - 1) ? 0 : *(p + dst.step + j);
				uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + dst.step + j + 1);
				uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + dst.step + j - 1);

				if ((p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) >= 2 && (p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9) <= 6)
				{
					int ap = 0;
					if (p2 == 0 && p3 == 1) ++ap;
					if (p3 == 0 && p4 == 1) ++ap;
					if (p4 == 0 && p5 == 1) ++ap;
					if (p5 == 0 && p6 == 1) ++ap;
					if (p6 == 0 && p7 == 1) ++ap;
					if (p7 == 0 && p8 == 1) ++ap;
					if (p8 == 0 && p9 == 1) ++ap;
					if (p9 == 0 && p2 == 1) ++ap;

					if (ap == 1 && p2 * p4 * p8 == 0 && p2 * p6 * p8 == 0)
					{
						//���  
						mFlag.push_back(p + j);
					}
				}
			}
		}

		//����ǵĵ�ɾ��  
		for (std::vector<uchar *>::iterator i = mFlag.begin(); i != mFlag.end(); ++i)
		{
			**i = 0;
		}

		//ֱ��û�е����㣬�㷨����  
		if (mFlag.empty())
		{
			break;
		}
		else
		{
			mFlag.clear();//��mFlag���  
		}
	}
	return dst;
}

/**
* @brief �Թ�����ͼ���ݽ��й��ˣ�ʵ��������֮�����ٸ�һ���հ�����
* @param thinSrcΪ����Ĺ�����ͼ��,8λ�Ҷ�ͼ���ʽ��Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
*/
void filterOver(cv::Mat thinSrc)
{
	assert(thinSrc.type() == CV_8UC1);
	int width = thinSrc.cols;
	int height = thinSrc.rows;
	for (int i = 0; i < height; ++i)
	{
		uchar * p = thinSrc.ptr<uchar>(i);
		for (int j = 0; j < width; ++j)
		{
			// ʵ��������֮�����ٸ�һ������  
			//  p9 p2 p3    
			//  p8 p1 p4    
			//  p7 p6 p5    
			uchar p1 = p[j];
			if (p1 != 1) continue;
			uchar p4 = (j == width - 1) ? 0 : *(p + j + 1);
			uchar p8 = (j == 0) ? 0 : *(p + j - 1);
			uchar p2 = (i == 0) ? 0 : *(p - thinSrc.step + j);
			uchar p3 = (i == 0 || j == width - 1) ? 0 : *(p - thinSrc.step + j + 1);
			uchar p9 = (i == 0 || j == 0) ? 0 : *(p - thinSrc.step + j - 1);
			uchar p6 = (i == height - 1) ? 0 : *(p + thinSrc.step + j);
			uchar p5 = (i == height - 1 || j == width - 1) ? 0 : *(p + thinSrc.step + j + 1);
			uchar p7 = (i == height - 1 || j == 0) ? 0 : *(p + thinSrc.step + j - 1);
			if (p2 + p3 + p8 + p9 >= 1)
			{
				p[j] = 0;
			}
		}
	}
}

/**
* @brief �ӹ��˺�Ĺ�����ͼ����Ѱ�Ҷ˵�ͽ����
* @param thinSrcΪ����Ĺ��˺������ͼ��,8λ�Ҷ�ͼ���ʽ��Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
* @param raudis����뾶���Ե�ǰ���ص�λԲ�ģ���Բ��Χ���жϵ��Ƿ�Ϊ�˵�򽻲��
* @param thresholdMax�������ֵ���������ֵΪ�����
* @param thresholdMin�˵���ֵ��С�����ֵΪ�˵�
* @return Ϊ��srcϸ��������ͼ��,��ʽ��src��ʽ��ͬ��Ԫ����ֻ��0��1,1������Ԫ�أ�0����Ϊ�հ�
*/
std::vector<cv::Point> getPoints(const cv::Mat &thinSrc, unsigned int raudis, unsigned int thresholdMax, unsigned int thresholdMin)
{
	assert(thinSrc.type() == CV_8UC1);
	int width = thinSrc.cols;
	int height = thinSrc.rows;
	cv::Mat tmp;
	thinSrc.copyTo(tmp);
	std::vector<cv::Point> points;
	for (int i = 0; i < height; ++i)
	{
		for (int j = 0; j < width; ++j)
		{
			if (*(tmp.data + tmp.step * i + j) == 0)
			{
				continue;
			}
			int count = 0;
			for (int k = i - raudis; k < i + raudis + 1; k++)
			{
				for (int l = j - raudis; l < j + raudis + 1; l++)
				{
					if (k < 0 || l < 0 || k>height - 1 || l>width - 1)
					{
						continue;

					}
					else if (*(tmp.data + tmp.step * k + l) == 1)
					{
						count++;
					}
				}
			}

			if (count > thresholdMax || count < thresholdMin)
			{
				Point point(j, i);
				points.push_back(point);
			}
		}
	}
	return points;
}

void humanRecognition(Mat img) {
	HOGDescriptor hog;
	hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());//opencv��Ĭ�ϵ�ѵ������

	vector< Rect > detections;
	vector< double > foundWeights;

	hog.detectMultiScale(img, detections, foundWeights);//���ü�����ز�������������
														//hog.detectMultiScale(img, detections, 0, Size(8, 8), Size(32, 32), 1.05, 2);
	for (size_t j = 0; j < detections.size(); j++)
	{
		if (foundWeights[j] < 1) continue; //���Ȩֵ��С�ļ�ⴰ��  
		Scalar color = Scalar(0, foundWeights[j] * foundWeights[j] * 200, 0);
		rectangle(img, detections[j], color, img.cols / 400 + 1);
	}

	imshow("human", img);
}

void outRect(Mat &src) {

}