//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml.hpp>  
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
#include <fstream>  
#include <sstream>  
#include "shlwapi.h"
//本工程
#include "util.h"
#include "ViBe.h"

using namespace cv;
using namespace std;
using namespace cv::ml;

#pragma comment(lib,"shlwapi.lib")


int main() {
	Mat elem3 = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat elem5 = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat elem7 = getStructuringElement(MORPH_RECT, Size(7, 7));
	//读写文件
	string fileName;
	string dataName;
	string saveFile;
	for (int i = 1; i <= 40; i++) {
		saveFile = "C:\\dataset\\svm2\\neg\\";
		saveFile += to_string(i);
		LPCSTR savePath = saveFile.c_str();
		if (!PathIsDirectory(savePath))               // 需要加入上面的头文件
		{
			CreateDirectory(savePath, NULL);
		}
		dataName = "C:\\dataset\\dataset\\neg\\";
		dataName += to_string(i);
		//outFile.open(fileName, ios::app); // 打开模式可省略  
		//读取图像
		string FilePath = dataName;
		VideoCapture capture(FilePath);
		vector< String > files;
		bool isVideo = 0;
		bool isGauss = 0;


		if (isVideo == 1) {
			//读取视频
			if (!capture.isOpened()) {
				cout << "Can't open the video!" << endl;
				return  1;
			}
			//读取帧率
			double rate = capture.get(CV_CAP_PROP_FPS);
			int delay = 1000 / rate;
		}
		else {
			glob(FilePath, files);
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

		//自适应混合高斯背景建模的背景减除法
		Ptr<BackgroundSubtractorMOG2>bgsubstractor = createBackgroundSubtractorMOG2(500, 25, false);
		//bgsubstractor.dynamicCast<BackgroundSubtractor>();
		bgsubstractor->setVarThreshold(20);
		//ViBe
		ViBe_BGS Vibe_Bgs;

		//Ptr<ml::TrainData> train_data;
		//train_data = ml::TrainData::loadFromCSV("C:\\dataset\\data\\stand_pos.csv", 1);
		//Mat m1 = train_data->getTrainSamples();
		//normalize(m1, m1, CV_32FC1);
		//vector< int > labels;
		//labels.assign(m1.rows, +1);
		//train_data = ml::TrainData::loadFromCSV("C:\\dataset\\data\\stand_neg.csv", 1);
		//Mat m2 = train_data->getTrainSamples();
		//normalize(m2, m2, CV_32FC1);
		//labels.insert(labels.end(), m2.rows, -1);
		//Mat m = mergeRow(m1, m2);


		//Ptr< SVM > svm = SVM::create();
		///* Default values to train SVM */
		//svm->setCoef0(0.0);
		//svm->setDegree(3);
		//svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
		//svm->setGamma(0);
		//svm->setKernel(SVM::LINEAR); //采用线性核函，其他的sigmoid 和RBF 可自行设置，其值由0-5。  
		//svm->setNu(0.5);
		//svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?  
		//svm->setC(0.01); // From paper, soft classifier  
		//svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task  
		//svm->train(m, ROW_SAMPLE, Mat(labels));
		//cout << svm->getVarCount() << endl;

		int count = 0;
		while (isVideo == 1 || count < files.size() - 1) {
			saveFile = "C:\\dataset\\svm2\\neg\\";
			saveFile += to_string(i);
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

			if (isGauss == 1) {//高斯
				//自适应混合高斯背景建模的背景减除法
				//图像前景提取处理
				//srcAmend(src);//增加对比度
				//resize(src, src, Size(src.cols * 2, src.rows * 2), 0, 0, INTER_LINEAR);//图像大小变化
				bgsubstractor->apply(src, mask, -1);//得到前景灰度图
				bgAmend(mask);
			}
			else {//vibe
				if (count == 0) {
					Vibe_Bgs.init(src);
					Vibe_Bgs.processFirstFrame(src);
					cout << "training gmm compelete" << endl;
				}
				else {
					//threshold(src, src, 0,255, THRESH_BINARY);//最大类间差
					Vibe_Bgs.testAndUpdate(src);
					mask = Vibe_Bgs.getMask();
					//morphologyEx(mask, mask, MORPH_OPEN, elem3);

					morphologyEx(mask, mask, MORPH_OPEN, elem7);//开运算=腐蚀+膨胀
					morphologyEx(mask, mask, MORPH_CLOSE, elem7);//闭运算=膨胀+腐蚀
					medianBlur(mask, mask, 3);//中值滤波
				}
			}

			Wicket bound = filterBg_boundRect(mask);

			/*cout << fixed << setw(4) << double(bound.width) / double(bound.height) << " -- ";
			cout << bound.width * bound.height << " -- ";
			cout << bound.width + bound.height << " -- ";
			cout << bound.height << endl;*/

			//if (bound.isEx == 1) {
			//	//outFile << count + 1 << ',' << double(bound.width) / double(bound.height) << ',' << bound.width * bound.height << ',' << bound.width + bound.height << ',' << bound.height << endl;
			//	float labels[4] = { float(bound.width) / float(bound.height), bound.width * bound.height, bound.width + bound.height, bound.height };
			//	Mat test(1, 4, CV_32F, labels);
			//	float response = svm->predict(test);
			//	if (response < 0.23) {
			//		cout << 0 << endl;
			//	}
			//	else {
			//		cout << 1 << endl;
			//	}
			//}

			//double point_tmp = wicket.core;//灰度图形态学处理
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
			//Mat traits(480, 640, CV_8UC3, Scalar(0, 0, 0));//创建一个全黑的图片
			/*for (int i = 0; i < points->size(); i++) {
				tmp.x = (i + 1) * 8;
				tmp.y = points->front();
				points->pop();
				points->push(tmp.y);
				//cout << tmp.x << " -- " << tmp.y << endl;
				circle(traits, tmp, 3, cv::Scalar(0, 255, 0));
			}*/

			//稀疏光流特征
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


				//特征切割
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
				/*if (mask.empty() != 1) {
					imshow("mask", mask);
				}
				imshow("src", src);*/
			waitKey(10);
			count++;
			saveFile += "\\";
			saveFile += to_string(count);
			saveFile += ".png";
			cout << saveFile << endl;
			imwrite(saveFile,mask);
		}
	}
	//capture.release();
	return 0;
}