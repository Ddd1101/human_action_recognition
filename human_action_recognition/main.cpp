#pragma once
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
#include<windows.h>
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
#include "ZernikeMoment.h"

using namespace cv;
using namespace std;
using namespace cv::ml;

#pragma comment(lib,"shlwapi.lib")



int main() {
	Mat elem3 = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat elem5 = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat elem7 = getStructuringElement(MORPH_RECT, Size(7, 7));

	//读写文件
	//string fileName;
	string dataName;
	//string saveFile;
	//saveFile = "C:\\dataset\\svm2\\neg\\";
	//saveFile += to_string(i);
	//LPCSTR savePath = saveFile.c_str();
	//if (!PathIsDirectory(savePath))               // 需要加入上面的头文件
	//{
	//	CreateDirectory(savePath, NULL);
	//}
	dataName = "C:\\dataset\\src\\pos\\1";
	//outFile.open(fileName, ios::app); // 打开模式可省略  
	//读取图像
	string FilePath = dataName;
	VideoCapture capture(FilePath);
	vector< String > files;
	bool isVideo = 0;
	bool isGauss = 1;


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

	//svm1训练数据
	Ptr<ml::TrainData> train_data1;
	train_data1 = ml::TrainData::loadFromCSV("C:\\dataset\\svm1\\stand_pos.csv", 1);
	Mat m1 = train_data1->getTrainSamples();
	normalize(m1, m1, CV_32FC1);
	vector< int > labels1;
	labels1.assign(m1.rows, +1);
	train_data1 = ml::TrainData::loadFromCSV("C:\\dataset\\svm1\\stand_neg.csv", 1);
	Mat m2 = train_data1->getTrainSamples();
	normalize(m2, m2, CV_32FC1);
	labels1.insert(labels1.end(), m2.rows, -1);
	Mat m = mergeRow(m1, m2);

	Ptr< SVM > svm1 = SVM::create();
	/* Default values to train SVM */
	svm1->setCoef0(0.0);
	svm1->setDegree(3);
	svm1->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm1->setGamma(0);
	svm1->setKernel(SVM::LINEAR); //采用线性核函，其他的sigmoid 和RBF 可自行设置，其值由0-5。  
	svm1->setNu(0.5);
	svm1->setP(0.1); // for EPSILON_SVR, epsilon in loss function?  
	svm1->setC(0.01); // From paper, soft classifier  
	svm1->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task  
	svm1->train(m, ROW_SAMPLE, Mat(labels1));

	//svm2训练数据
	Ptr<ml::TrainData> train_data2;
	train_data2 = ml::TrainData::loadFromCSV("C:\\dataset\\svm2\\train\\svm2_pos.csv", 1);
	m1 = train_data2->getTrainSamples();
	normalize(m1, m1, CV_32FC1);
	vector< int > labels2;
	labels2.assign(m1.rows, +1);
	train_data2 = ml::TrainData::loadFromCSV("C:\\dataset\\svm2\\train\\svm2_neg.csv", 1);
	m2 = train_data2->getTrainSamples();
	normalize(m2, m2, CV_32FC1);
	labels2.insert(labels2.end(), m2.rows, -1);
	m = mergeRow(m1, m2);

	Ptr< SVM > svm2 = SVM::create();
	/* Default values to train SVM */
	svm2->setCoef0(0.0);
	svm2->setDegree(3);
	svm2->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm2->setGamma(0);
	svm2->setKernel(SVM::LINEAR); //采用线性核函，其他的sigmoid 和RBF 可自行设置，其值由0-5。  
	svm2->setNu(0.5);
	svm2->setP(0.1); // for EPSILON_SVR, epsilon in loss function?  
	svm2->setC(0.01); // From paper, soft classifier  
	svm2->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task  
	svm2->train(m, ROW_SAMPLE, Mat(labels2));

	int count = 0;
	while (isVideo == 1 || count < files.size()) {
		/*saveFile = "C:\\dataset\\svm2\\neg\\";
		saveFile += to_string(i);*/
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
			//resize(src, src, Size(src.cols * 0.5, src.rows * 0.5), 0, 0, INTER_LINEAR);//图像大小变化
			for (int i = 0; i < files.size();i++) {
				src = imread(files[i]);
				cvtColor(src, src, COLOR_RGB2GRAY);
				string name = "C:\\dataset\\src\\gray\\";
				name += to_string(i);
				name += ".png";
				imwrite(name,src);
				bgsubstractor->apply(src, mask, -1);//得到前景灰度图
				name= "C:\\dataset\\src\\guss\\";
				name += to_string(i);
				name += ".png";
				imwrite(name, mask);
			}
			//bgsubstractor->apply(src, mask, -1);//得到前景灰度图
			cout << "finish" << endl;
			return 0;
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

		bool isStand = 0;

		//svm1-站姿检测
		if (bound.isEx == 1) {
			float labels[4] = { float(bound.width) / float(bound.height), bound.width * bound.height, bound.width + bound.height, bound.height };
			Mat test(1, 4, CV_32F, labels);
			float response = svm1->predict(test);
			cout << "svm1  :";
			if (response < 0.23) {
				cout << 0 << "    ";
				isStand = 0;
			}
			else {
				cout << 1 << "    ";
				isStand = 1;
			}
		}

		//svm2-摔倒检测
		float z_modes[10];
		ZernikeMoment *z_m = new ZernikeMoment();
		int index = 0;
		if (isStand == 0 && bound.isEx == 1) {
			z_m->readImg(mask);

			z_modes[index++] = z_m->getZernike(0, 0);

			z_modes[index++] = z_m->getZernike(1, 1);

			z_modes[index++] = z_m->getZernike(2, 0);

			z_modes[index++] = z_m->getZernike(2, 2);

			z_modes[index++] = z_m->getZernike(3, 1);

			z_modes[index++] = z_m->getZernike(3, 3);

			z_modes[index++] = z_m->getZernike(4, 0);

			z_modes[index++] = z_m->getZernike(4, 2);

			z_modes[index++] = z_m->getZernike(4, 4);

			z_modes[index] = float(bound.width) / float(bound.height);

			Mat test(1, 10, CV_32F, z_modes);
			float response = svm2->predict(test);
			cout << "svm2  : " << response;
		}

		if (bound.isEx == 1) {
			cout << endl;
		}

		if (mask.empty() != 1) {
			imshow("mask", mask);
		}
		imshow("src", src);
		waitKey(2000);
		isStand = 0;
		count++;
		/*saveFile += "\\";
		saveFile += to_string(count);
		saveFile += ".png";
		cout << saveFile << endl;
		imwrite(saveFile, mask);*/
	}
	//capture.release();
	return 0;
}