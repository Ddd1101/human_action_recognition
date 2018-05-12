#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "train.h"

using namespace std;
using namespace cv;
using namespace cv::ml;


void getTrainData() {
	string FilePath = "C:\\dataset\\data\\squat";
	vector< String > files;
	glob(FilePath, files);
	ofstream outFile("C:\\dataset\\\data\\stand_neg.csv", ios::app);
	for (int i = 0; i < files.size(); i++) {
		ifstream inFile(files[i], ios::in);
		string input;
		string inputTmp;
		string output;
		vector<string> data;
		int it;
		while (getline(inFile, input)) {
			stringstream sstr(input);
			it = 0;
			while (getline(sstr, inputTmp, ',')) {
				if (it == 0) {
					it++;
					continue;
				}
				if (it != 1) {
					outFile << ',';
				}
				outFile << inputTmp;
				it++;
			}
			outFile << endl;
		}
	}
	cout << "finish" << endl;
	outFile.close();
}


Mat mergeRows(cv::Mat A, cv::Mat B){
	// cv::CV_ASSERT(A.cols == B.cols&&A.type() == B.type());
	int totalRows = A.rows + B.rows;
	cv::Mat mergedDescriptors(totalRows, A.cols, A.type());
	cv::Mat submat = mergedDescriptors.rowRange(0, A.rows);
	A.copyTo(submat);
	submat = mergedDescriptors.rowRange(A.rows, totalRows);
	B.copyTo(submat);
	return mergedDescriptors;
}

float stringToNum(string str){
	istringstream iss(str);
	float num;
	iss >> num;
	return num;
}

void myTrain() {

	Ptr<ml::TrainData> train_data;
	train_data = ml::TrainData::loadFromCSV("C:\\dataset\\data\\stand_pos.csv", 1);
	Mat m1 = train_data->getTrainSamples();
	normalize(m1, m1, CV_32FC1);
	vector< int > labels;
	labels.assign(m1.rows, +1);
	cout << m1.rows << endl;
	train_data = ml::TrainData::loadFromCSV("C:\\dataset\\data\\stand_neg.csv", 1);
	Mat m2 = train_data->getTrainSamples();
	normalize(m2, m2, CV_32FC1);
	cout << m2.rows << endl;
	labels.insert(labels.end(), m2.rows, -1);
	Mat m = mergeRows(m1, m2);
	cout << m.rows << endl;

	Ptr< SVM > svm = SVM::create();
	/* Default values to train SVM */
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR); //采用线性核函，其他的sigmoid 和RBF 可自行设置，其值由0-5。  
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?  
	svm->setC(0.01); // From paper, soft classifier  
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task  
	svm->train(m, ROW_SAMPLE, Mat(labels));
	svm->save("C:\\dataset\\data\\svm1.xml");
	clog << "...[done]" << endl;

	int a;
	cin >> a;
}


