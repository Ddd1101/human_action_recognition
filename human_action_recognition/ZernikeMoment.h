#pragma once
#include <iostream>  
#include "opencv2/opencv.hpp"  
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


using namespace cv;
using namespace std;

#define Pi  3.1415926



typedef struct
{
	float    rou;
	float    sigma;
}
RS_POINT;

typedef struct
{
	float    x;
	float    y;
}CARTESIAN_POINT;


typedef struct
{
	float    r;
	float    t;
}POLAR_POINT;


typedef struct
{
	float    Cnm;
	float    Snm;
}ZERNIKE;


class ZernikeMoment {

private:
	IplImage * oriImg;
	IplImage*    grayImg;
	float        Cnm, Snm;
	float        Z_mode;


public:

	ZernikeMoment();
	~ZernikeMoment();

	float               getRpqr(float p, float q, float r);
	RS_POINT*           get_rou_sigma(float x, float y);
	CARTESIAN_POINT*    get_x_y(float r, float s, int N);

	int                 Img2Gray(void);
	void                 ClearOpenCV(void);

	float               get_8_XYValue(int x, int y);
	float               get_32_XYValue(int x, int y);

	float               getZernike(int n, int m);


	void Caculate_8_Zernike(int n, int m);
	void Caculate_32_Zernike(int n, int m);

	void readImg(Mat& mask);

};