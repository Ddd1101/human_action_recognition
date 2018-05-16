#include <math.h>
#include "ZernikeMoment.h"
#include <iostream>  
using namespace std;

double factorial(long n)//////////////
{
	if (n < 0)
		return(0.0);
	if (n == 0)
		return(1.0);
	else
		return(n * factorial(n - 1));
}

float zernikeR(int n, int l, float r)/////////////////
{
	int m;
	float sum = 0.0;

	if (((n - l) % 2) != 0)
	{
		cout << "zernikeR(): improper values of n,l\n";
		return(0);
	}

	for (m = 0; m <= (n - l) / 2; m++)
	{
		sum += (pow((float)-1.0, (float)m)) * (factorial(n - m)) /
			(factorial(m) * (factorial((n - 2 * m + l) / 2)) *
			(factorial((n - 2 * m - l) / 2))) *
				(pow((float)r, (float)(n - 2 * m)));
	}

	return(sum);
}


ZernikeMoment::ZernikeMoment()
{

	oriImg = NULL;

	grayImg = NULL;

	Z_mode = 0;

}



ZernikeMoment::~ZernikeMoment()
{

}



//Fast Compute  Zernike Polynomials Rpq(r)
float ZernikeMoment::getRpqr(float p, float q, float r)//////////////////////
{
	/****  verify that p-q is even  ***


	**********************************/
	float Rpqr = 0;


	float Bppp = 1;


	int times = (p - q) / 2;
	int numbers = times + 1;


	float  Bpqp = pow((p + q) / (p - q + 2), times);
	float* Bpqk = new float[numbers];

	Bpqk[0] = Bpqp;

	int k = (int)p;
	int t = (int)p;


	//Bpqk[0] is Bpqp , Bpqk[1] is Bpq(p-2) ... Bpqk[numbers-1] is Bpqq respectively

	for (int i = 0; i < (numbers - 1); i++)
	{
		float  coeff = ((k + q)*(k - q)) / ((k + p)*(p - k + 2));
		Bpqk[i + 1] = (-1)*(Bpqk[i])*coeff;
		k = k - 2;
	}



	int temp = numbers - 1;


	//Compute Rpqr
	for (k = (int)q; k <= t; k = k + 2)
	{

		Rpqr = Rpqr + (Bpqk[temp])*(pow(r, k));

		temp--;
	}


	delete[]  Bpqk;

	float  a = Rpqr;
	float  b = Rpqr;

	return Rpqr;

}


//Ordinary Compute Rpqr



//convert to  rou, sigma coordinate,this function was never used.
RS_POINT* ZernikeMoment::get_rou_sigma(float x, float y)
{

	RS_POINT   *rs_p = new RS_POINT();

	float  rou = (abs(x) > abs(y) ? abs(x) : abs(y));

	float  sigma;


	if (abs(x) == rou)

		sigma = (2 * (rou - x)*y) / abs(y) + x * y / rou;

	if (abs(y) == rou)

		sigma = 2 * y - x * y / rou;

	rs_p->rou = rou;
	rs_p->sigma = sigma;


	return  rs_p;

}


//Convert rou-sigma to x-y coordinate,also never used.
CARTESIAN_POINT* ZernikeMoment::get_x_y(float rou, float sigma, int N)
{



	float r = 2 * rou / N;
	float t = (Pi*sigma) / 4 * rou;

	CARTESIAN_POINT *xy_point = new CARTESIAN_POINT();

	xy_point->x = r * cos(t);
	xy_point->y = r * sin(t);

	return  xy_point;
}





//Get the x,y Pixel value of Image ,8 depths
float ZernikeMoment::get_8_XYValue(int x, int y)//////////////////
{
	int height = grayImg->height;
	int widthStep = grayImg->widthStep;
	char*  Data = grayImg->imageData;

	uchar  c_value = ((uchar *)(Data + x * widthStep))[y];

	float  value = (float)c_value;

	return value;


}
//Get the x,y Pixel value of Image ,32 depths
float ZernikeMoment::get_32_XYValue(int x, int y)/////////////////
{
	int height = grayImg->height;
	int widthStep = grayImg->widthStep;
	char*  Data = grayImg->imageData;

	float  value = ((float *)(Data + x * widthStep))[y];

	return value;


}





//RGB to Gray
//本函数部分代码被注释了，因为改为直接读入灰度图了
int  ZernikeMoment::Img2Gray() {
	int a = 1;
	//if((oriImg = cvLoadImage("E:\\XH.jpg", 1)) != 0   )
	if ((grayImg = cvLoadImage("C:\\Users\\dell\\Desktop\\测试用图\\T5.bmp", 1)) != 0)//读入一张灰度图


	{ //grayImg  = cvCreateImage(cvSize(oriImg->width,oriImg->height),IPL_DEPTH_8U,1);


	  //cvCvtColor(oriImg,grayImg,CV_BGR2GRAY);

		return 1;
	}

	return 0;

}


//Cleanning Work,release memory,etc
void ZernikeMoment::ClearOpenCV(void)///////////////
{

	if (oriImg != NULL) {
		cvReleaseImage(&oriImg);
		oriImg = NULL;

	}


	if (grayImg != NULL) {

		cvReleaseImage(&grayImg);

		grayImg = NULL;

	}




}


//Function to caculate Zernike_8_(n,m), a very important function.
void ZernikeMoment::Caculate_8_Zernike(int n, int m)////////////////////
{
	int height = grayImg->height;
	int widthStep = grayImg->widthStep;

	float N = ((float)height < (float)widthStep ? (float)height : (float)widthStep);

	//float N2 = N / 2;



	float Rpqr_C = 0;
	float Rpqr_S = 0;





	for (float r = 1; r < N; r++)
	{

		float temp_C = 0;
		float temp_S = 0;


		for (float s = 1; s <= 8 * r; s++)
		{

			float xy_v = get_8_XYValue(r, s);

			temp_C = temp_C + cos((Pi*m*s) / (4 * r))*xy_v;
			temp_S = temp_S + sin((Pi*m*s) / (4 * r))*xy_v;

		}

		//float Rpqr   =    getRpqr(n,m,(2*r)/N);
		float Rpqr = zernikeR(n, m, (2 * r) / N);
		Rpqr_C = Rpqr_C + temp_C * Rpqr;
		Rpqr_S = Rpqr_S + temp_S * Rpqr;

	}



	Cnm = Rpqr_C * (2 * n + 2) / pow(N, 2);
	Snm = Rpqr_S * (2 * n + 2) / pow(N, 2);

	float l_c = pow(Cnm, 2);
	float l_s = pow(Cnm, 2);
	float l_p = l_c + l_s;

	Z_mode = pow((float)l_p, (float)0.5);
}



//Function to caculate Zernike_32_(n,m), a very important function.

void ZernikeMoment::Caculate_32_Zernike(int n, int m)////////////
{
	int height = grayImg->height;
	int widthStep = grayImg->widthStep;

	float N = ((float)height < (float)widthStep ? (float)height : (float)widthStep);

	float N2 = N / 2;



	float Rpqr_C = 0;
	float Rpqr_S = 0;





	for (float r = 1; r < N2; r++)
	{

		float temp_C = 0;
		float temp_S = 0;


		for (float s = 1; s <= 8 * r; s++)
		{

			float xy_v = get_32_XYValue(r, s);

			temp_C += cos((Pi*m*s) / (4 * r))*xy_v;
			temp_S += sin((Pi*m*s) / (4 * r))*xy_v;


		}

		float     Rpqr = getRpqr(n, m, (2 * r) / N);
		Rpqr_C = Rpqr_C + temp_C * Rpqr;
		Rpqr_S = Rpqr_S + temp_S * Rpqr;

	}



	Cnm = Rpqr_C * (2 * n + 2) / pow(N, 2);
	Snm = Rpqr_S * (2 * n + 2) / pow(N, 2);

	Z_mode = pow((float)pow(Cnm, 2) + pow(Cnm, 2), (float)0.5);
}



float ZernikeMoment::getZernike(int n, int m)//////////////
{

	/*int pass = Img2Gray();

	if (!pass) {
	return -1;
	}*/
	int depth = 0;
	int nChannels = 0;
	//cout << grayImg->nChannels << endl;
	//nChannels = grayImg->nChannels;

	//    if(nChannels!=1)
	// 	   return -1;

	depth = grayImg->depth;


	switch (depth)
	{

	case   IPL_DEPTH_8U:   Caculate_8_Zernike(n, m);    break;
	case   IPL_DEPTH_32F:  Caculate_32_Zernike(n, m);   break;
	default:     break;
	}


	//ClearOpenCV();


	return  Z_mode;


}

void ZernikeMoment::readImg(Mat& mask) {
	//Mat img = imread(filePath);
	grayImg = &IplImage(mask);
}