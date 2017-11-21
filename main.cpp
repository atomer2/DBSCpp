#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
//#include "boost/smart_ptr.hpp"
#define _USE_MATH_DEFINES                 // for M_PI
#include <math.h>
#include <vector>
#include <iostream>
#include <limits>

using namespace cv;
//using namespace boost;

template<typename T>
std::vector<double> linspace(T start_in, T end_in, int num_in)
{

	std::vector<double> linspaced;

	double start = static_cast<double>(start_in);
	double end = static_cast<double>(end_in);
	double num = static_cast<double>(num_in);

	if (num == 0) { return linspaced; }
	if (num == 1)
	{
		linspaced.push_back(start);
		return linspaced;
	}

	double delta = (end - start) / (num - 1);

	for (int i = 0; i < num - 1; ++i)
	{
		linspaced.push_back(start + delta * i);
	}
	linspaced.push_back(end); // I want to ensure that start and end
							  // are exactly the same as the input
	return linspaced;
}


float nasanenCsf(float u, float v) {
	const static float c = 0.525;
	const static float d = 3.91;
	const static float L = 11;
	const static float constant = -1 / (c*log(L) + d);
	return exp(sqrt(u*u + v*v)*constant);
}

// 15.6寸 1080P 笔记本屏幕的PPI 为141
const int g_dpi = 200;
const float g_viewDistance = 9.5;
const int g_filterSize = 11;
const int g_maxItertionTimes = 1000;

// 计算傅里叶逆变换
void calcIDFT(Mat &src1f, Mat &dst1f) {
	Mat padded;
	//int m = getOptimalDFTSize(src1f.rows);
	//int n = getOptimalDFTSize(src1f.cols);
	int m = src1f.rows;
	int n = src1f.cols;
	copyMakeBorder(src1f, padded, 0, m - src1f.rows, 0, n - src1f.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(),CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	// DFT_INVERSE 逆变换标志
	// DFT_SCALE scales the result: divide it by the number of array elements. Normally, it is combined with DFT_INVERSE.
	dft(complexI, complexI, DFT_INVERSE | DFT_SCALE);

	split(complexI, planes);
	magnitude(planes[0], planes[1], planes[0]);

	dst1f = Mat::zeros(planes[0].size(), CV_32F);

	int cx = planes[0].cols / 2;
	int cy = planes[0].rows / 2;

	// IFFT shift

	Mat q0(planes[0], Rect(0, 0, cx + 1, cy + 1));
	Mat q1(planes[0], Rect(cx + 1, 0, cx, cy + 1));
	Mat q2(planes[0], Rect(0, cy + 1, cx + 1, cy));
	Mat q3(planes[0], Rect(cx + 1, cy + 1, cx, cy));

	Mat p0(dst1f, Rect(0, 0, cx, cy));
	Mat p1(dst1f, Rect(cx, 0, cx + 1, cy));
	Mat p2(dst1f, Rect(0, cy, cx, cy + 1));
	Mat p3(dst1f, Rect(cx, cy, cx + 1, cy + 1));
	
	q3.copyTo(p0);
	q2.copyTo(p1);
	q1.copyTo(p2);
	q0.copyTo(p3);
}

void calcCpp(Mat &dst1f, int filterSize ) {
	assert(filterSize & 1); // filterSize 只能是奇数
	int h = filterSize / 2;
	Mat p(filterSize, filterSize, CV_32F);
	float scale = 180 / (g_dpi*g_viewDistance*M_PI);    // CSF函数的单位是cyc/deg,故应该现进行单位变换
	float upper_bound = 0.5 / scale;
	float lower_bound = -0.5 / scale;
	auto U = linspace(lower_bound, upper_bound, filterSize);
	auto V = linspace(lower_bound, upper_bound, filterSize);
	for (int i = 0; i < filterSize; i++) {
		for (int j = 0; j < filterSize; j++) {
			p.ptr<float>(i)[j] = nasanenCsf(U[i], V[j]);
		}
	}
	Mat spatialFilter;
	calcIDFT(p, spatialFilter);
	// std::cout << "空间滤波器大小：" << spatialFilter.rows << "x" << spatialFilter.cols << std::endl;
	// std::cout << spatialFilter;
	Mat cpp;
	// matchTemplate(spatialFilter, spatialFilter, cpp, CV_TM_CCORR);
	Mat padded;
	copyMakeBorder(spatialFilter, padded, h, h, h, h, BORDER_CONSTANT, Scalar::all(0));

	// need to specify the border type
	filter2D(padded, cpp, -1, spatialFilter, Point(-1, -1), 0, BORDER_REPLICATE);
	//std::cout << "cpp大小" << cpp.rows << "x" << cpp.cols << std::endl;
	//std::cout << cpp;
	dst1f = cpp;

}
// 简单的阈值加网 作为起始图像
void threshold(Mat &src1f, Mat& dst1f, float thd = 0.5) {
	dst1f.create(src1f.size(), CV_32F);
	for (int i = 0; i < src1f.rows; i++) {
		for (int j = 0; j < src1f.cols; j++) {
			if (src1f.ptr<float>(i)[j] < thd) {
				dst1f.ptr<float>(i)[j] = 0;
			}
			else {
				dst1f.ptr<float>(i)[j] = 1;
			}
		}
	}
}

// 直接二值查找加网
void dbs(Mat &src1b, Mat& dst1b) {

	assert(src1b.type() == CV_8UC1);  // 只支持8位单通道灰度图

	// cpp center
	int mx = g_filterSize - 1;
	int my = g_filterSize - 1;

	Mat src1f;
	src1f.create(src1b.size(), CV_32F);
	for (int i = 0; i < src1f.rows; i++) {
		for (int j = 0; j < src1f.cols; j++) {
			// 浮点数除法
			src1f.ptr<float>(i)[j] = src1b.ptr<uchar>(i)[j] / static_cast<float>(std::numeric_limits<uchar>::max());
		}
	}
	Mat H,cpp;
	threshold(src1f, H);
	calcCpp(cpp, g_filterSize);
	Mat error = H - src1f;
	Mat cpe;
	// 要指定border type,默认的有问题
	filter2D(error, cpe, -1, cpp,Point(-1,-1),0,BORDER_CONSTANT);  // Is that right？

	// for fast access
	int r= cpe.rows;
	int c = cpe.cols;
	double *cpeFastData = new double[r*c];
	double **cpeFast = new double*[r];
	//shared_array<double> cpeFastData(new double[r*c]);
	//shared_array<double*> cpeFast(new double*[r]);
	for (int i = 0; i < r; i++) {
		cpeFast[i] = &cpeFastData[i*c];
		for (int j = 0; j < c; j++) {
			cpeFast[i][j] = cpe.ptr<float>(i)[j];
		}
	}

	r = cpp.rows;
	c = cpp.cols;
	double *cppFastData = new double[r*c];
	double **cppFast = new double*[r];
	//shared_array<double> cppFastData(new double[r*c]);
	//shared_array<double*> cppFast(new double*[r]);
	for (int i = 0; i < r; i++) {
		cppFast[i] = &cppFastData[i*c];
		for (int j = 0; j < c; j++) {
			cppFast[i][j] = cpp.ptr<float>(i)[j];
		}
	}

	r = H.rows;
	c = H.cols;
	double *HFastData = new double[r*c];
	double **HFast = new double*[r];
	//shared_array<double> HFastData(new double[r*c]);
	//shared_array<double*> HFast(new double*[r]);
	for (int i = 0; i < r; i++) {
		HFast[i] = &HFastData[i*c];
		for (int j = 0; j < c; j++) {
			HFast[i][j] = H.ptr<float>(i)[j];
		}
	}

	// begin swap and toggle
	int iter = 0;
	const int neighbor[8][2] = { -1,-1, -1,0, -1,1, 0,-1, 0,1, 1,-1, 1,0, 1,1 };
	while (iter < g_maxItertionTimes) {
		iter++;
		int benefit_points = 0;
		int r = H.rows;
		int c = H.cols;
		for (int i = 0; i < r; i++) {
			for (int j = 0; j < c; j++) {
				int minIdx = 9;
				// toggle
				float a0 = 1 - 2 * HFast[i][j];
				float a1 = -a0;
				float delta = a0*a0 * cppFast[mx][my] + 2 * a0*cpeFast[i][j];
				float minDelta = delta;
				// swap
				for (int k = 0; k < 8; k++) {
					int ni = i + neighbor[k][0];
					int nj = j + neighbor[k][1];
					if (ni >= 0 && ni < r && nj >= 0 && nj < c && HFast[i][j] != HFast[ni][nj]) {
						float de = delta + a1*a1*cppFast[mx][my] + 2 * a0*a1*cppFast[mx + neighbor[k][0]][my + neighbor[k][1]] + 2 * a1*cpeFast[ni][nj];
						if (de < minDelta) {
							minDelta = de;
							minIdx = k;
						}
					}
				}
				if (minDelta < 0) {
					benefit_points++;
					// update cpe
					for (int m=0; m < cpp.rows; m++) {
						for (int n = 0; n < cpp.cols; n++) {
							int mm = m - mx + i;
							int nn = n - my + j;
							if (mm >= 0 && mm < r && nn >= 0 && nn < c) {
								cpeFast[mm][nn] += a0*cppFast[m][n];
							}
						}
					}
					HFast[i][j] += a0;
					if (minIdx != 9) { // swap
						int ni = i + neighbor[minIdx][0];
						int nj = j + neighbor[minIdx][1];
						HFast[ni][nj] += a1;
						// update cpe
						for (int m = 0; m < cpp.rows; m++) {
							for (int n = 0; n < cpp.cols; n++) {
								int mm = m - mx + ni;
								int nn = n - my + nj;
								if (mm >= 0 && mm < r && nn >= 0 && nn < c) {
									cpeFast[mm][nn] += a1*cppFast[m][n];
								}
							}
						}
					}
				}
			}
		}
	    std::cout << "{ " << iter << "th iteration } benefit points is " << benefit_points << std::endl;
		if (benefit_points == 0)
			break;
	}
	dst1b.create(src1b.size(), CV_8UC1);
	for (int i = 0; i < src1b.rows; i++) {
		for (int j = 0; j < src1b.cols; j++) {
			dst1b.ptr<uchar>(i)[j] =  HFast[i][j] * std::numeric_limits<uchar>::max();
		}
	}
	// 释放内存
	delete[] cppFast;
	delete[] cppFastData;
	delete[] cpeFast;
	delete[] cpeFastData;
	delete[] HFast;
	delete[] HFastData;
}

int main() {
	//Mat src(256, 256, CV_8U, Scalar::all(230));
	Mat dst;
	Mat src = imread("lena_src.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	if (!src.empty()) {
		dbs(src, dst);
		imwrite("lena_dst.bmp", dst);
		imshow("src Image", src);
		imshow("dst Image", dst);
		waitKey(0);
	}
	else {
		std::cout << "Image Open Failed!! " << std::endl;
		getchar();
	}
	return 0;
}
