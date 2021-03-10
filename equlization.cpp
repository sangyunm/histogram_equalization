#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "CUDA_equalization.h"

using namespace std;
using namespace cv;

int main() {

	Mat input_image = imread("C:/Users/sxm41/Pictures/low_contrast.bmp", 0);
	 
	Histogram_equalization_cuda(input_image.data, input_image.rows, input_image.cols, input_image.channels());

	imwrite("Histogram_image.png", input_image);
 
	system("pause");

	return 0;
}