// Copyright 2021 Shulman Egor
#include <stdio.h>
#include "contrast_enhancement.h"

int main()
{
	cv::Mat pic = cv::imread("test.jpg", 0);
	cv::imshow("Before", pic);
	ContrastEnhancement(pic);
	cv::imshow("After", pic);
	cv::waitKey(0);
	return 0;
}