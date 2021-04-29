// Copyright 2021 Shulman Egor
#include <stdio.h>
#include "contrast_enhancement.h"
#include <iostream>

int main()
{
	cv::Mat resultSeq, resultOmp, resultTbb;
	
	cv::Mat pic = cv::imread("test.jpg", 0);
	cv::imshow("Исходное", pic);
	printHistogram(pic);

	resultSeq = ContrastEnhancement(pic);
	cv::imshow("Последовательное", resultSeq);

	resultOmp = ContrastEnhancementOMP(pic);
	cv::imshow("OMP", resultOmp);

	resultTbb = ContrastEnhancementTBB(pic);
	cv::imshow("TBB", resultTbb);
	printHistogram(resultTbb);

	cv::waitKey(0);
	return 0;
}