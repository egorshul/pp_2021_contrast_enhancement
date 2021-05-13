// Copyright 2021 Shulman Egor
#include <stdio.h>
#include "contrast_enhancement.h"
#include <iostream>

int main()
{
	cv::Mat resultSeq, resultOmp, resultTbb, resultStd;
	
	cv::Mat pic = cv::imread("test.jpg", 0);
	cv::imshow("Исходное", pic);
	printHistogram(pic, "Histogram");

	resultSeq = ContrastEnhancement(pic);
	cv::imshow("SEQ", resultSeq);
	printHistogram(resultSeq, "HistogramSeq");

	resultOmp = ContrastEnhancementOMP(pic);
	cv::imshow("OMP", resultOmp);
	printHistogram(resultOmp, "HistogramOmp");

	resultTbb = ContrastEnhancementTBB(pic);
	cv::imshow("TBB", resultTbb);
	printHistogram(resultTbb, "HistogramTbb");

	resultStd = ContrastEnhancementTBB(pic);
	cv::imshow("STD", resultStd);
	printHistogram(resultStd, "HistogramStd");

	cv::waitKey(0);
	return 0;
}