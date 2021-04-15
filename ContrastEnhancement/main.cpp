// Copyright 2021 Shulman Egor
#include <stdio.h>
#include "contrast_enhancement.h"
#include <iostream>

int main()
{
	double startSeq = 0., endSeq = 0., startOmp = 0., endOmp = 0.;
	cv::Mat pic = cv::imread("omp.jpg", 0);
	cv::Mat resultSeq, resultOmp;
	cv::imshow("��������", pic);

	startSeq = omp_get_wtime();
	resultSeq = ContrastEnhancement(pic);
	endSeq = omp_get_wtime();
	cv::imshow("����������������", resultSeq);

	startOmp = omp_get_wtime();
	resultOmp = ContrastEnhancementOMP(pic);
	endOmp = omp_get_wtime();
	cv::imshow("������������", resultOmp);

	double timeSeq = endSeq - startSeq;
	double timeOmp = endOmp - startOmp;
	double tick = omp_get_wtick();

	std::cout << "Sequential: " << timeSeq
			  << " Omp: " << timeOmp
		      << " Precision: " << tick
		      << std::endl;

	cv::waitKey(0);
	return 0;
}