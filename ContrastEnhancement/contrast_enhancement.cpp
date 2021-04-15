// Copyright 2021 Shulman Egor
#include "contrast_enhancement.h"
#include <iostream>

cv::Mat ContrastEnhancement(const cv::Mat& matrix) {
	cv::Mat result(matrix);
	int yMax = 0;
	int yMin = 255;
	int n = matrix.rows;
	int m = matrix.cols;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			yMax = matrix.at<uchar>(i, j) > yMax ? matrix.at<uchar>(i, j) : yMax;
			yMin = matrix.at<uchar>(i, j) < yMin ? matrix.at<uchar>(i, j) : yMin;
		}
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			result.at<uchar>(i, j) = ((matrix.at<uchar>(i, j) - yMin) * (255 / (yMax - yMin)));
		}
	}
	return result;
}

cv::Mat ContrastEnhancementOMP(const cv::Mat& matrix) {
	cv::Mat result(matrix);
	int yMax = 0;
	int yMin = 255;
	int n = matrix.rows;
	int m = matrix.cols;

#pragma omp parallel
	{
		int localMax = 0;
		int localMin = 255;
#pragma omp for
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				localMax = matrix.at<uchar>(i, j) > localMax ? matrix.at<uchar>(i, j) : localMax;
				localMin = matrix.at<uchar>(i, j) < localMin ? matrix.at<uchar>(i, j) : localMin;
			}
		}
#pragma omp critical
		{
			yMax = yMax > localMax ? yMax : localMax;
			yMin = yMin < localMin ? yMin : localMin;
		}
#pragma omp barrier
#pragma omp for
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				result.at<uchar>(i, j) = (matrix.at<uchar>(i, j) - yMin) * (255 / (yMax - yMin));
			}
		}
	}
	return result;
}