// Copyright 2021 Shulman Egor
#include "contrast_enhancement.h"

void ContrastEnhancement(cv::Mat& matrix) {
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
			matrix.at<uchar>(i, j) = (matrix.at<uchar>(i, j) - yMin) * (255 / (yMax - yMin));
		}
	}
}