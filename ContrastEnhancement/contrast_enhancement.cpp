// Copyright 2021 Shulman Egor
#include "tbb/parallel_for.h"
#include "contrast_enhancement.h"
#include <iostream>

void printHistogram(const cv::Mat& matrix) {
	std::vector<int> histogram(256);
	for (int i = 0; i < matrix.rows; i++) {
		for (int j = 0; j < matrix.cols; j++) {
			histogram[matrix.at<uchar>(i, j)]++;
		}
	}
	std::cout << "-----------------------------------------------------------------------" << std::endl;
	for (int i = 0; i < histogram.size(); i++) {
		std::cout << "[" << i << "]: " << histogram[i] << std::endl;
	}
	std::cout << "-----------------------------------------------------------------------" << std::endl;
}

cv::Mat ContrastEnhancement(const cv::Mat& matrix) {
	cv::Mat result = matrix.clone();
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
			result.at<uchar>(i, j) = ((matrix.at<uchar>(i, j) - yMin) * 255) / (yMax - yMin);
		}
	}
	return result;
}

cv::Mat ContrastEnhancementOMP(const cv::Mat& matrix) {
	cv::Mat result = matrix.clone();
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
				result.at<uchar>(i, j) = ((matrix.at<uchar>(i, j) - yMin) * 255) / (yMax - yMin);
			}
		}
	}
	return result;
}

cv::Mat ContrastEnhancementTBB(const cv::Mat& matrix) {
	cv::Mat result = matrix.clone();
	std::vector<uchar> res(matrix.rows * matrix.cols);
	int it = 0;
	for (int i = 0; i < matrix.rows; ++i) {
		for (int j = 0; j < matrix.cols; ++j) {
			res[it++] = matrix.at<uchar>(i, j);
		}
	}
	int yMax = *std::max_element(res.begin(), res.end());
	int yMin = *std::min_element(res.begin(), res.end());
	tbb::parallel_for(0, static_cast<int>(res.size()), [&](const int i) {
		res[i] = ((res[i] - yMin) * 255) / (yMax - yMin);
	});
	it = 0;
	for (int i = 0; i < matrix.rows; ++i) {
		for (int j = 0; j < matrix.cols; ++j) {
			result.at<uchar>(i, j) = res[it++];
		}
	}
	return result;
}
