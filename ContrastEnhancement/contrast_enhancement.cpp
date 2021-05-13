// Copyright 2021 Shulman Egor
#include "tbb/parallel_for.h"
#include "contrast_enhancement.h"
#include <thread>
#include <iostream>

void printHistogram(const cv::Mat& matrix, std::string name) {
	int bins = 255;
	int histSize[] = { bins };

	float lranges[] = { 0, 255 };
	const float* ranges[] = { lranges };

	cv::Mat hist;
	int channels[] = { 0 };

	int const hist_height = 255;
	cv::Mat3b hist_image = cv::Mat3b::zeros(hist_height, bins);

	cv::calcHist(&matrix, 1, channels, cv::Mat(), hist, 1, histSize, ranges, true, false);

	double max_val = 0;
	minMaxLoc(hist, 0, &max_val);

	for (int b = 0; b < bins; b++) {
		float const binVal = hist.at<float>(b);
		int const height = cvRound(binVal * hist_height / max_val);
		cv::line(hist_image, cv::Point(b, hist_height - height), cv::Point(b, hist_height), cv::Scalar::all(255));
	}
	cv::imshow(name, hist_image);
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

cv::Mat ContrastEnhancementSTD(const cv::Mat& matrix) {
	cv::Mat result = matrix.clone();
	std::vector<int> res(matrix.rows * matrix.cols);
	int it = 0;
	for (int i = 0; i < matrix.rows; ++i) {
		for (int j = 0; j < matrix.cols; ++j) {
			res[it++] = matrix.at<uchar>(i, j);
		}
	}
	auto temp = std::minmax_element(res.begin(), res.end());
	int yMax = *(temp.second);
	int yMin = *(temp.first);

	auto threadFill = [&](std::vector<int>::iterator first,
		std::vector<int>::iterator last) {
			for (auto it = first; it != last; ++it) {
				*it = ((*it - yMin) * 255) / (yMax - yMin);
			}
	};

	const size_t size = std::thread::hardware_concurrency();
	std::vector<std::thread> threads(size);
	auto block = res.size() / size;
	auto work_iter = std::begin(res);

	for (auto it = std::begin(threads); it != std::end(threads) - 1; ++it) {
		*it = std::thread(threadFill, work_iter, work_iter + block);
		work_iter += block;
	}
	threads.back() = std::thread(threadFill, work_iter, std::end(res));

	for (auto&& i : threads) {
		i.join();
	}

	it = 0;
	for (int i = 0; i < matrix.rows; ++i) {
		for (int j = 0; j < matrix.cols; ++j) {
			result.at<uchar>(i, j) = res[it++];
		}
	}
	return result;
}
