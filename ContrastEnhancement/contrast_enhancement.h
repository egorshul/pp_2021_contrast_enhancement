// Copyright 2021 Shulman Egor
#ifndef CONTRAST_ENHANCEMENT_H_
#define CONTRAST_ENHANCEMENT_H_

#include <vector>
#include <omp.h>
#include "opencv2/highgui/highgui.hpp"

void printHistogram(const cv::Mat& matrix);
cv::Mat ContrastEnhancement(const cv::Mat& matrix);
cv::Mat ContrastEnhancementOMP(const cv::Mat& matrix);
cv::Mat ContrastEnhancementTBB(const cv::Mat& matrix);

#endif // CONTRAST_ENHANCEMENT_H_