#pragma once

#include <opencv2/opencv.hpp>

class FaceDetector
{
public:
	struct FaceRegion
	{
		cv::Rect rect;
		cv::Mat image;
	};

	FaceDetector(const std::string& configPath);
	std::vector<FaceRegion> detect(cv::Mat img);

private:
	cv::Mat processFace(cv::Mat img);
	void applyMask(cv::Mat& img);
	void denoise(cv::Mat& img);
	cv::Point2i detectEye(cv::CascadeClassifier& classifier, cv::Mat img);
	void geometryTransform(cv::Mat& img);
	void normalizeIllumination(cv::Mat& img);

	cv::CascadeClassifier m_classifier;
	//cv::CascadeClassifier m_eyeClassifier;
	cv::CascadeClassifier m_leftEyeClassifier;
	cv::CascadeClassifier m_rightEyeClassifier;
	cv::CascadeClassifier m_eyeglassClassifier;
};
