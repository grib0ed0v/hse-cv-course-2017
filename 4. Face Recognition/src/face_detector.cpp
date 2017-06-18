#include "face_detector.h"

#include <opencv2/imgproc.hpp>

FaceDetector::FaceDetector(const std::string& configPath)
{
	std::string path = configPath;
	if (path.back() != '/')
		path += '/';
	path += "haarcascade_frontalface_default.xml";
	m_classifier.load(path);
}

std::vector<FaceDetector::FaceRegion> FaceDetector::detect(cv::Mat img)
{
	cv::Mat gray;
	cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	std::vector<cv::Rect> faceRects;
	m_classifier.detectMultiScale(gray, faceRects);
	std::vector<FaceRegion> faces;
	faces.reserve(faceRects.size());
	for (const cv::Rect& faceRect : faceRects) {
		faces.push_back({ faceRect, cv::Mat(gray, faceRect) });
	}
	return faces;
}
