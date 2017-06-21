#pragma once

#include "dataset.h"

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>
#include <vector>

class FaceRecognizer
{
public:
	FaceRecognizer();

	void load(const std::string& filename);
	void save(const std::string& filename);
	void train(Dataset&& dataset);
	void update(Dataset& newData);

	const std::vector<cv::Mat>& newImages() const { return m_newImages; };
	const std::vector<label_t>& newLabels() const { return m_newLabels; };
	bool ready() const { return m_ready; }

	std::string predict(cv::Mat image) const;

private:
	bool m_ready = false;
	cv::Ptr<cv::face::FaceRecognizer> m_facerec;
	std::vector<cv::Mat> m_newImages;
	std::vector<label_t> m_newLabels;
};
