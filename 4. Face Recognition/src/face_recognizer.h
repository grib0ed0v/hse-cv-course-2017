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

	bool ready() const { return m_ready; }

	std::string predict(cv::Mat image) const;

private:
	bool m_ready = false;
	cv::Ptr<cv::face::FaceRecognizer> m_facerec;
	Dataset m_dataset;
};
