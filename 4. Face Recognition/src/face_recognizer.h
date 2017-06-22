#pragma once

#include "dataset.h"

#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <string>
#include <vector>

class FaceRecognizer
{
public:
	FaceRecognizer(const std::string& configPath);

	void load(const std::string& filename);
	void save(const std::string& filename);
	void train(Dataset&& dataset);
	void update(Dataset& newData);

	const Dataset& newData() const { return m_newData; };
	bool ready() const { return m_ready; }

	std::string predict(cv::Mat image) const;

private:
	bool m_ready = false;
	cv::Ptr<cv::face::FaceRecognizer> m_facerec;
	Dataset m_newData;

	struct Config {
		int radius = 1;
		int neighbors = 8;
		int grid_x = 8;
		int grid_y = 8;
		double threshold = DBL_MAX;

		bool write(const std::string& path) const
		{
			cv::FileStorage fs(path, cv::FileStorage::WRITE);
			if (!fs.isOpened()) {
				return false;
			}
			fs << "radius" << radius;
			fs << "neighbors" << neighbors;
			fs << "grid_x" << grid_x;
			fs << "grid_y" << grid_y;
			fs << "threshold" << threshold;
			return true;
		}

		bool read(const std::string& path)
		{
			cv::FileStorage fs(path, cv::FileStorage::READ);
			if (!fs.isOpened()) {
				return false;
			}
			radius = fs["radius"];
			neighbors = fs["neighbors"];
			grid_x = fs["grid_x"];
			grid_y = fs["grid_y"];
			threshold = fs["threshold"];
			return true;
		}
	};

	Config m_config;
	
};
