#pragma once

#include "dataset.h"

#include <opencv2/opencv.hpp>

#include <string>


class DatasetManager
{
public:
	void readConfig(const std::string& configPath);
	void saveConfig(const std::string& configPath);

	void load(const std::string& datasetFolder);
	bool datasetChanged();
	Dataset readDataset();

private:
	std::string m_datasetFolder;
	bool m_datasetChanged = false;
};
