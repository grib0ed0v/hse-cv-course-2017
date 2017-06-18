#pragma once

#include "dataset.h"
#include <opencv2/core.hpp>
#include <string>


class DatasetManager
{
public:
	DatasetManager(const std::string& configPath)
	{
		(void)configPath;
	}

	void load(const std::string& datasetFolder);
	bool datasetChanged();
	Dataset readDataset();

private:
	std::string m_datasetFolder;
};
