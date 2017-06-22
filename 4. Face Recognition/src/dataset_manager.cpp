#include "dataset_manager.h"

#include "util/log.h"
#include "util/fsutil.h"

#include <opencv2/imgcodecs.hpp>
#include <array>

void DatasetManager::readConfig(const std::string& configPath)
{
	cv::FileStorage fileStorage(configPath, cv::FileStorage::READ);
	if (fileStorage.isOpened()) {
		fileStorage["datasetFolder"] >> m_datasetFolder;
	}
}

void DatasetManager::saveConfig(const std::string& configPath)
{
	cv::FileStorage fileStorage(configPath, cv::FileStorage::WRITE);
	fileStorage << "datasetFolder" << m_datasetFolder;
}

void DatasetManager::load(const std::string& datasetFolder)
{
	if (m_datasetFolder != datasetFolder) {
		m_datasetChanged = true;
		m_datasetFolder = datasetFolder;
	}
	if (datasetFolder.empty()) {
		logError() << "Empty string passed to DatasetManager::load";
		return;
	}
}

bool DatasetManager::datasetChanged()
{
	return m_datasetChanged;
}

Dataset DatasetManager::readDataset()
{
	Dataset data;
	if (m_datasetFolder.empty()) {
		logError() << "Dataset folder not set!";
		return data;
	}

	label_t curLabel = 1;
	std::vector<cv::Mat> images;
	std::vector<std::string> datasetContent = fs::getFilesInDir(m_datasetFolder);
	auto removeIt = std::remove_if(datasetContent.begin(), datasetContent.end(),
	[this](const std::string& path)
	{
		return !fs::isDir(fs::concatPath(m_datasetFolder, path));
	});
	datasetContent.erase(removeIt, datasetContent.end());

	if (datasetContent.empty()) {
		logError() << "Empty datset:" << m_datasetFolder;
	}

	for (const std::string& folderName : datasetContent) {
		images.clear();

		std::string folderPath = fs::concatPath(m_datasetFolder, folderName);
		std::vector<std::string> folderContent = fs::getFilesInDir(folderPath);
		if (folderContent.empty()) {
			logWarning() << "Empty folder:" << folderPath;
			continue;
		}
		for (const std::string imgName : folderContent) {
			std::string imgPath = fs::concatPath(folderPath, imgName);
			cv::Mat image = cv::imread(imgPath, cv::IMREAD_GRAYSCALE);
			if (image.data == nullptr) {
				logError() << "Could not read" << imgPath;
				continue;
			}
			images.push_back(image);
		}
		data.addImages(curLabel, images);
		data.setLabelString(curLabel, folderName);
		++curLabel;
	}
	return data;
}
