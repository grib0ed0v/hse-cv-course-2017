#include "dataset_manager.h"

#include "util/log.h"

#include <opencv2/imgcodecs.hpp>
#include <array>

void DatasetManager::load(const std::string& datasetFolder)
{
	m_datasetFolder = datasetFolder;
	if (!m_datasetFolder.empty() && m_datasetFolder.back() != '/')
		m_datasetFolder += '/';
}

bool DatasetManager::datasetChanged()
{
	return true;
}

Dataset DatasetManager::readDataset()
{
	Dataset data;
	if (m_datasetFolder.empty()) {
		logError() << "Dataset folder not set!";
		return data;
	}

	// TODO: Hardocoded part, need to change it
	struct FolderInfo
	{
		size_t count;
		std::string ext;
		std::string str;
	};
	const size_t folderCount = 4;
	const std::array<FolderInfo, folderCount> folders = {
		FolderInfo{10, "pgm", "Test Test"},
		FolderInfo{7, "jpg", "N P"},
		FolderInfo{10, "pgm", "Foo Bar"},
		FolderInfo{8, "jpg", "A Ya"}
	};

	label_t curLabel = 1;
	size_t curFolder = 1;
	std::vector<cv::Mat> images;
	for (const FolderInfo& folder : folders) {
		for (size_t imgIdx = 1; imgIdx <= folder.count; ++imgIdx) {
			std::string imgname = m_datasetFolder + std::to_string(curFolder) + "/" + std::to_string(imgIdx) + "." + folder.ext;
			cv::Mat image = cv::imread(imgname, cv::IMREAD_GRAYSCALE);
			if (image.data == nullptr) {
				logError() << "Could not read" << imgname;
				continue;
			}
			images.push_back(image);
		}
		data.addImages(curLabel, images);
		data.setLabelString(curLabel, folder.str);
		++curFolder;
		++curLabel;
	}
	return data;
}
