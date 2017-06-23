#include "face_recognizer.h"

#include "util/fsutil.h"
#include "util/log.h"

FaceRecognizer::FaceRecognizer(const std::string& configPath)
{
	if (!fs::pathExists(configPath)) {
		m_config.write(configPath);
	} else {
		if (!m_config.read(configPath)) {
			logWarning() << "FaceRecognizer config exists, but can't be read from" << configPath;
		}
	}
	m_facerec = cv::face::createLBPHFaceRecognizer(m_config.radius, m_config.neighbors, m_config.grid_x, m_config.grid_y, m_config.threshold);
}

void FaceRecognizer::load(const std::string& filename)
{
	m_facerec->load(filename);
	m_ready = true;
}

void FaceRecognizer::save(const std::string& filename)
{
	m_facerec->save(filename);
}

void FaceRecognizer::train(Dataset&& dataset)
{
	if (dataset.images().empty()) {
		logError() << "Empty dataset passed to FaceRecognizer";
		return;
	}

	m_facerec->train(dataset.images(), dataset.labels());
	for (int label = 0; label < (int)dataset.labelCount(); ++label) {
		m_facerec->setLabelInfo(label, dataset.stringByLabel(label));
	}
	m_ready = true;
}

void FaceRecognizer::update(Dataset& newData)
{
	std::vector<label_t> newLabels(newData.labelCount(), -1);
	std::vector<std::string> info(newData.labelCount());
	int modelLabelCount = -1;
	
	for (int label = 0; label < newData.labelCount(); ++label) {
		info[label] = newData.stringByLabel(label);
		label_t newLabel = 0;
		while (newLabels[label] == -1) {
			std::string str = m_facerec->getLabelInfo(newLabel);
			if (str == info[label]) {
				newLabels[label] = newLabel;
			}
			else if (str == "") {
				newLabels[label] = newLabel;
				if (modelLabelCount == -1) {
					modelLabelCount = newLabel;
				}
			}
			newLabel++;
		}
	}

	std::vector<label_t> updatedLabels(newData.labels().size());
	for (size_t i = 0; i < updatedLabels.size(); ++i) {
		updatedLabels[i] = newLabels[ newData.labels()[i] ];
	}

	m_facerec->update(newData.images(), updatedLabels);
	for (size_t i = 0; i < newLabels.size(); ++i) {
		if (newLabels[i] >= modelLabelCount) {
			m_facerec->setLabelInfo(newLabels[i], info[i]);
		}
	}

	for (size_t i = 0; i < newData.images().size(); ++i) {
		m_newData.addImage(newData.stringByLabel(newData.labels()[i]), newData.images()[i]); 
	}
}

std::string FaceRecognizer::predict(cv::Mat image) const
{
	double confidence = 0.0;
	label_t label = -1;
	m_facerec->predict(image, label, confidence);
	if (label == -1) {
		return "Unknown";
	}
	std::string s = m_facerec->getLabelInfo(label);
	//s += " ";
	//s += std::to_string(confidence);
	return s;
}
