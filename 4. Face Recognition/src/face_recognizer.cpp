#include "face_recognizer.h"

#include "util/log.h"

FaceRecognizer::FaceRecognizer()
{
	m_facerec = cv::face::createLBPHFaceRecognizer();
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
	int modelLabelCount = m_facerec->getLabelsByString("").size();
	int newModelLabelCount = modelLabelCount;
 
	for (int label = 0; label < newData.labelCount(); ++label) {
		info[label] = newData.stringByLabel(label);
		std::vector<int> modelLabels = m_facerec->getLabelsByString(info[label]);

		for (auto& x: modelLabels) {
			if (m_facerec->getLabelInfo(x) == info[label]) {
				newLabels[label] = x;
				break;
			}
		}
		if (newLabels[label] == -1) {
			newLabels[label] = newModelLabelCount++;
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

	m_newImages.insert(m_newImages.end(), newData.images().begin(), newData.images().end());
	m_newLabels.insert(m_newLabels.end(), updatedLabels.begin(), updatedLabels.end());	
}

std::string FaceRecognizer::predict(cv::Mat image) const
{
	double confidence = 0.0;
	label_t label = -1;
	m_facerec->predict(image, label, confidence);
	std::string s = m_facerec->getLabelInfo(label);
	s += " ";
	s += std::to_string(confidence);
	return s;
}
