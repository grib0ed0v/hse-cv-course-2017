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
	m_dataset = dataset;
	if (m_dataset.images().empty()) {
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
	int oldLabelCount = m_dataset.labelCount();
	std::vector<label_t> labels = m_dataset.addDataset(newData);

	m_facerec->update(newData.images(), labels);
	for (int label = oldLabelCount; label < (int)m_dataset.labelCount(); ++label){
		m_facerec->setLabelInfo(label, m_dataset.stringByLabel(label));
	}
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
