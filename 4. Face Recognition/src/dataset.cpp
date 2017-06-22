#include "dataset.h"
#include <iterator>

void Dataset::addImages(label_t label, const std::vector<cv::Mat>& images)
{
	if (label < 0) return;
	ensureLabel(label);
	m_images.insert(m_images.end(), images.begin(), images.end());
	std::fill_n(std::back_inserter(m_labels), images.size(), label);
}

void Dataset::addImages(const std::string& str, const std::vector<cv::Mat>& images)
{
	label_t strLabel = labelByString(str);
	if (strLabel < 0) {
		strLabel = labelCount();
		setLabelString(strLabel, str);
	}
	addImages(strLabel, images);
}

std::vector<label_t> Dataset::addDataset(const Dataset& new_data)
{
	std::vector<label_t> new_labels(new_data.labels().size());
	for (int i = 0; i < (int)new_labels.size(); ++i){
		std::string str = new_data.stringByLabel(new_data.labels()[i]);
		label_t label = labelByString(str);
		if (label < 0){
			label = labelCount();
			setLabelString(label, str);
		}
		new_labels[i] = label;
	}
	m_images.insert(m_images.end(), new_data.images().begin(), new_data.images().end());
	m_labels.insert(m_labels.end(), new_labels.begin(), new_labels.end());
	return new_labels;
}

void Dataset::setLabelString(label_t label, const std::string& str)
{
	if (label < 0) return;
	ensureLabel(label);
	if (!m_labelToString[label].empty())
		m_stringToLabel.erase(m_labelToString[label]);
	m_labelToString[label] = str;
	m_stringToLabel[str] = label;
}

label_t Dataset::labelByString(const std::string& str) const
{
	auto labelIt = m_stringToLabel.find(str);
	return labelIt != m_stringToLabel.end() ? labelIt->second : -1;
}

const std::string& Dataset::stringByLabel(label_t label) const
{
	static std::string empty;
	return 0 <= label && label < (int)m_labelToString.size() ? m_labelToString[label] : empty;
}

void Dataset::ensureLabel(label_t label)
{
	if (label >= (label_t)m_labelToString.size())
		m_labelToString.resize(label + 1);
}