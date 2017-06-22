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
	label_t strLabel = assignLabel(str);
	addImages(strLabel, images);
}

void Dataset::addImage(label_t label, const cv::Mat& image)
{
	if (label < 0) return;
	ensureLabel(label);
	m_images.push_back(image);
	m_labels.push_back(label);
}

void Dataset::addImage(const std::string& str, const cv::Mat& image)
{
	label_t strLabel = assignLabel(str);
	addImage(strLabel, image);
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

label_t Dataset::assignLabel(const std::string& str)
{
	label_t label = labelByString(str);
	if (label < 0) {
		label = labelCount();
		setLabelString(label, str);
	}
	return label;
}
