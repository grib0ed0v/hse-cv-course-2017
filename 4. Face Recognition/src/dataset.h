#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <map>

typedef int label_t;

class Dataset
{
public:
	void addImages(label_t label, const std::vector<cv::Mat>& images);
	void addImages(const std::string& str, const std::vector<cv::Mat>& images);
	void setLabelString(label_t label, const std::string& str);
	
	int labelCount() const { return (int)m_labelToString.size(); }

	const std::vector<cv::Mat>& images() const { return m_images; }
	const std::vector<label_t>& labels() const { return m_labels; }

	int labelByString(const std::string& str) const;
	const std::string& stringByLabel(label_t label) const;

private:
	void ensureLabel(label_t label);

	std::vector<cv::Mat> m_images;
	std::vector<label_t> m_labels;
	std::vector<std::string> m_labelToString;
	std::map<std::string, label_t> m_stringToLabel;
};
