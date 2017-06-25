#pragma once

#include <opencv2/core/utility.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>
#include <ctype.h>
#include <fstream>
#include <cstdlib>
#include <map>
#include <iomanip>
#include <time.h>
#include <string>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class ImageClassification
{
public:
	ImageClassification();
	~ImageClassification();
	std::vector<String> readClassNames(const char *filename = "synset_words.txt");
	void getMaxClass(const Mat &probBlob, int *classId, double *classProb);
	void testOpenCVDnnWithCaffeModel(const string & modelTxtPath, const string & modelBinPath,
		const string & classesFilePath, Mat & image);
};

