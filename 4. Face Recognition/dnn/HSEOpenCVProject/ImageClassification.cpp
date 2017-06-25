#include "ImageClassification.h"

//////////////////////////////////////////////////////////////////////////////////////////////////////

ImageClassification::ImageClassification()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

ImageClassification::~ImageClassification()
{
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageClassification::getMaxClass(const Mat &probBlob, int *classId, double *classProb)
{
	Mat probMat = probBlob.reshape(1, 1); //reshape the blob to 1x1000 matrix
	Point classNumber;
	minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
	*classId = classNumber.x;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<String> ImageClassification::readClassNames(const char *filename)
{
	std::vector<String> classNames;
	std::ifstream fp(filename);
	if (!fp.is_open())
	{
		std::cerr << "File with classes labels not found: " << filename << std::endl;
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name.substr(name.find(' ') + 1));
	}
	fp.close();
	return classNames;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////

void ImageClassification::testOpenCVDnnWithCaffeModel(const string & modelTxtPath, const string & modelBinPath,
	const string & classesFilePath, Mat & image)
{
	cv::dnn::initModule();  //Required if OpenCV is built as static libs

	Net net = dnn::readNetFromCaffe(modelTxtPath, modelBinPath);
	if (net.empty())
	{
		std::cerr << "Can't load network by using the following files: " << std::endl;
		std::cerr << "prototxt:   " << modelTxtPath << std::endl;
		std::cerr << "caffemodel: " << modelBinPath << std::endl;
		std::cerr << "bvlc_googlenet.caffemodel can be downloaded here:" << std::endl;
		std::cerr << "http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel" << std::endl;
		exit(-1);
	}
	Mat img = image;

	resize(img, img, Size(224, 224));                   //GoogLeNet accepts only 224x224 RGB-images
	Mat inputBlob = blobFromImage(img);   //Convert Mat to batch of images
	net.setInput(inputBlob, "data");        //set the network input
	Mat prob = net.forward("prob");                          //compute output
	int classId;
	double classProb;
	getMaxClass(prob, &classId, &classProb);//find the best class
	if (classProb > 0.5)
	{
		std::vector<String> classNames = readClassNames(classesFilePath.c_str());
		std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
		std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
