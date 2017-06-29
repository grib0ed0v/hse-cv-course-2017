#include "ImageClassification.h"

Mat image;

Point origin;
Rect selection;
ImageClassification *imageClassificator;

int main(int argc, const char** argv)
{
	VideoCapture cap;
	Rect trackWindow;
	int hsize = 16;
	float hranges[] = { 0,180 };
	const float* phranges = hranges;

	string modelTxtPath = "C:\\Users\\Denis\\Documents\\Visual Studio 2017\\Projects\\HSEOpenCVProject\\x64\\Debug\\data\\bvlc_googlenet\\bvlc_googlenet.prototxt";
	string modelBinPath = "C:\\Users\\Denis\\Documents\\Visual Studio 2017\\Projects\\HSEOpenCVProject\\x64\\Debug\\data\\bvlc_googlenet\\bvlc_googlenet.caffemodel";
	string classesPath = "C:\\Users\\Denis\\Documents\\Visual Studio 2017\\Projects\\HSEOpenCVProject\\x64\\Debug\\data\\bvlc_googlenet\\synset_words.txt";

	imageClassificator = new ImageClassification();

	cap.open(0);

	if (!cap.isOpened())
	{
		cout << "***Could not initialize capturing...***\n";
		cout << "Current parameter's value: \n";
		return -1;
	}

	namedWindow("Camera", 0);

	Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
	bool paused = false;

	for (;;)
	{
		if (!paused)
		{
			cap >> frame;
			if (frame.empty())
				break;
		}

		frame.copyTo(image);

		imageClassificator->testOpenCVDnnWithCaffeModel(modelTxtPath, modelBinPath, classesPath, image);

		imshow("Camera", image);

		char c = (char)waitKey(10);
		if (c == 27)
			break;
		switch (c)
		{
		case 'p':
			paused = !paused;
			break;
		default:
			break;
		}
	}

	return 0;
}
