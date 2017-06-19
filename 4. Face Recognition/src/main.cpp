#include "dataset_manager.h"
#include "face_recognizer.h"
#include "face_detector.h"
#include "util/log.h"
#include "util/fsutil.h"

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

void videoLoop(cv::VideoCapture& cap, FaceDetector& detector, FaceRecognizer& recognizer)
{
	while (true) {
		cv::Mat frame;
		cap >> frame;
		std::vector<FaceDetector::FaceRegion> faces = detector.detect(frame);
		for (const auto& face : faces) {
			cv::rectangle(frame, face.rect, cv::Scalar(0, 255, 0));
			std::string pred = recognizer.predict(face.image);
			int pos_x = std::max(face.rect.tl().x - 10, 0);
			int pos_y = std::max(face.rect.tl().y - 10, 0);
			putText(frame, pred, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0,255,0), 2);
		}
		cv::imshow("frame", frame);
		static const int KEY_ESC = 27;
		if (cv::waitKey(10) == KEY_ESC) break;
	}
}

const std::string g_hardcodedRoot = "/home/np/Stuff/hse-cv-course-2017/4. Face Recognition";
const std::string g_hardcodedDataset = g_hardcodedRoot + "/_data/";
const std::string g_hardcodedData = g_hardcodedRoot + "/data/";

int main()
{
	std::string recognizerConfig = g_hardcodedDataset + "facerec_config";
	FaceRecognizer facerec;
	if (fs::pathExists(recognizerConfig))
		facerec.load(recognizerConfig);
	DatasetManager mgr;
	std::string mgrConfig = fs::concatPath(g_hardcodedDataset, "mgr_config.xml");
	mgr.readConfig(mgrConfig);
	mgr.load(g_hardcodedDataset);
	if (mgr.datasetChanged()) {
		facerec.train(mgr.readDataset());
		facerec.save(recognizerConfig);
		mgr.saveConfig(mgrConfig);
	}

	if (!facerec.ready()) {
		logError() << "Failed to start face recognizer!";
		return 1;
	}

	std::string detectorConfig = g_hardcodedData + "haarcascades";
	FaceDetector detector(detectorConfig);

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		logError() << "Can't open capture device!";
		return 1;
	}
	logInfo() << "Press ESC to exit";
	videoLoop(cap, detector, facerec);

	//facerec.save(recognizerConfig);
	return 0;
}

