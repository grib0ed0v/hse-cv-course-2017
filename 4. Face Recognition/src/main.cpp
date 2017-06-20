#include "dataset_manager.h"
#include "face_recognizer.h"
#include "face_detector.h"
#include "util/log.h"
#include "util/fsutil.h"

#include <opencv2/opencv.hpp>

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

struct ProgramFolders
{
	std::string configFolder;
	std::string datasetFolder;
};

ProgramFolders readProgramFolders(const std::string& path)
{
	ProgramFolders programFolders;
	cv::FileStorage fileStorage(path, cv::FileStorage::READ);
	if (!fileStorage.isOpened())
		return programFolders;

	programFolders.configFolder = (std::string)fileStorage["configFolder"];
	programFolders.datasetFolder = (std::string)fileStorage["datasetFolder"];
	return programFolders;
}

void writeProgramFolders(const ProgramFolders& config, const std::string& path)
{
	cv::FileStorage fileStorage(path, cv::FileStorage::WRITE);
	fileStorage << "configFolder" << config.configFolder;
	fileStorage << "datasetFolder" << config.datasetFolder;
}

int main(int argc, const char* argv[])
{
	ProgramFolders config;
	if (argc < 3) {
		config = readProgramFolders("folders.xml");
		if (!config.configFolder.empty()) {
			logInfo() << "Using config folder:" << config.configFolder;
			logInfo() << "Using dataset folder:" << config.datasetFolder;
		}
	} else {
		config.configFolder = argv[1];
		config.datasetFolder = argv[2];
	}

	if (config.configFolder.empty()) {
		logInfo() << "On first usage pass two params:";
		logInfo() << argv[0] << "config_folder dataset_folder";
		logError() << "No config folder, aborting";
		return 1;
	}

	if (!fs::isDir(config.configFolder)) {
		logError() << "Invalid config folder:" << config.configFolder;
		return 1;
	}

	std::string recognizerConfig = fs::concatPath(config.configFolder, "facerec_config");
	FaceRecognizer facerec;
	if (fs::pathExists(recognizerConfig)) {
		logInfo() << "Loading pre-trained model from" << recognizerConfig;
		facerec.load(recognizerConfig);
	}
	if (!config.datasetFolder.empty()) {
		DatasetManager mgr;
		std::string mgrConfig = fs::concatPath(config.configFolder, "mgr_config.xml");
		mgr.readConfig(mgrConfig);
		mgr.load(config.datasetFolder);
		if (mgr.datasetChanged() || !facerec.ready()) {
			logInfo() << "Training recognizer...";
			facerec.train(mgr.readDataset());
			logInfo() << "Done";
			if (facerec.ready()) {
				facerec.save(recognizerConfig);
				mgr.saveConfig(mgrConfig);
			}
		}
	}

	if (!facerec.ready()) {
		logError() << "Failed to start face recognizer!";
		return 1;
	}

	std::string detectorConfig = "haarcascades";
	FaceDetector detector(detectorConfig);

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		logError() << "Can't open capture device!";
		return 1;
	}
	logInfo() << "Press ESC to exit";
	videoLoop(cap, detector, facerec);

	writeProgramFolders(config, "folders.xml");
	return 0;
}
