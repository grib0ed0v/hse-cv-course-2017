#include "dataset_manager.h"
#include "face_recognizer.h"
#include "face_detector.h"
#include "webcam_ui.h"
#include "util/log.h"
#include "util/fsutil.h"
#include "util/argparser.h"

#include <opencv2/opencv.hpp>

struct ProgramParams
{
	std::string configFolder;
	std::string datasetFolder;
	bool retrain = false;
};

enum ProgramArgIds
{
	ConfigFolder,
	Dataset,
	Retrain,
};

std::vector<ArgParser::Arg> g_programArgs = {
	{
		ProgramArgIds::ConfigFolder,
		"c",
		"config",
		"Folder with configuration files",
		true
	},
	{
		ProgramArgIds::Dataset,
		"d",
		"dataset",
		"Folder with images for training",
		true
	},
	{
		ProgramArgIds::Retrain,
		"r",
		"retrain",
		"Disregard stored model and train recognizer again",
		false
	},
};

class ProgramArgParser : public ArgParser
{
public:
	ProgramArgParser()
	{
		setArgInfo(g_programArgs);
	}

	ProgramParams params() { return m_params; }

protected:
	virtual bool processArg(const Arg& arg, const char* param) override
	{
		switch (arg.id)
		{
		case ConfigFolder:
			m_params.configFolder = param;
			break;

		case Dataset:
			m_params.datasetFolder = param;
			break;

		case Retrain:
			m_params.retrain = true;
			break;

		default:
			return false;
		}
		return true;
	};

private:
	ProgramParams m_params;
};

ProgramArgParser g_argParser;

ProgramParams readProgramFolders(const std::string& path)
{
	ProgramParams programFolders;
	cv::FileStorage fileStorage(path, cv::FileStorage::READ);
	if (!fileStorage.isOpened())
		return programFolders;

	programFolders.configFolder = (std::string)fileStorage["configFolder"];
	programFolders.datasetFolder = (std::string)fileStorage["datasetFolder"];
	return programFolders;
}

void writeProgramFolders(const ProgramParams& config, const std::string& path)
{
	cv::FileStorage fileStorage(path, cv::FileStorage::WRITE);
	fileStorage << "configFolder" << config.configFolder;
	fileStorage << "datasetFolder" << config.datasetFolder;
}

ProgramParams prepareParams(int argc, char* argv[])
{
	ProgramParams config;
	size_t invalidArgIdx = g_argParser.parseArgs(argc - 1, argv + 1);
	if (invalidArgIdx != 0) {
		logError() << "Invalid argument:" << argv[invalidArgIdx];
		exit(1);
	}
	config = g_argParser.params();

	ProgramParams storedFolders = readProgramFolders("folders.xml");
	if (config.configFolder.empty()) {
		config.configFolder = storedFolders.configFolder;
	}
	if (config.datasetFolder.empty()) {
		config.datasetFolder = storedFolders.datasetFolder;
	}

	if (config.configFolder.empty()) {
		logInfo() << g_argParser.generateUsageMessage(argv[0]);
		logError() << "No config folder, aborting";
		exit(1);
	}

	if (!fs::isDir(config.configFolder)) {
		logError() << "Invalid config folder:" << config.configFolder;
		exit(1);
	}
	return config;
}

int main(int argc, char* argv[])
{
	ProgramParams config = prepareParams(argc, argv);

	logInfo() << "Config folder:" << config.configFolder;
	logInfo() << "Dataset folder:" << (config.datasetFolder.empty() ? "(empty)" : config.datasetFolder);

	std::string recognizerConfig = fs::concatPath(config.configFolder, "facerec_config");
	FaceRecognizer facerec;
	if (!config.retrain && fs::pathExists(recognizerConfig)) {
		logInfo() << "Loading pre-trained model from" << recognizerConfig;
		facerec.load(recognizerConfig);
	}
	if (config.retrain && config.datasetFolder.empty()) {
		logError() << "Can't train recognizer because dataset folder is not specified!";
		return 1;
	}
	if (!config.datasetFolder.empty()) {
		DatasetManager mgr;
		std::string mgrConfig = fs::concatPath(config.configFolder, "mgr_config.xml");
		mgr.readConfig(mgrConfig);
		mgr.load(config.datasetFolder);
		if (config.retrain || mgr.datasetChanged() || !facerec.ready()) {
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

	std::string detectorConfig = ".";
	FaceDetector detector(detectorConfig);

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		logError() << "Can't open capture device!";
		return 1;
	}
	logInfo() << "Press ESC to exit";

	WebcamUI ui("frame", cap, detector, facerec);
	ui.videoLoop();

	writeProgramFolders(config, "folders.xml");
	return 0;
}
