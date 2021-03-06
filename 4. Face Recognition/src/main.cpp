#include "dataset_manager.h"
#include "face_recognizer.h"
#include "face_detector.h"
#include "webcam_ui.h"
#include "util/log.h"
#include "util/fsutil.h"
#include "util/argparser.h"

#include <opencv2/opencv.hpp>

const struct ConfigNames
{
	std::string manager = "manager_config.json";
	std::string detector = "detector_config.json";
	std::string recognizer = "facerec_config";
	std::string pretrainedRecognizer = "pretrained/facerec_config";
	std::string folders = "folders.json";
	std::string recognizerparams = "facerec_params_config.json";
} g_configNames;

struct ProgramParams
{
	std::string configFolder;
	std::string datasetFolder;
	std::string cascadeFolder;
	bool retrain = false;
	bool doPreprocessDataset = false;
	bool doPreprocessImage = false;
	bool doRecognizeImage = false;
	bool doTestOnDataset = false;
	std::string inputPath;
	std::string outputPath;
	bool printHelp = false;
};

enum ProgramArgIds
{
	ConfigFolder,
	DatasetFolder,
	CascadeFolder,
	Retrain,
	PreprocessDataset,
	PreprocessImage,
	RecognizeImage,
	TestDataset,
	InputPath,
	OutputPath,
	Help,
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
		ProgramArgIds::DatasetFolder,
		"d",
		"dataset",
		"Folder with images for training",
		true
	},
	{
		ProgramArgIds::CascadeFolder,
		"ca",
		"cascade",
		"Folder with cascade classifiers",
		true
	},
	{
		ProgramArgIds::Retrain,
		"t",
		"train",
		"Disregard stored model and train recognizer again",
		false
	},
	{
		ProgramArgIds::PreprocessDataset,
		"pd",
		"preprocess-dataset",
		"Run face detection and preprocessing on dataset and write to output",
		false
	},
	{
		ProgramArgIds::PreprocessImage,
		"pi",
		"preprocess-image",
		"Run face detection and preprocessing on single image",
		false
	},
	{
		ProgramArgIds::RecognizeImage,
		"r",
		"recognize",
		"Preprocess and recognize faces on single image",
		false
	},
	{
		ProgramArgIds::TestDataset,
		"td",
		"test",
		"Test recognizer against specified dataset",
		false
	},
	{
		ProgramArgIds::InputPath,
		"i",
		"input",
		"Specify input directory or file for --preprocess-image, --recognize, and --test",
		true
	},
	{
		ProgramArgIds::OutputPath,
		"o",
		"output",
		"Specify output directory for --preprocess-image and --recognize",
		true
	},
	{
		ProgramArgIds::Help,
		"h",
		"help",
		"Print help message and exit",
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

		case DatasetFolder:
			m_params.datasetFolder = param;
			break;

		case CascadeFolder:
			m_params.cascadeFolder = param;
			break;

		case Retrain:
			m_params.retrain = true;
			break;

		case PreprocessDataset:
			m_params.doPreprocessDataset = true;
			break;

		case PreprocessImage:
			m_params.doPreprocessImage = true;
			break;

		case RecognizeImage:
			m_params.doRecognizeImage = true;
			break;

		case InputPath:
			m_params.inputPath = param;
			break;

		case OutputPath:
			m_params.outputPath = param;
			break;

		case TestDataset:
			m_params.doTestOnDataset = true;
			break;

		case Help:
			m_params.printHelp = true;
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
	programFolders.cascadeFolder = (std::string)fileStorage["cascadeFolder"];
	return programFolders;
}

void writeProgramFolders(const ProgramParams& config, const std::string& path)
{
	cv::FileStorage fileStorage(path, cv::FileStorage::WRITE);
	fileStorage << "configFolder" << config.configFolder;
	fileStorage << "datasetFolder" << config.datasetFolder;
	fileStorage << "cascadeFolder" << config.cascadeFolder;
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

	if (config.printHelp) {
		logInfo() << g_argParser.generateUsageMessage(argv[0]);
		exit(0);
	}

	ProgramParams storedFolders = readProgramFolders(g_configNames.folders);
	if (config.configFolder.empty()) {
		config.configFolder = storedFolders.configFolder;
	}
	if (config.datasetFolder.empty()) {
		config.datasetFolder = storedFolders.datasetFolder;
	}
	if (config.cascadeFolder.empty()) {
		config.cascadeFolder = storedFolders.cascadeFolder;
	}

	if (config.configFolder.empty()) {
		config.configFolder = ".";
	}

	if (!fs::pathExists(config.configFolder) && !fs::mkdir(config.configFolder)) {
		logError() << "Invalid config folder:" << config.configFolder;
		exit(1);
	}

	if (config.cascadeFolder.empty()) {
		config.cascadeFolder = ".";
	}

	if (!fs::isDir(config.cascadeFolder)) {
		logError() << "Invalid cascade folder:" << config.cascadeFolder;
		exit(1);
	}
	return config;
}

void preprocessDataset(ProgramParams config)
{

	if (!fs::pathExists(config.outputPath) && !fs::mkdir(config.outputPath)) {
		logError() << "Could not create folder:" << config.outputPath;
		return;
	}

	FaceDetector detector(config.cascadeFolder, fs::concatPath(config.configFolder, g_configNames.detector));

	Dataset data;
	if (config.datasetFolder.empty()) {
		logError() << "Dataset folder not set!";
		return;
	}

	std::vector<std::string> datasetContent = fs::getFilesInDir(config.datasetFolder);
	auto removeIt = std::remove_if(datasetContent.begin(), datasetContent.end(),
		[&config](const std::string& path)
	{
		return !fs::isDir(fs::concatPath(config.datasetFolder, path));
	});
	datasetContent.erase(removeIt, datasetContent.end());

	if (datasetContent.empty()) {
		logError() << "Empty datset:" << config.datasetFolder;
	}

	for (const std::string& folderName : datasetContent) {
		std::string folderPath = fs::concatPath(config.datasetFolder, folderName);
		std::vector<std::string> folderContent = fs::getFilesInDir(folderPath);
		if (folderContent.empty()) {
			logWarning() << "Empty folder:" << folderPath;
			continue;
		}

		for (const std::string imgName : folderContent) {
			std::string imgPath = fs::concatPath(folderPath, imgName);
			cv::Mat image = cv::imread(imgPath, cv::IMREAD_COLOR);
			if (image.data == nullptr) {
				logError() << "Could not read" << imgPath;
				continue;
			}

			std::string imgFolder = fs::concatPath(config.outputPath, folderName);
			if (!fs::pathExists(imgFolder) && !fs::mkdir(imgFolder)) {
				logError() << "Could not create folder:" << imgFolder;
				return;
			}

			std::vector<FaceDetector::FaceRegion> faces = detector.detect(image);
			if (faces.size() > 1) {
				logWarning() << folderName << " : more than one face found on image" << imgName;
			}
			else if (faces.empty()) {
				logWarning() << folderName << " : no faces found on image" << imgName;
				continue;
			}

			size_t createdImages = 0;
			for (const auto& face : faces) {
				std::string outImgName = imgName + "_" + std::to_string(createdImages) + ".jpg";
				cv::imwrite(fs::concatPath(imgFolder, outImgName), face.image);
				++createdImages;
			}
		}
	}
}

void processImage(ProgramParams config, FaceRecognizer* pRecognizer = nullptr)
{
	cv::Mat image = cv::imread(config.inputPath, cv::IMREAD_COLOR);
	if (image.data == nullptr) {
		logError() << "Could not read" << config.inputPath;
		return;
	}

	FaceDetector detector(config.cascadeFolder, fs::concatPath(config.configFolder, g_configNames.detector));
	size_t createdImages = 0;

	std::vector<FaceDetector::FaceRegion> faces = detector.detect(image);
	logInfo() << "Found" << faces.size() << (faces.size() == 1 ? "face" : "faces");

	for (const auto& face : faces) {
		std::string outImgName = std::to_string(createdImages);
		if (pRecognizer) {
			std::string pred = pRecognizer->predict(face.image);
			if (pred.empty()) pred = "unknown";
			logInfo() << pred;

			if (config.outputPath.empty()) {
				continue;
			}

			outImgName += '_';
			outImgName += pred;
		}
		outImgName += ".png";
		std::string outImgPath = fs::concatPath(config.outputPath, outImgName);
		if (!cv::imwrite(outImgPath, face.image)) {
			logError() << "Could not write" << outImgPath;
			return;
		}
		++createdImages;
	}
}

bool ensureOutput(std::string& output)
{
	if (output.empty()) {
		logInfo() << "Output path not specified, won't write images";
		return true;
	}
	if (!fs::pathExists(output) && !fs::mkdir(output)) {
		logError() << "Could not create folder:" << output;
		return false;
	}
	return true;
}

bool promptYN(std::string prompt, bool defaultValue = false)
{
	bool resp = false;
	std::string response;
	char ychar = defaultValue ? 'Y' : 'y';
	char nchar = defaultValue ? 'n' : 'N';
	std::string yesNo = " (";
	yesNo += ychar;
	yesNo += "(es)/";
	yesNo += nchar;
	yesNo += "(o)): ";
	while (true) {
		std::cout << prompt << yesNo;
		std::getline(std::cin, response);
		if (response.empty()) {
			return defaultValue;
		}

		std::transform(response.begin(), response.end(), response.begin(), tolower);
		if (response == "y" || response == "yes") {
			resp = true;
			break;
		} else if (response == "n" || response == "no") {
			resp = false;
			break;
		}
		std::cout << "Yes or no, please" << std::endl;
	}
	return resp;
}

void saveImages(ProgramParams config, const Dataset& newData) {
	if (!fs::pathExists(config.datasetFolder) && !fs::mkdir(config.datasetFolder)) {
		logError() << "Could not create folder:" << config.datasetFolder;
		return;
	}

	std::map<label_t, size_t> createdImages;
	for (size_t i = 0; i < newData.images().size(); ++i) {
		const cv::Mat& image = newData.images()[i];
		const label_t label = newData.labels()[i];
		const std::string& str = newData.stringByLabel(label);

		std::string imgFolder = fs::concatPath(config.datasetFolder, str);
		if (!fs::pathExists(imgFolder) && !fs::mkdir(imgFolder)) {
			logError() << "Could not create folder:" << imgFolder;
			return;
		}

		while (true) {
			size_t imgNumber = createdImages[label]++;
			std::string imgName = std::to_string(imgNumber) + ".png";
			std::string imgPath = fs::concatPath(imgFolder, imgName);
			if (fs::pathExists(imgPath)) {
				continue;
			}
			if (!cv::imwrite(imgPath, image)) {
				logError() << "Could not write" << imgPath;
				break;
			}
			break;
		}
	}
}

void testOnDataset(ProgramParams config, FaceDetector& detector, FaceRecognizer& recognizer)
{

	std::vector<std::string> datasetContent = fs::getFilesInDir(config.inputPath);
	auto removeIt = std::remove_if(datasetContent.begin(), datasetContent.end(),
		[&config](const std::string& path)
	{
		return !fs::isDir(fs::concatPath(config.inputPath, path));
	});
	datasetContent.erase(removeIt, datasetContent.end());

	if (datasetContent.empty()) {
		logError() << "Empty datset:" << config.inputPath;
	}

	size_t totalImg = 0;
	size_t detectedImg = 0;
	size_t multiDetections = 0;
	size_t correctImg = 0;
	for (const std::string& folderName : datasetContent) {
		std::string folderPath = fs::concatPath(config.inputPath, folderName);
		std::vector<std::string> folderContent = fs::getFilesInDir(folderPath);
		if (folderContent.empty()) {
			logWarning() << "Empty folder:" << folderPath;
			continue;
		}

		for (const std::string imgName : folderContent) {
			++totalImg;
			std::string imgPath = fs::concatPath(folderPath, imgName);
			cv::Mat image = cv::imread(imgPath, cv::IMREAD_COLOR);
			if (image.data == nullptr) {
				logError() << "Could not read" << imgPath;
				continue;
			}

			std::vector<FaceDetector::FaceRegion> faces = detector.detect(image);
			if (!faces.empty()) ++detectedImg;
			if (faces.size() > 1) ++multiDetections;
			for (const auto& face : faces) {
				std::string pred = recognizer.predict(face.image);
				logInfo() << pred << "(" << folderName << ")";
				if (pred == folderName) ++correctImg;
			}
		}
	}
	logInfo() << "Images total:" << totalImg;
	logInfo() << "Detected:" << detectedImg;
	logInfo() << "Multidetect:" << multiDetections;
	logInfo() << "Accuracy:" << (double)correctImg / totalImg << "(" << correctImg << "/" << totalImg << ")";
}

int main(int argc, char* argv[])
{
	ProgramParams config = prepareParams(argc, argv);

	logInfo() << "Config folder:" << config.configFolder;
	logInfo() << "Dataset folder:" << (config.datasetFolder.empty() ? "(empty)" : config.datasetFolder);
	logInfo() << "Cascade folder:" << config.cascadeFolder;

	if (config.doPreprocessDataset) {
		ensureOutput(config.outputPath);
		logInfo() << "Preprocessing dataset, output:" << config.outputPath;
		preprocessDataset(config);
		return 0;
	}

	if (config.doPreprocessImage) {
		ensureOutput(config.outputPath);
		logInfo() << "Processing" << config.inputPath << " output:" << (config.outputPath.empty() ? "(empty)" : config.outputPath);
		processImage(config);
		if (!config.doRecognizeImage) {
			return 0;
		}
	}

	std::string recognizerConfig = fs::concatPath(config.configFolder, g_configNames.recognizer);
	FaceRecognizer facerec(fs::concatPath(config.configFolder, g_configNames.recognizerparams));
	if (!config.retrain && fs::pathExists(recognizerConfig)) {
		logInfo() << "Loading pre-trained model from" << recognizerConfig;
		facerec.load(recognizerConfig);
	}

	if (config.retrain && config.datasetFolder.empty()) {
		logError() << "Can't train recognizer because dataset folder is not specified!";
		return 1;
	}

	if (config.datasetFolder.empty() && !fs::pathExists(recognizerConfig)) {
		std::string defaultRecognizerConfig = fs::concatPath(config.configFolder, g_configNames.pretrainedRecognizer);
		logInfo() << "Loading default pre-trained model from" << defaultRecognizerConfig;
		facerec.load(defaultRecognizerConfig);
	}

	if (!config.datasetFolder.empty()) {
		DatasetManager mgr;
		std::string mgrConfig = fs::concatPath(config.configFolder, g_configNames.manager);
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

	if (config.doRecognizeImage) {
		ensureOutput(config.outputPath);
		logInfo() << "Recognizing" << config.inputPath << " output:" << (config.outputPath.empty() ? "(empty)" : config.outputPath);
		processImage(config, &facerec);
		return 0;
	}

	FaceDetector detector(config.cascadeFolder, fs::concatPath(config.configFolder, g_configNames.detector));

	if (config.doTestOnDataset) {
		testOnDataset(config, detector, facerec);
		return 0;
	}

	cv::VideoCapture cap(0);
	if (!cap.isOpened()) {
		logError() << "Can't open capture device!";
		return 1;
	}
	logInfo() << "Press ESC to exit";

	WebcamUI ui("frame", cap, detector, facerec);
	ui.videoLoop();

	if (!facerec.newData().images().empty()) {
		if (!config.datasetFolder.empty() && promptYN("Save new images?")) {
			saveImages(config, facerec.newData());
			logInfo() << "Done";
		}
		if (promptYN("Update stored recognizer with new data?")) {
			facerec.save(recognizerConfig);
			logInfo() << "Done";
		}
	}

	writeProgramFolders(config, g_configNames.folders);
	return 0;
}
