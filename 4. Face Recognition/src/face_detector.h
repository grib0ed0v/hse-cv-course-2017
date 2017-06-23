#pragma once

#include <opencv2/opencv.hpp>

class FaceDetector
{
public:
	struct FaceRegion
	{
		cv::Rect rect;
		cv::Mat image;
	};

	FaceDetector(const std::string& cascadePath, const std::string& configPath);
	std::vector<FaceRegion> detect(cv::Mat img);

	bool readConfig(const std::string& path);
	bool saveConfig(const std::string& path);

private:
	cv::Mat processFace(cv::Mat img);
	void applyMask(cv::Mat& img);
	void denoise(cv::Mat& img);
	void resize(cv::Mat& img);
	cv::Point2i detectEye(cv::CascadeClassifier& classifier, cv::Mat img);
	void geometryTransform(cv::Mat& img);
	void normalizeIllumination(cv::Mat& img);

#define STREAM_VAR(var) << #var << var
#define READ_VAR_NODE(var) var = (decltype(var))node[#var]

	struct Config
	{
		struct CascadePaths
		{
			std::string classifierPath = "haarcascades/haarcascade_frontalface_default.xml";
			std::string leftEyeClassifierPath = "haarcascades/haarcascade_mcs_lefteye.xml";
			std::string rightEyeClassifierPath = "haarcascades/haarcascade_mcs_righteye.xml";
			std::string eyeglassClassifierPath = "haarcascades/haarcascade_eye_tree_eyeglasses.xml";

			void write(cv::FileStorage& fs) const
			{
				fs << "{"
					STREAM_VAR(classifierPath)
					STREAM_VAR(leftEyeClassifierPath)
					STREAM_VAR(rightEyeClassifierPath)
					STREAM_VAR(eyeglassClassifierPath)
				<< "}";
			}
			void read(const cv::FileNode& node)
			{
				READ_VAR_NODE(classifierPath);
				READ_VAR_NODE(leftEyeClassifierPath);
				READ_VAR_NODE(rightEyeClassifierPath);
				READ_VAR_NODE(eyeglassClassifierPath);
			}
		} cascades;

		struct Denoise
		{
			int d = 5;
			double sigmaColor = 50.0;
			double sigmaSpace = 50.0;

			void write(cv::FileStorage& fs) const
			{
				fs << "{"
					STREAM_VAR(d)
					STREAM_VAR(sigmaColor)
					STREAM_VAR(sigmaSpace)
				<< "}";
			}
			void read(const cv::FileNode& node)
			{
				READ_VAR_NODE(d);
				READ_VAR_NODE(sigmaColor);
				READ_VAR_NODE(sigmaSpace);
			}
		} denoise;

		struct Resize
		{
			float size = 120.0f;
			void write(cv::FileStorage& fs) const
			{
				fs << "{"
					STREAM_VAR(size)
				<< "}";
			}
			void read(const cv::FileNode& node)
			{
				READ_VAR_NODE(size);
			}
		} resize;

		struct EyeDetect
		{
			double scaleFactor   = 1.1;
			int    minNeighbors  = 6;
			double minSizeFactor = 1.0 / 6;
			double maxSizeFactor = 1.0 / 2;

			void write(cv::FileStorage& fs) const
			{
				fs << "{"
					STREAM_VAR(scaleFactor)
					STREAM_VAR(minNeighbors)
					STREAM_VAR(minSizeFactor)
					STREAM_VAR(maxSizeFactor)
				<< "}";
			}
			void read(const cv::FileNode& node)
			{
				READ_VAR_NODE(scaleFactor);
				READ_VAR_NODE(minNeighbors);
				READ_VAR_NODE(minSizeFactor);
				READ_VAR_NODE(maxSizeFactor);
			}
		} eyeDetect;

		struct GeometryTransform
		{
			double eyeTopFactor = 1.0 / 4;
			double eyeBottomFactor = 1.0 / 1.5;
			double maxAngle = 30.0;

			void write(cv::FileStorage& fs) const
			{

				fs << "{"
					STREAM_VAR(eyeTopFactor)
					STREAM_VAR(eyeBottomFactor)
					STREAM_VAR(maxAngle)
				<< "}";
			}
			void read(const cv::FileNode& node)
			{
				READ_VAR_NODE(eyeTopFactor);
				READ_VAR_NODE(eyeBottomFactor);
				READ_VAR_NODE(maxAngle);
			}
		} geometry;

		struct IlluminationCorrection
		{
			double gamma = 0.7;
			double clipLimit = 2.0;
			int claheTileSize = 8;

			void write(cv::FileStorage& fs) const
			{

				fs << "{"
					STREAM_VAR(gamma)
					STREAM_VAR(clipLimit)
					STREAM_VAR(claheTileSize)
				<< "}";
			}
			void read(const cv::FileNode& node)
			{
				READ_VAR_NODE(gamma);
				READ_VAR_NODE(clipLimit);
				READ_VAR_NODE(claheTileSize);
			}
		} illumination;

		struct MaskApply
		{
			double axeXFactor = 0.4;
			double axeYFactor = 0.7;

			void write(cv::FileStorage& fs) const
			{

				fs << "{"
					STREAM_VAR(axeXFactor)
					STREAM_VAR(axeYFactor)
					<< "}";
			}
			void read(const cv::FileNode& node)
			{
				READ_VAR_NODE(axeXFactor);
				READ_VAR_NODE(axeYFactor);
			}
		} mask;

		struct FaceDetect
		{
			double scaleFactor = 1.2;
			int    minNeighbors = 5;
			int    minSizeX = 30;
			int    minSizeY = 30;

			void write(cv::FileStorage& fs) const
			{

				fs << "{"
					STREAM_VAR(scaleFactor)
					STREAM_VAR(minNeighbors)
					STREAM_VAR(minSizeX)
					STREAM_VAR(minSizeY)
					<< "}";
			}
			void read(const cv::FileNode& node)
			{
				READ_VAR_NODE(scaleFactor);
				READ_VAR_NODE(minNeighbors);
				READ_VAR_NODE(minSizeX);
				READ_VAR_NODE(minSizeY);
			}
		} faceDetect;

		struct Debug
		{
			// bool does not work well
			int showResult = false;
			int waitKey = false;
			int drawEyes = false;
			int noDetect = false;
			int noProcessing = false;
			int preserveColor = false;
			double extendRectFactor = 1.0;

			void write(cv::FileStorage& fs) const
			{

				fs << "{"
					STREAM_VAR(showResult)
					STREAM_VAR(waitKey)
					STREAM_VAR(drawEyes)
					STREAM_VAR(noDetect)
					STREAM_VAR(noProcessing)
					STREAM_VAR(preserveColor)
					STREAM_VAR(extendRectFactor)
					<< "}";
			}
			void read(const cv::FileNode& node)
			{
				READ_VAR_NODE(showResult);
				READ_VAR_NODE(waitKey);
				READ_VAR_NODE(drawEyes);
				READ_VAR_NODE(noDetect);
				READ_VAR_NODE(noProcessing);
				READ_VAR_NODE(preserveColor);
				READ_VAR_NODE(extendRectFactor);
			}
		} debug;

		bool write(const std::string& path) const
		{
			cv::FileStorage fs(path, cv::FileStorage::WRITE);
			if (!fs.isOpened()) {
				return false;
			}
			fs << "cascades";		cascades.write(fs);
			fs << "denoise";		denoise.write(fs);
			fs << "resize";			resize.write(fs);
			fs << "eyeDetect";		eyeDetect.write(fs);
			fs << "geometry";		geometry.write(fs);
			fs << "illumination";	illumination.write(fs);
			fs << "mask";			mask.write(fs);
			fs << "faceDetect";		faceDetect.write(fs);
			fs << "debug";			debug.write(fs);
			return true;
		}
		bool read(const std::string& path)
		{
			cv::FileStorage fs(path, cv::FileStorage::READ);
			if (!fs.isOpened()) {
				return false;
			}
			cascades.read(fs["cascades"]);
			denoise.read(fs["denoise"]);
			resize.read(fs["resize"]);
			eyeDetect.read(fs["eyeDetect"]);
			geometry.read(fs["geometry"]);
			illumination.read(fs["illumination"]);
			mask.read(fs["mask"]);
			faceDetect.read(fs["faceDetect"]);
			debug.read(fs["debug"]);
			return true;
		}
	};

#undef STREAM_VAR
#undef READ_VAR_NODE

	Config m_config;
	cv::CascadeClassifier m_classifier;
	//cv::CascadeClassifier m_eyeClassifier;
	cv::CascadeClassifier m_leftEyeClassifier;
	cv::CascadeClassifier m_rightEyeClassifier;
	cv::CascadeClassifier m_eyeglassClassifier;
};
