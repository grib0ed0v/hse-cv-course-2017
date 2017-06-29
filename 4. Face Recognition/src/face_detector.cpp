#include "face_detector.h"

#include "util/log.h"
#include "util/fsutil.h"

#include <opencv2/imgproc.hpp>
#include <cmath>

static const double PI = 3.14159265358979323846;

FaceDetector::FaceDetector(const std::string& cascadePath, const std::string& configPath)
{
	if (!fs::pathExists(configPath)) {
		saveConfig(configPath);
	} else {
		if (!readConfig(configPath)) {
			logWarning() << "FaceDetector config exists, but can't be read from" << configPath;
		}
	}
	m_classifier.load(fs::concatPath(cascadePath, m_config.cascades.classifierPath));

	m_leftEyeClassifier.load(fs::concatPath(cascadePath, m_config.cascades.leftEyeClassifierPath));
	m_rightEyeClassifier.load(fs::concatPath(cascadePath, m_config.cascades.rightEyeClassifierPath));

	m_eyeglassClassifier.load(fs::concatPath(cascadePath, m_config.cascades.eyeglassClassifierPath));
}

bool FaceDetector::readConfig(const std::string& path)
{
	return m_config.read(path);
}

bool FaceDetector::saveConfig(const std::string& path)
{
	return m_config.write(path);
}

void FaceDetector::denoise(cv::Mat& img)
{
	cv::Mat dest = img.clone();
	cv::bilateralFilter(img, dest, m_config.denoise.d, m_config.denoise.sigmaColor, m_config.denoise.sigmaSpace);
	img = dest;
}

void FaceDetector::resize(cv::Mat& img)
{
	float scaleFactor = m_config.resize.size / std::max(img.rows, img.cols);
	cv::resize(img, img, cv::Size(), scaleFactor, scaleFactor, scaleFactor > 1.0f ? CV_INTER_CUBIC : CV_INTER_AREA);
}

cv::Point2i FaceDetector::detectEye(cv::CascadeClassifier& classifier, cv::Mat img)
{
	std::vector<cv::Rect> eyeRects;
	classifier.detectMultiScale(img, eyeRects,
		m_config.eyeDetect.scaleFactor,
		m_config.eyeDetect.minNeighbors,
		cv::CASCADE_FIND_BIGGEST_OBJECT,
		cv::Size((int)(img.cols * m_config.eyeDetect.minSizeFactor), (int)(img.rows * m_config.eyeDetect.minSizeFactor)),
		cv::Size((int)(img.cols * m_config.eyeDetect.maxSizeFactor), (int)(img.rows * m_config.eyeDetect.maxSizeFactor)));

	if (eyeRects.empty())
		return cv::Point2i(-1, -1);

	if (m_config.debug.drawEyes) {
		cv::rectangle(img, eyeRects[0], cv::Scalar(1), 3);
	}
	cv::Point2i eyePoint;
	eyePoint.x = eyeRects[0].x + eyeRects[0].width / 2;
	eyePoint.y = eyeRects[0].y + eyeRects[0].height / 2;
	return eyePoint;
}

void FaceDetector::geometryTransform(cv::Mat& img)
{
	std::vector<cv::Rect> eyeRects;

	int eyesTop = (int)(img.rows * m_config.geometry.eyeTopFactor);
	int eyesBottom = (int)(img.rows * m_config.geometry.eyeBottomFactor);

	cv::Mat topLeft(img, cv::Range(eyesTop, eyesBottom), cv::Range(0, img.cols/2));
	cv::Mat topRight(img, cv::Range(eyesTop, eyesBottom), cv::Range(img.cols/2, img.cols));

	cv::Point2i leftEye = detectEye(m_leftEyeClassifier, topLeft);
	cv::Point2i rightEye(-1, -1);
	bool glass = false;
	if (leftEye.x > 0) {
		rightEye = detectEye(m_rightEyeClassifier, topRight);
	}

	if (leftEye.x < 0 || rightEye.x < 0) {
		leftEye = detectEye(m_eyeglassClassifier, topLeft);
		if (leftEye.x > 0) {
			rightEye = detectEye(m_eyeglassClassifier, topRight);
			if (rightEye.x > 0) {
				glass = true;
			}
		}
	}

	if (leftEye.x < 0 || rightEye.x < 0)
		return;

	leftEye.y += eyesTop;
	rightEye.y += eyesTop;

	rightEye.x += img.cols/2;

	if (m_config.debug.drawEyes) {
		cv::circle(img, leftEye, 20, cv::Scalar(glass ? 127 : 1), 3);
		cv::circle(img, rightEye, 20, cv::Scalar(255), 3);
	}

	int dx = rightEye.x - leftEye.x;
	int dy = rightEye.y - leftEye.y;

	double angle = atan2(dy, dx) * 180.0 / PI;

	if (std::abs(angle) > m_config.geometry.maxAngle)
		return;

	int centerX = leftEye.x + dx / 2;
	int centerY = leftEye.y + dy / 2;

	cv::Mat rotationMat = cv::getRotationMatrix2D(cv::Point2f((float)centerX, (float)centerY), angle, 1.0);

	cv::Mat rotated;
	cv::warpAffine(img, rotated, rotationMat, img.size());

	img = rotated;
	img.forEach<uint8_t>([](uint8_t &p, const int*) { p = (p == 0 ? 127 : p);});
}

void FaceDetector::normalizeIllumination(cv::Mat& img)
{
	cv::Mat lookUpTable(1, 256, CV_8U);

	uint8_t* p = lookUpTable.ptr();
	double gamma = m_config.illumination.gamma;
	for( int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	cv::Mat res = img.clone();
	LUT(img, lookUpTable, res);
	img = res;

	int tileGridSize = m_config.illumination.claheTileSize;
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(m_config.illumination.clipLimit, cv::Size(tileGridSize, tileGridSize));
	clahe->apply(img, img);
}

void FaceDetector::applyMask(cv::Mat& img)
{
	cv::Mat mat(img.size(), CV_8U, cv::Scalar(0));
	cv::Point center(img.size().width/2, img.size().height/2);
	cv::ellipse(mat,
		center,
		cv::Size((int)(img.cols * m_config.mask.axeXFactor), (int)(img.rows * m_config.mask.axeYFactor)),
		0, 0, 360, cv::Scalar(255), CV_FILLED);
	cv::Mat masked(img.size(), CV_8U, cv::Scalar(127));
	img.copyTo(masked, mat);
	img = masked;
}

cv::Mat FaceDetector::processFace(cv::Mat img)
{
	denoise(img);
	resize(img);
	geometryTransform(img);
	normalizeIllumination(img);
	applyMask(img);
	if (m_config.debug.showResult) {
		cv::imshow("after", img);
	}
	if (m_config.debug.waitKey) {
		cv::waitKey(0);
	}
	return img;
}

std::vector<FaceDetector::FaceRegion> FaceDetector::detect(cv::Mat img)
{
	cv::Mat gray;

	if (img.channels() == 3) {
		cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	} else if (img.channels() == 1) {
		gray = img.clone();
	}

	std::vector<cv::Rect> faceRects;
	if (m_config.debug.noDetect) {
		faceRects.push_back(cv::Rect(0, 0, img.cols, img.rows));
	} else {
		m_classifier.detectMultiScale(gray, faceRects,
			m_config.faceDetect.scaleFactor, m_config.faceDetect.minNeighbors,
			0,
			cv::Size(m_config.faceDetect.minSizeX, m_config.faceDetect.minSizeY));
	}
	std::vector<FaceRegion> faces;
	faces.reserve(faceRects.size());
	for (cv::Rect& faceRect : faceRects) {
		if (std::abs(m_config.debug.extendRectFactor - 1.0) > 0.01) {
			double factor = m_config.debug.extendRectFactor;
			int dx = (int)(faceRect.width * factor - faceRect.width);
			int dy = (int)(faceRect.height * factor - faceRect.height);
			faceRect.x = std::max(0, faceRect.x - dx/2);
			faceRect.y = std::max(0, faceRect.y - dy/2);
			faceRect.width = std::min(img.cols - faceRect.x, faceRect.width + dx);
			faceRect.height = std::min(img.rows - faceRect.y, faceRect.height + dy);
		}
		cv::Mat roi(m_config.debug.preserveColor ? img : gray, faceRect);
		cv::Mat processed;
		if (m_config.debug.noProcessing) {
			processed = roi;
		} else {
			processed = processFace(roi);
		}
		faces.push_back({ faceRect, processed });
	}
	return faces;
}
