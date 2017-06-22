#include "face_detector.h"

#include "util/log.h"
#include "util/fsutil.h"

#include <opencv2/imgproc.hpp>
#include <cmath>

static const double PI = 3.14159265358979323846;

FaceDetector::FaceDetector(const std::string& configPath)
{
	std::string haars = fs::concatPath(configPath, "haarcascades");
	std::string lbps = fs::concatPath(configPath, "lbpcascades");
	m_classifier.load(fs::concatPath(haars, "haarcascade_frontalface_default.xml"));
	//m_classifier.load(fs::concatPath(lbps, "lbpcascade_frontalface_improved.xml"));
	//m_classifier.load(fs::concatPath(lbps, "lbpcascade_frontalface.xml"));

	//m_eyeClassifier.load(fs::concatPath(haars, "haarcascade_eye.xml"));
	m_leftEyeClassifier.load(fs::concatPath(haars, "haarcascade_mcs_lefteye.xml"));
	m_rightEyeClassifier.load(fs::concatPath(haars, "haarcascade_mcs_righteye.xml"));
	m_eyeglassClassifier.load(fs::concatPath(haars, "haarcascade_eye_tree_eyeglasses.xml"));
}

void FaceDetector::denoise(cv::Mat& img)
{
	cv::Mat dest = img.clone();
	cv::bilateralFilter(img, dest, 5, 50, 50);
	img = dest;
}

cv::Point2i FaceDetector::detectEye(cv::CascadeClassifier& classifier, cv::Mat img)
{
	std::vector<cv::Rect> eyeRects;
	classifier.detectMultiScale(img, eyeRects, 1.1, 6, cv::CASCADE_FIND_BIGGEST_OBJECT, cv::Size(img.cols/6, img.rows/6), cv::Size(img.cols/2, img.rows/2));

	if (eyeRects.empty())
		return cv::Point2i(-1, -1);

	cv::Point2i eyePoint;
	eyePoint.x = eyeRects[0].x + eyeRects[0].width / 2;
	eyePoint.y = eyeRects[0].y + eyeRects[0].height / 2;
	return eyePoint;
}

void FaceDetector::geometryTransform(cv::Mat& img)
{
	std::vector<cv::Rect> eyeRects;

	int eyesTop = (int)(img.rows / 4);
	int eyesBottom = (int)(img.rows / 1.5);

	cv::Mat topLeft(img, cv::Range(eyesTop, eyesBottom), cv::Range(0, img.cols/2));
	cv::Mat topRight(img, cv::Range(eyesTop, eyesBottom), cv::Range(img.cols/2, img.cols));

	cv::Point2i leftEye = detectEye(m_leftEyeClassifier, topLeft);
	cv::Point2i rightEye(-1, -1);
	if (leftEye.x > 0) {
		rightEye = detectEye(m_rightEyeClassifier, topRight);
	} else {
		leftEye = detectEye(m_eyeglassClassifier, topLeft);
		if (leftEye.x > 0) {
			rightEye = detectEye(m_eyeglassClassifier, topRight);
		}
	}

	if (leftEye.x < 0 || rightEye.x < 0)
		return;

	leftEye.y += eyesTop;
	rightEye.y += eyesTop;

	rightEye.x += img.cols/2;

	//cv::circle(img, leftEye, 20, cv::Scalar(255, 255, 255, 255), 5);
	//cv::circle(img, rightEye, 20, cv::Scalar(0), 5);

	int dx = rightEye.x - leftEye.x;
	int dy = rightEye.y - leftEye.y;

	double angle = atan2(dy, dx) * 180.0 / PI;

	if (std::abs(angle) > 30.0)
		return;

	int centerX = leftEye.x + dx / 2;
	int centerY = leftEye.y + dy / 2;

	cv::Mat rotationMat = cv::getRotationMatrix2D(cv::Point2f((float)centerX, (float)centerY), angle, 1.0);

	cv::Mat rotated;
	cv::warpAffine(img, rotated, rotationMat, img.size());

	img = rotated;
	//img.forEach<uint8_t>([](uint8_t &p, const int*) { p = (p == 0 ? 127 : p);});

	//int sum = 0;
	//img.forEach<uint8_t>([&sum](uint8_t &p, const int*) { sum += p; });

	//double weight = 127.0;
	//double coeff = weight / ((weight / img.size().area()) * sum);
	//logInfo() << sum << coeff;
	//img.forEach<uint8_t>([coeff](uint8_t &p, const int*) { p *= coeff; });
}

void FaceDetector::normalizeIllumination(cv::Mat& img)
{
	cv::Mat lookUpTable(1, 256, CV_8U);

	uint8_t* p = lookUpTable.ptr();
	double gamma = 0.7;
	for( int i = 0; i < 256; ++i)
		p[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
	cv::Mat res = img.clone();
	LUT(img, lookUpTable, res);
	//cv::imshow("before", img);
	img = res;
	//cv::imshow("gamma", img);

	float claheFactor = 0.1f;
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size((int)(claheFactor * img.cols), (int)(claheFactor * img.rows)));
	clahe->apply(img, img);
	//cv::imshow("clahe", img);
	//cv::equalizeHist(img, img);
}

void FaceDetector::applyMask(cv::Mat& img)
{
	cv::Mat mat(img.size(), CV_8U, cv::Scalar(0));
	cv::Point center(img.size().width/2, img.size().height/2);
	cv::ellipse(mat, center, cv::Size((int)(img.cols * 0.4), (int)(img.rows * 0.7)), 0, 0, 360, cv::Scalar(255), CV_FILLED);
	cv::Mat masked(img.size(), CV_8U, cv::Scalar(127));
	img.copyTo(masked, mat);
	img = masked;
}

cv::Mat FaceDetector::processFace(cv::Mat img)
{
	//cv::imshow("before", img);
	geometryTransform(img);
	normalizeIllumination(img);
	denoise(img);
	applyMask(img);
	//cv::imshow("processed", img);
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
	m_classifier.detectMultiScale(gray, faceRects, 1.2, 5, 0, cv::Size(30, 30));
	std::vector<FaceRegion> faces;
	faces.reserve(faceRects.size());
	for (const cv::Rect& faceRect : faceRects) {
		cv::Mat processed = processFace(cv::Mat(gray, faceRect));
		faces.push_back({ faceRect, processed });
	}
	return faces;
}
