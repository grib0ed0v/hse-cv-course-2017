#include "webcam_ui.h"

#include "util/log.h"

#include <iostream>

enum ScanCodes
{
	KeyEsc = 27,
	KeyLF = 10,
	KeyCR = 13,
	KeySpace = 32,
};

WebcamUI::WebcamUI(const std::string& windowName, cv::VideoCapture& cap, FaceDetector& detector, FaceRecognizer& recognizer)
	: m_windowName(windowName)
	, m_cap(cap)
	, m_detector(detector)
	, m_recognizer(recognizer)
{
	cv::namedWindow(m_windowName);
	cv::setMouseCallback(m_windowName, onMouse, this);
}

void WebcamUI::videoLoop()
{
	m_stop = false;
	setMode(Mode::Recognizing);

	while (true) {
		if (m_stop)
			return;

		prepareFrame();
		if (m_stop)
			return;

		cv::imshow(m_windowName, m_frame);
		int key = cv::waitKey(1);
		if (key != -1)
			processKey(key);
	}
}

const std::map<WebcamUI::Mode, std::string> WebcamUI::s_modeToWindowTitle = {
	{ Recognizing, "Recognizing. Press space to pause" },
	{ Paused, "Paused. Select face to add" },
	{ DetectingFaces, "Detecting Faces. Enter to save changes" },
	{ AddingFace, "Adding Face"},
};

void WebcamUI::setMode(Mode mode)
{
	if (s_modeToWindowTitle.count(mode))
		cv::setWindowTitle(m_windowName, s_modeToWindowTitle.at(mode));
	m_mode = mode;
}

void WebcamUI::onMouse(int event, int x, int y, int /*flags*/)
{
	m_mouseX = x;
	m_mouseY = y;
	if (event == cv::EVENT_LBUTTONUP) {
		if (m_mode == Mode::Paused) {
			cv::setWindowTitle(m_windowName, "Go to console!");
			promptForName();
		} else if (m_mode == Mode::AddingFace) {
			addFace();
		}
	}
}

void WebcamUI::promptForName()
{
	m_facesToAdd.clear();
	m_nameToAdd.clear();

	std::string response;

	cv::Mat activeFace;
	for (const auto& predFace : m_predictedFaces) {
		bool isActiveRect = predFace.face.rect.contains(cv::Point2i(m_mouseX, m_mouseY));
		if (!isActiveRect)
			continue;

		std::cout << "(rect@" << predFace.face.rect.x << ", " << predFace.face.rect.y << "): Enter name: ";

		std::getline(std::cin, response);

		if (response.empty()) {
			std::cout << "No name provided, cancelling" << std::endl;
			setMode(Mode::Recognizing);
			return;
		}

		m_nameToAdd = response;
		m_facesToAdd.push_back(predFace.face.image);

		break;
	}

	while (true) {
		std::cout << "Take more pictures? (y(es)/n(o)): ";
		std::getline(std::cin, response);

		std::transform(response.begin(), response.end(), response.begin(), tolower);
		if (response == "y" || response == "yes") {
			setMode(Mode::DetectingFaces);
			return;
		} else if (response == "n" || response == "no") {
			commitFaces();
			setMode(Mode::Recognizing);
			return;
		}
		std::cout << "Yes or no, please" << std::endl;
	}
}

void WebcamUI::addFace()
{
	for (const auto& face : m_detectedFaces) {
		bool isActiveRect = face.rect.contains(cv::Point2i(m_mouseX, m_mouseY));
		if (!isActiveRect)
			continue;

		m_facesToAdd.push_back(face.image);
		setMode(Mode::DetectingFaces);
		return;
	}
}

void WebcamUI::commitFaces()
{
	Dataset newData;
	newData.addImages(m_nameToAdd, m_facesToAdd);
	m_recognizer.update(newData);
	m_facesToAdd.clear();
	m_nameToAdd.clear();
}

void WebcamUI::prepareFrame()
{
	switch (m_mode) {
	case Recognizing:
		prepareRecognizingFrame();
		break;
	case Paused:
		preparePausedFrame();
		break;
	case DetectingFaces:
		prepareDetectingFacesFrame();
		break;
	case AddingFace:
		prepareAddingFaceFrame();
		break;
	default:
		logDebug() << "Invalid frame mode!";
		stop();
		return;
	}
}

void WebcamUI::prepareRecognizingFrame()
{
	m_cap >> m_frameOrig;
	m_frameOrig.copyTo(m_frame);
	m_predictedFaces.clear();
	std::vector<FaceDetector::FaceRegion> faceRegions = m_detector.detect(m_frame);
	for (const auto& face : faceRegions) {
		cv::rectangle(m_frame, face.rect, cv::Scalar(0, 255, 0));
		std::string pred = m_recognizer.predict(face.image);
		m_predictedFaces.push_back({ face, pred });
		int pos_x = std::max(face.rect.tl().x - 10, 0);
		int pos_y = std::max(face.rect.tl().y - 10, 0);
		putText(m_frame, pred, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, cv::Scalar(0, 255, 0), 2);
	}
}

void WebcamUI::preparePausedFrame()
{
	m_frameOrig.copyTo(m_frame);
	for (const auto& face : m_predictedFaces) {
		bool isActiveRect = face.face.rect.contains(cv::Point2i(m_mouseX, m_mouseY));
		cv::Scalar color = isActiveRect ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);

		cv::rectangle(m_frame, face.face.rect, color);

		std::string pred = face.pred;
		int pos_x = std::max(face.face.rect.tl().x - 10, 0);
		int pos_y = std::max(face.face.rect.tl().y - 10, 0);
		putText(m_frame, pred, cv::Point(pos_x, pos_y), cv::FONT_HERSHEY_PLAIN, 1.0, color, 2);
	}
}

void WebcamUI::prepareDetectingFacesFrame()
{
	m_cap >> m_frameOrig;
	m_frameOrig.copyTo(m_frame);
	m_detectedFaces = m_detector.detect(m_frame);
	for (const auto& face : m_detectedFaces) {
		cv::rectangle(m_frame, face.rect, cv::Scalar(0, 255, 0));
	}
}

void WebcamUI::prepareAddingFaceFrame()
{
	m_frameOrig.copyTo(m_frame);
	for (const auto& face : m_detectedFaces) {
		bool isActiveRect = face.rect.contains(cv::Point2i(m_mouseX, m_mouseY));
		cv::Scalar color = isActiveRect ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0);

		cv::rectangle(m_frame, face.rect, color);
	}
}

void WebcamUI::processKey(int key)
{
	switch (key) {
	case KeyEsc:
		stop();
		break;
	case KeySpace:
		switch (m_mode) {
		case Mode::Recognizing:
			setMode(Mode::Paused);
			break;
		case Mode::Paused:
			setMode(Mode::Recognizing);
			break;
		case Mode::DetectingFaces:
			setMode(Mode::AddingFace);
			cv::setWindowTitle(m_windowName, std::string("Select face for ") + m_nameToAdd);
			break;
		case Mode::AddingFace:
			setMode(Mode::DetectingFaces);
			break;
		default:
			break;
		}
		break;
	case KeyLF:
	case KeyCR:
		if (m_mode == Mode::DetectingFaces) {
			commitFaces();
			setMode(Mode::Recognizing);
		}
	default:
		break;
	}
}
