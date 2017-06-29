#pragma once

#include "face_detector.h"
#include "face_recognizer.h"

#include <opencv2/opencv.hpp>
#include <string>

class WebcamUI
{
public:
	WebcamUI(const std::string& windowName, cv::VideoCapture& cap, FaceDetector& detector, FaceRecognizer& recognizer);

	void videoLoop();

	void stop() { m_stop = true; }

private:
	enum Mode
	{
		Recognizing,
		Paused,
		DetectingFaces,
		AddingFace
	};

	const static std::map<Mode, std::string> s_modeToWindowTitle;

	static void onMouse(int event, int x, int y, int flags, void* data)
	{
		((WebcamUI*)data)->onMouse(event, x, y, flags);
	}
	void onMouse(int event, int x, int y, int flags);
	void processKey(int key);

	void promptForName();
	void addFace();
	void commitFaces();

	void setMode(Mode mode);
	void prepareFrame();
	void prepareRecognizingFrame();
	void preparePausedFrame();
	void prepareDetectingFacesFrame();
	void prepareAddingFaceFrame();

	int m_mouseX = 0;
	int m_mouseY = 0;

	bool m_stop = false;
	Mode m_mode = Mode::Recognizing;

	std::vector<cv::Mat> m_facesToAdd;
	std::string m_nameToAdd;

	cv::Mat m_frameOrig;
	cv::Mat m_frame;

	struct PredictedFace
	{
		FaceDetector::FaceRegion face;
		std::string pred;
	};
	std::vector<PredictedFace> m_predictedFaces;

	std::vector<FaceDetector::FaceRegion> m_detectedFaces;

	std::string m_windowName;
	cv::VideoCapture& m_cap;
	FaceDetector& m_detector;
	FaceRecognizer& m_recognizer;
};