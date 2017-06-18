set(SOURCES
	main.cpp
	dataset_manager.cpp
	dataset_manager.h
	face_detector.cpp
	face_detector.h
	face_recognizer.cpp
	face_recognizer.h
	image_providers.cpp
	image_providers.h
	dataset.h
	dataset.cpp
	util/log.h
	util/log.cpp
	util/fsutil.h
	util/fsutil.cpp
)

find_package(OpenCV REQUIRED)

add_executable(iad_facerec ${SOURCES})
target_link_libraries(iad_facerec
	opencv_core 
	opencv_videoio 
	opencv_highgui 
	opencv_imgproc 
	opencv_imgcodecs 
	opencv_face 
	opencv_objdetect
)
