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
	util/argparser.h
	util/argparser.cpp
)

find_package(OpenCV REQUIRED)

add_executable(iad_facerec ${SOURCES})

set(OPENCV_LIBS
	opencv_core
	opencv_videoio
	opencv_highgui
	opencv_imgproc
	opencv_imgcodecs
	opencv_face
	opencv_objdetect
)

target_link_libraries(iad_facerec
	${OPENCV_LIBS}
)

file (GLOB CASCADES
	"${CMAKE_SOURCE_DIR}/data/haarcascades/*.xml"
)

IF (WIN32)
	set (OUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>)

	set (LIB_SUFFIX $<$<CONFIG:Debug>:d>$<$<CONFIG:RelWithDebInfo>:d>)
	set (LIB_PATH ${OpenCV_DIR}/bin/$<$<CONFIG:Debug>:Debug>$<$<CONFIG:RelWithDebInfo>:Debug>$<$<CONFIG:Release>:Release>)
	set (LIB_VER ${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH})

	foreach(libname ${OPENCV_LIBS})
		add_custom_command(TARGET iad_facerec POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different ${LIB_PATH}/${libname}${LIB_VER}${LIB_SUFFIX}.dll ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIGURATION>
		)
	endforeach()
ELSE()
	set (OUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
ENDIF()

add_custom_command(TARGET iad_facerec POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E make_directory ${OUT_DIR}/haarcascades
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CASCADES} ${OUT_DIR}/haarcascades
)