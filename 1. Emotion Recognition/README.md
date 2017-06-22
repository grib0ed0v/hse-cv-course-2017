#Emotion Recognition Project
<b>Authors</b>: Maxim Viko, Inna Vasiljeva, Alexander Kozitsin, Elena Pavlova  
<b>Insipred by</b>: Alexey Gruzdev

###Prerequisites: 
* Anaconda with Python 3   
Download: https://www.continuum.io/downloads 
* OpenCV 3.2 version to work with DNN module   
To install: conda install -c conda-forge opencv=3.2.0
* Emotion Recognition Caffe model (<i>EmotiW_VGG_S.caffemodel</i>):  
Available here: https://gist.github.com/GilLevi/54aee1b8b0397721aa4b
* Emotion Recognition Caffe model configuration:  
Available here: https://gist.github.com/GilLevi/54aee1b8b0397721aa4b#file-deploy-txt

###How to run:  
* `cd <path_to_folder>/1.Emotion Recognition/starter`
* `<path_to_python>/python.exe run.py [args]`  
Available arguments described in section below.
---
Program applies following command line arguments:  
* -i \<path to image\>
* -i \<path to folder with images\>
* -v \<id of the opened video capturing device\>
* -v \<path to video file\>

If no arguments specified, application start working with videostream from default(0) device.  
If both parameters(-i, -v) are specified or there is an issue with device or file, application will raise appropriate exception.  
---
###High level architecture:
* preprocessor  
    implements image preprocessing pipeline, which contains following building blocks:
    - tonal correction;
    - contrast correction; 
    - colour correction;
    - denoising;
    - face detection via Haar cascades.  
* recognizer   
  contains wrapper for work with Convolutional Neural Network.
* starter  
  provides different sources for solving Emotion-Recognition problem:
    - single image;
    - folder with images;
    - video stream.  
    
    starts flowexecutor, which encapsulates image processing logic 
---
### Configuration
As this application suggests to use image preprocessors, there is a configuration for pipeline in <i>config.ini</i>.

This block specifies which processors will be used and their order. E.g.:
```
[ProcessorChain]
chain = ContrastProcessor ColorProcessor TonalProcessor NoiseProcessor
```
For each processor you might specify customized parameters. E.g:
```
[NoiseProcessor]
h = 5
hColor = 5
templateWindowSize = 5
searchWindowSize = 7
```
If processor configuration block has been missed, but processor was specified in chain - default parameters will be used.  

Also <i>config.ini</i> contains Face Detector configuration.
```
[FaceDetector]
cascadeClassifier = ./resources/haarcascade_frontalface_default.xml
scaleFactor = 1.3
minNeighbors = 4
minSize_x = 40
minSize_y = 40
```
Face Detector is one of the application key elements, but it is out of processors pipeline.  
<i>Note</i>: You might specify your own cascade classifier for Face Detection problem via `cascadeClassifier` parameter.

###Resources
Resource folder contains following files:
* <b>config.ini</b>  
described in previous section
* <b>haarcascade_frontalface_default.xml</b>  
default cascade for face detection from OpenCV (download: https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml)
* <b>EmotiW_VGG_S.caffemodel</b>  
description is provided in Prerequisites section
* <b>deploy.txt</b>  
description is provided in Prerequisites section

<i>Note</i>: Some files have large size. You will have to download them separately via specified links.