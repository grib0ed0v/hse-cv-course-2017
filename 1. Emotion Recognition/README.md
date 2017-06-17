Dashboard link: https://trello.com/b/DKflqDOB/emotion-recognition  
Development branch is "Emotion-Recognition"
---
Architecture:
* preprocessor  
    implements image preprocessing pipeline, which contains following building blocks:
    - tonal correction;
    - contrast correction; 
    - colour correction;
    - denoising;
    - light correction;
    - face detection via Haar cascades.  
    
    Folder resources contains cascades descriptors.
* recognizer   
  contains wrapper for work with Convolutional Neural Network with resources.
* starter  
  provides different sources for solving Emotion-Recognition problem:
    - single image;
    - folder with images;
    - video stream.  
    
    starts flowexecutor, which encapsulates image processing logic 
---

Program applies following command line arguments:  
* -i \<path to image\>
* -i \<path to folder with images\>
* -v \<id of the opened video capturing device\>
* -v \<path to video file\>

If no arguments specified, application start working with videostream from default(0) device.  
If both parameters(-i, -v) are specified or there is an issue with device or file, application will raise appropriate exception (IO or Illegal Argument).  

---

Prerequisites: 

* Anaconda with Python 3 
Download: https://www.continuum.io/downloads 
* OpenCV 3.2 version to work with DNN module 
To install: conda install -c conda-forge opencv=3.2.0
