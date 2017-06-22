# Emotion Recognition Project
<b>Authors</b>: Maxim Viko, Inna Vasiljeva, Alexander Kozitsin, Elena Pavlova  
<b>Insipred by</b>: Alexey Gruzdev

### Prerequisites: 
* Anaconda with Python 3   
Download: https://www.continuum.io/downloads 
* OpenCV 3.2 version to work with DNN module   
To install: conda install -c conda-forge opencv=3.2.0
* Emotion Recognition Caffe model (<i>EmotiW_VGG_S.caffemodel</i>):  
Available here: https://gist.github.com/GilLevi/54aee1b8b0397721aa4b
* Emotion Recognition Caffe model configuration:  
Available here: https://gist.github.com/GilLevi/54aee1b8b0397721aa4b#file-deploy-txt

### How to run:  
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
### High level architecture:
* preprocessor  
    implements image preprocessing pipeline, which contains following building blocks:
    - <b>tonal correction</b> - process that changes the tonality of the image. If the gamma is greater than 1, then the color tone
    of the image becomes lighter. If the gamma less than 1, then the color tone becomes darker;
    - <b>contrast correction</b> - process that improves the contrast on image. This process apply CLAHE (Contrast Limited Adaptive Histogram Equalization) algorithm.
    The algorithm works only with grayscale images. Therefore the image is first converted to grayscale image, then apply CLAHE algorithm and then image
    is converted back to bgr (http://docs.opencv.org/trunk/d5/daf/tutorial_py_histogram_equalization.html);
    - <b>colour correction</b> - process that corrects the balance of colors in the image. These process apply gray-world white balance algorithm.
    This algorithm scales the values of pixels based on a gray-world assumption which states that the average of all channels should result in a gray image.
    It adds a modification which thresholds pixels based on their saturation value and only uses pixels below the provided threshold in finding average pixel values
    (http://docs.opencv.org/trunk/de/daa/group__xphoto.html);
    - <b>denoising</b> - process that performs image denoising (http://docs.opencv.org/3.0-beta/modules/photo/doc/denoising.html);
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

---
### Resources
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

---
### Preprocessors
In this section preprocessor's impact on picture is described. All of them applied independently to the same original image.  

Original Image
![alt text](https://raw.githubusercontent.com/grib0ed0v/hse-cv-course-2017/Emotion-Recognition/1.%20Emotion%20Recognition/materials/1.jpg "Original image")

Let's tune contrast with following code:
```
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv)
clahe = cv2.createCLAHE(self.clipLimit, self.tileGridSize)
v = clahe.apply(v)
hsv = cv2.merge((h, s, v))
cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB, image)
```
Contrasted Image
![alt text](https://raw.githubusercontent.com/grib0ed0v/hse-cv-course-2017/c9d9295bf827f536148fcf0941336c3d7b158784/1.%20Emotion%20Recognition/materials/contrast.jpg "Contrast image")

Let's apply Denoising via following code:
```
cv2.fastNlMeansDenoisingColored(image, image, self.h, self.hColor, self.templateWindowSize, self.searchWindowSize)
```
It should be noticed, that denoising is quite expensive operation, so there is a need to tune parameters carefully.

Denoising with following parameters: [h = 10, hColor = 10, templateWindowSize = 7, searchWindowSize = 21]
![alt text](https://raw.githubusercontent.com/grib0ed0v/hse-cv-course-2017/c9d9295bf827f536148fcf0941336c3d7b158784/1.%20Emotion%20Recognition/materials/denoising%2010%2C10%2C7%2C21.jpg "Denoised image")

Denoising with following parameters: [h = 5, hColor = 5, templateWindowSize = 5, searchWindowSize = 7]
![alt text](https://raw.githubusercontent.com/grib0ed0v/hse-cv-course-2017/c9d9295bf827f536148fcf0941336c3d7b158784/1.%20Emotion%20Recognition/materials/denoising%205%2C5%2C5%2C7.jpg "Denoised image")

Tonal correction might be done via following code:
```
new_gamma = 1.0 / self.gamma
adj_gamma = np.array([((i / 255.0) ** new_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")  # build table with their adjusted gamma values
cv2.LUT(image, adj_gamma, image)  # function LUT fills the output array with values from the adjusted gamma
```
Tonal correction with gamma = 1.4
![alt text](https://raw.githubusercontent.com/grib0ed0v/hse-cv-course-2017/c9d9295bf827f536148fcf0941336c3d7b158784/1.%20Emotion%20Recognition/materials/tonal%20correction%20gamma%20%3D%201.4.jpg "Tonal correction")

Color correction is done via following code:
```
whiteBalancer = cv2.xphoto.createGrayworldWB()
whiteBalancer.balanceWhite(image, image)
```
Pay attention to color correction, cause this might strongly affect result.  

Color correction for image
![alt text](https://raw.githubusercontent.com/grib0ed0v/hse-cv-course-2017/Emotion-Recognition/1.%20Emotion%20Recognition/materials/color%20correction.jpg "Color correction")

Here you might see all described processors in one gif.  
![alt text](https://github.com/grib0ed0v/hse-cv-course-2017/blob/Emotion-Recognition/1.%20Emotion%20Recognition/materials/processors_gif.gif "All in one")

---
### Convolutional Neural Network
<i>more info is available here: https://gist.github.com/GilLevi/54aee1b8b0397721aa4b</i>
<i>project page: http://www.openu.ac.il/home/hassner/projects/cnn_emotions/</i>  

This neural network applies RGB images(224x224) and produces array with emotion probabilities.  
It can predict: 'Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise' emotions.
According to the paper this network provides following results:  
Paper results (Confusion Matrix)  
![alt text](https://raw.githubusercontent.com/grib0ed0v/hse-cv-course-2017/Emotion-Recognition/1.%20Emotion%20Recognition/materials/Paper_results.JPG "Confusion Matrix")