Face recognition project
----
## Prerequisites
* [cmake](https://cmake.org/)
* [opencv](http://opencv.org/)
built with `face` module from [opencv\_contrib](https://github.com/opencv/opencv_contrib)
* C++11-capable compiler

----
## Building

### Linux and Windows
    mkdir _build && cd _build
    cmake -DOpenCV_DIR=/path/to/your/opencv/build ..
    cmake --build .

----
## Running

To get complete list of arguments, call the program with `--help`.

### Default
    cd bin
    ./iad_facerec

Recognizer will create configuration files in current folder and start recognizing from webcam with pre-trained model (`pretrained/facerec_config`). To specify other folder for configuration use argument `-c`. You need to set folders only once - recognizer will remember your choice.

You can update the model on the fly - press Space and select the face you want to add. You will be prompted for name in the terminal. Take some pictures and save the changes with Enter. Model will update with new data and you can save the changes on exit.

### Individual images
    ./iad_facerec -i input.jpg

You will get a list of people that were found in the image. You can also save face regions with names by specifying output folder:

    ./iad_facerec -i input.jpg -o out_dir


### Specifying your own dataset

If you wish, you may supply your own dataset and use it for training. Dataset is simply a folder consisting of subfolders with images. Name of subfolders will be used as a label. Image names do not matter. Example:

    dataset/Sergei Semenov/1.jpg
    dataset/Sergei Semenov/2.jpg
    dataset/Sergei Semenov/3.png
    dataset/Nikita Putikhin/1.jpg
    dataset/Nikita Putikhin/2.jpg
    dataset/Anna Yaushkina/1.jpg
    dataset/Anna Yaushkina/2.jpg
    
For better results, especially if your images are not cropped or contain more than one face, run preprocessing on your dataset:

    ./iad_facerec -pd -d dataset/ -o processed_dataset/
    
You will get warnings if your images confuse face detector. Remove (manually) false detections from processed folder and train the recognizer:

    ./iad_facerec -t -d dataset
    
----
## Configuraion and algorithm description

### Pipeline overview

There are several stages of processing an image:
 * Face detection
 * Preprocessing of individual faces
   * Conversion to grayscale
   * Denoising
   * Resizing
   * Rotation
   * Illumination correction
   * Masking
 * Recognition
 
 ### Face detection and preprocessing
 
 Both detection and preprocessing are done by FaceDetector (see `src/face_detector.h`). Its config is generated in `config_folder/detector_config.json`.
 
 Face and eye detections are done using via [cv::CascadeClassifier](http://docs.opencv.org/trunk/d1/de5/classcv_1_1CascadeClassifier.html). You can specify custom cascades in "cascades" section of the config.
 
 #### Detection
 
 Face detection by default is done with `haarcascade_frontalface_default.xml` from OpenCV. Parameters are controlled in `faceDetect` section:
 
 ```json
"faceDetect": {
    "scaleFactor": 1.2,
    "minNeighbors": 5,
    "minSizeX": 30,
    "minSizeY": 30
}
```

 #### Denoising
 
 Denoising is done via [cv::bilateralFilter](http://docs.opencv.org/master/d4/d86/group__imgproc__filter.html) that was chosen because it preserves edges.
 
 ```json
"denoise": {
    "d": 5,
    "sigmaColor": 50.0,
    "sigmaSpace": 50.0
}
```
 
 #### Resizing
 Resizes image so that its biggest dimension is at most `size` while preserving aspect rate.
 ```json
"resize": {
    "size": 120.0
}
```
 
 #### Rotation
Rotation is done so that eyes lie on horizontal line.

Eye detection is done within the rectangle that lies between `eyeTopFactor` and `eyeBottomFactor` portions of the image. First then left eye is detected, then right. If detection fails, it is repeated with alternative classifier.

```json
"eyeDetect": {
    "scaleFactor": 1.1,
    "minNeighbors": 6,
    "minSizeFactor": 0.16,
    "maxSizeFactor": 0.5
}
```
Then rotation angle that is needed make eyes horizontal is computed. If it exceeds `maxAngle` then eye detector had probably detected the nose and rotation is discarded.

 ```json
"geometry": {
    "eyeTopFactor": 0.25,
    "eyeBottomFactor": 0.666,
    "maxAngle": 30.0
}
```
 
 #### Illumination correction
 
Gamma correction is applied first (using [this tutorial](http://docs.opencv.org/trunk/d3/dc1/tutorial_basic_linear_transform.html)) and [CLAHE](http://docs.opencv.org/trunk/d5/daf/tutorial_py_histogram_equalization.html) is applied on top of it.
 
 ```json
"illumination": {
    "gamma": 0.7,
    "clipLimit": 2.0,
    "claheTileSize": 8
},
```
 
 #### Masking

 Simple elliptical mask is applied to image. It cuts out most if not all backgroud. Note that parameters need to be tuned to every face detector.

```json
"mask": {
    "axeXFactor": 0.4,
    "axeYFactor": 0.7
},
```
 
 ### Recognition
 
 Face recognition is done with (cv::face::FaceRecognizer)[http://docs.opencv.org/trunk/dd/d65/classcv_1_1face_1_1FaceRecognizer.html]. Local Binary Patterns Histograms (LBPH) is used because it supports updates. It can be configured with `config_folder/facerec_params_config.json` file.
