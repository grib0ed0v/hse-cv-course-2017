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
    - scale;
    - face detection via Haar cascades.  
    Folder resources contains cascades descriptors.
* recognizer   
  contains wrapper for work with Convolutional Neural Network with resources.
* starter  
  provides different sources for solving Emotion-Recognition problem:
    - single image;
    - folder with images;
    - video stream.
---
