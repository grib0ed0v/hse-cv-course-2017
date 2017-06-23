# Template Matching
___
**Authors:** Daria Moshkina, Alexey Zelenov, Alexander Semin, Kirill Starkov, Alexey Navalny

**Insipred by:** Alexey Gruzdev

## Program description
Program find template on target image.
The algorithm uses SIFT features and FLANN knn matcher to detect matched keypoints between template and image
Then it applies Meansgift clustering on matched keypoints in order to determine estimated numbers of templates for an image
When we get the approximate quantity of templates, we use findHomography method and buila a bounding box for the first template, then we delete the first template from the image and repeat all steps described below as many times as the appriximate quantity of templates.

//TO DO Write fuc*ing description with screensots
### How to run:
* `cd ${git_repository}/6. Template Matching/`
* `python2 mathcing.py [args]`

Program applies following command line arguments:
* `--template_file, -t` - path to template image
* `--target_file, -f` - path to target image

### Prerequisites:
* [Anaconda]( https://www.continuum.io/downloads) with Python2
* [OpenCV 2.4.13](https://github.com/opencv/opencv/tree/2.4.13)
* [Scikit-learn](http://scikit-learn.org)
