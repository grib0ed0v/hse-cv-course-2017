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
