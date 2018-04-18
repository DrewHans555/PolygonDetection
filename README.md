# PytholygonDetection
This repository contains code written in Python 2.7 for Blackburn College's Fall 2016 Artificial Intelligence class CS370.

PytholygonDetection captures an image from a webcam, processes the webcam image with OpenCV, scans the processed image for polygon contours, and then attempts to detect and classify all polygons in the image. During testing I put shapes cut from dark-colored construction paper on a white canvas and used my Logitech webcam to take images of the canvas. While this program does detect polygons up to 10 sides, I found that polygons with more than 7 sides are often misclassified.

### Prerequisites
* Python 2.7 or higher
* NumPy 1.12 or higher
* OpenCV-Python 3.0 or higher
* imutils 0.3.7 or higher

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
