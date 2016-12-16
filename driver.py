'''
OpenCV Documentation:  http://docs.opencv.org/3.0-beta/doc/py_tutorials/
Numpy and Scipy Documentation:  https://docs.scipy.org/doc/
'''

import numpy as np
import cv2
import imutils


class CodeDriver:
    WEBCAM_WIDTH = 1280
    WEBCAM_HEIGHT = 720
    EPSILON_SCALAR = 0.04  # precision scalar for approximating polygon contours
    RESIZE_WIDTH = 1280  # width in pixels for resizing images to a more managable size

    # defines constructor for CodeDriver class
    def __init__(self):
        pass

    # defines representation for Python Interpreter
    def __repr__(self):
        return self

    # defines main method for using the CodeDriver class
    def main(self):
        print("CodeDriver.main was called.")

        print("Calling start_webcam method...")
        self.start_webcam()

        # use opencv to read an image
        # originalimg = cv2.imread('testimages/testimg02.bmp')
        originalimg = cv2.imread('test.bmp')
        print("image was read from project directory.")

        # show originalimg for testing purposes
        cv2.imshow("Original image", originalimg)
        cv2.waitKey(0)  # wait for key press when <= 0

        # use imutils to resize originalimg to a smaller factor so that the shapes can be approximated better
        resizedimg = imutils.resize(originalimg, width=self.RESIZE_WIDTH)

        # show resizedimg for testing purposes
        cv2.imshow("Resized image", resizedimg)
        cv2.waitKey(0)  # wait for key press when <= 0

        # save the ratio of originalimg / resizedimg for drawing contours later
        ratio = originalimg.shape[0] / float(resizedimg.shape[0])

        print("Calling binarization method...")
        threshimg = self.binarization(resizedimg)

        # show threshimg for testing purposes
        cv2.imshow("Image after Binarization", threshimg)
        cv2.waitKey(0)  # wait for key press when <= 0

        print("Calling find_contours method...")
        contours = self.find_contours(threshimg)

        print("Calling identify_shape method...")
        self.identify_shape(originalimg, ratio, contours)

    def start_webcam(self):
        print("CodeDriver.start_webcam was called.")
        loop = True  # used for looping over webcam frames

        # link to the webcam and store that link to webcam variable
        webcam = cv2.VideoCapture(0)

        webcam.set(3, self.WEBCAM_WIDTH)  # set video feed width
        webcam.set(4, self.WEBCAM_HEIGHT)  # set video feed height

        print("Begin VideoCapture and show webcam frames in window.")

        while (loop):
            # capture image from the webcam frame-by-frame
            ret, frame = webcam.read()

            # operations on the frame from the webcam start here

            # display the resulting frame from the webcam
            cv2.imshow('frame', frame)

            # button listeners
            # c key listener - if c key is pressed then save the frame as a bmp image
            if cv2.waitKey(1) & 0xFF == ord('c'):
                print("Detected c key press.")
                cv2.imwrite("test.bmp", frame)  # writes image test.bmp to project directory
                print("image was written to test.bmp.")

            # q key listener - if q key is pressed then break out of the while loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Detected q key press.")
                loop = False

            # exit button listener - if 'frame' window's exit button is pressed then break out of the while loop
            if cv2.getWindowProperty('frame', 0) < 0:
                print("Detected window exit button press.")
                loop = False

        # (5) release the capture and then destroy all windows after leaving loop
        webcam.release()
        cv2.destroyAllWindows()
        print("Released webcam and used destroyAllWindows.")

    def binarization(self, image):
        print("CodeDriver.binarization was called.")

        # convert resizedimg to grayscale
        grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # blur grayimg to decrease pixel "noise" (for low quality images)
        gaussianblurred = cv2.GaussianBlur(grayimg, (5, 5), 0)

        # perform otsu binarization - threshhold gaussianblurred with THRESH_BINARY
        otsuthreshimg = cv2.threshold(gaussianblurred, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # perform bilateralFilter blur to remove noise while keeping edges sharp
        bilateralfilterblurred = cv2.bilateralFilter(otsuthreshimg, 9, 75, 75)

        # threshhold bilateralfilterblurred with THRESH_BINARY_INV
        threshimg = cv2.threshold(bilateralfilterblurred, 128, 255, cv2.THRESH_BINARY_INV)[1]

        return threshimg

    def find_contours(self, threshimg):
        print("CodeDriver.find_contours was called.")

        # find contours in threshimg
        # parameters - findContours(InputOutputArray image, int retrievalMode, int ContourApproximationMethod)
        foundcontours = cv2.findContours(threshimg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        countours = foundcontours[1]  # note if using opencv1 use foundcontours[0]
        return countours

    def identify_shape(self, originalimg, ratio, threshimgcontours):
        print("CodeDriver.identify_shape was called.")

        # for every group of contours (shape found) drawContours and putText over originalimg
        for c in threshimgcontours:
            moments = cv2.moments(c)
            centerx = int((moments['m10'] / (moments['m00'] + 0.00001)) * ratio)  # find x-axis center of shape
            centery = int((moments['m01'] / (moments['m00'] + 0.00001)) * ratio)  # find y-axis center of shape
            shapename = self.detect_shape(c)

            # adjust the contour bounds found on the resized image to the original image
            c = c.astype("float")  # numpy casting to float for more precise adjustment
            c = c * ratio  # adjust the contour
            c = c.astype("int")  # numpy casting to int for cv2.drawContours

            # parameters - drawContours(image, InputArrayOfArrays contours, color (R,G,B), thickness)
            cv2.drawContours(originalimg, [c], -1, (21, 101, 192), 2)

            # parameters - putText(image, string, Point (xpos, ypos), fontFace, fontScale, color (R,G,B), thickness)
            cv2.putText(originalimg, shapename, (centerx, centery), cv2.FONT_HERSHEY_PLAIN, 1, (192, 112, 21), 2)

            cv2.imshow("Original Image With ShapeName Overlay", originalimg)
        cv2.waitKey(0)

    def detect_shape(self, contour):
        print("CodeDriver.detect_shape was called.")

        shapename = "unknown"

        # approximate the contour along the perimeter of the shape
        perimeter = cv2.arcLength(contour, True)  # returns perimeter of the shape
        epsilon = self.EPSILON_SCALAR * perimeter  # accuracy parameter - the maximum distance from contour to approximation
        approximation = cv2.approxPolyDP(contour, epsilon, True)  # approximates contour along epsilon
        convexity = cv2.isContourConvex(contour)  # returns true if contour is convex / false is concave

        if len(approximation) == 3:
            # if the approximation finds 3 vertices then a triangle has been found
            shapename = "triangle"
            print "Triangle was found"
        elif len(approximation) == 4:
            # if the approximation finds 4 vertices then a quadrilateral has been found
            print "Quadrilateral was found"
            if convexity:
                (x, y, w, h) = cv2.boundingRect(approximation)  # find bounding box of contour
                aspectratio = w / float(h)  # find aspect ratio of width / height

                if 0.90 <= aspectratio <= 1.10:
                    shapename = "square"
                else:
                    shapename = "rectangle"
            else:
                shapename = "quadrilateral"

        elif len(approximation) == 5:
            # if the approximation finds 5 vertices then a pentagon has been found
            shapename = "pentagon"
            print "Pentagon was found"
        elif len(approximation) == 6:
            # if the approximation finds 4 vertices then a quadrilateral has been found
            shapename = "hexagon"
            print "Hexagon was found"
        elif len(approximation) == 7:
            # if the approximation finds 5 vertices then a pentagon has been found
            shapename = "heptagon"
            print "Heptagon was found"
        elif len(approximation) == 8:
            # if the approximation finds 4 vertices then a quadrilateral has been found
            shapename = "octagon"
            print "Octagon was found"
        elif len(approximation) == 9:
            # if the approximation finds 5 vertices then a pentagon has been found
            shapename = "nonagon"
            print "Nonagon was found"
        elif len(approximation) == 10:
            # if the approximation finds 4 vertices then a quadrilateral has been found
            shapename = "decagon"
            print "Decagon was found"
        else:
            # if the approximation finds less than 3 vertices or more than 12 then we assume a circle has been found
            shapename = "circle"
            print "Circle was found"

        return shapename


driver = CodeDriver()
driver.main()
