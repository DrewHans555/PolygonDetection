'''
OpenCV Documentation:  http://docs.opencv.org/3.0-beta/doc/py_tutorials/
Numpy and Scipy Documentation:  https://docs.scipy.org/doc/
'''

import numpy as np
import cv2
import imutils


class CodeDriver:
    # global variables
    WEBCAM_WIDTH = 1280  # used to manually set the webcam capture width
    WEBCAM_HEIGHT = 720  # used to manually set the webcam capture height
    EPSILON_SCALAR = 0.04  # precision scalar for approximating polygon contours
    RESIZE_WIDTH = 600  # width in pixels for resizing images to a more managable size

    # defines constructor for CodeDriver class
    def __init__(self):
        pass

    # defines representation for Python Interpreter
    def __repr__(self):
        return self

    # defines main method for using the CodeDriver class
    def main(self):
        print("CodeDriver.main was called.")

        print("Calling CodeDriver.start_webcam method...")
        self.start_webcam()

        # use opencv to read an image
        # original_image = cv2.imread('testimages/testimg02.bmp')
        original_image = cv2.imread('test.bmp')
        print("image was read from project directory.")

        # show originalimg for testing purposes
        cv2.imshow("Original Image Captured From Webcam", original_image)
        cv2.waitKey(0)  # wait for key press when <= 0

        # use imutils to resize originalimg to a smaller factor so that the shapes can be approximated better
        resized_image = imutils.resize(original_image, width=self.RESIZE_WIDTH)

        # save the ratio of originalimg / resizedimg for drawing contours later
        ratio = original_image.shape[0] / float(resized_image.shape[0])

        print("Calling CodeDriver.binarization method...")
        binary_image = self.binarization(resized_image)

        # show binaryimg for testing purposes
        cv2.imshow("Image After Resize, Grayscale, GaussianBlur, OtsuThreshold, and Inversion", binary_image)
        cv2.waitKey(0)  # wait for key press when <= 0

        print("Calling CodeDriver.find_contours method...")
        list_of_contours = self.find_contours(binary_image)

        print("Calling CodeDriver.identify_shape method...")
        self.identify_shape(original_image, ratio, list_of_contours)

    def start_webcam(self):
        print("CodeDriver.start_webcam was called.")
        loop = True  # used for looping over webcam frames

        webcam = cv2.VideoCapture(0)  # link to the webcam and store that link to webcam variable
        webcam.set(3, self.WEBCAM_WIDTH)  # set video feed width
        webcam.set(4, self.WEBCAM_HEIGHT)  # set video feed height

        print("Begin VideoCapture and show webcam frames in window.")

        while loop:
            ret, frame = webcam.read()  # capture image from the webcam frame-by-frame
            cv2.imshow('frame', frame)  # display the resulting frame from the webcam

            # key-press listeners
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

        webcam.release()  # release the capture after leaving loop
        cv2.destroyAllWindows()  # destroy all open windows before proceeding
        print("Released webcam and used destroyAllWindows.")

    def binarization(self, resized_image):
        print("CodeDriver.binarization was called.")

        # convert resized_image to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # blur gray_image with a Gaussian Function to decrease pixel "noise"
        blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # binarize blur_image with Otsu's Threshold Method
        otsuthreshold_image = cv2.threshold(blur_image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        # invert otsuthreshold_image to make background black and polygons white (for findContours to work)
        invert_image = cv2.threshold(otsuthreshold_image, 128, 255, cv2.THRESH_BINARY_INV)[1]

        return invert_image

    def find_contours(self, binary_image):
        print("CodeDriver.find_contours was called.")
        
        # parameters - findContours(InputOutputArray image, int retrievalMode, int ContourApproximationMethod)
        found_contours = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        countours = found_contours[1]  # note if using opencv1 use foundcontours[0]
        return countours

    def identify_shape(self, original_image, ratio, list_of_contours):
        print("CodeDriver.identify_shape was called.")

        # for every group of contours (potential polygon) in list_of_contours
        for c in list_of_contours:
            moments = cv2.moments(c)  # a weighted average (moment) of the image pixels' intensities
            centerx = int((moments['m10'] / (moments['m00'] + 0.00001)) * ratio)  # find x-axis center of polygon
            centery = int((moments['m01'] / (moments['m00'] + 0.00001)) * ratio)  # find y-axis center of polygon
            
            print("Calling CodeDriver.detect_shape method...")
            shapename = self.detect_shape(c)

            # adjust the contour bounds found on the resized_image to the original_image
            c = c.astype("float")  # numpy casting to float for more precise adjustment
            c = c * ratio  # apply adjustment to the contour
            c = c.astype("int")  # numpy casting to int for cv2.drawContours

            # parameters - drawContours(image, InputArrayOfArrays contours, contourIdx, color (R,G,B), thickness)
            cv2.drawContours(original_image, [c], -1, (21, 101, 192), 2)

            # parameters - putText(image, string, Point (xpos, ypos), fontFace, fontScale, color (R,G,B), thickness)
            cv2.putText(original_image, shapename, (centerx, centery), cv2.FONT_HERSHEY_PLAIN, 1, (192, 112, 21), 2)

            cv2.imshow("Original Image With ShapeName & Contour Overlay", original_image)
        cv2.waitKey(0)  # wait for key press when <= 0

    def detect_shape(self, contours):
        print("CodeDriver.detect_shape was called.")
        shapename = "unknown"  # initialize shapename

        # approximate the contour along the perimeter of the shape
        perimeter = cv2.arcLength(contours, True)  # returns perimeter of the shape
        epsilon = self.EPSILON_SCALAR * perimeter  # accuracy parameter - the maximum distance from contour to approximation
        approximation = cv2.approxPolyDP(contours, epsilon, True)  # approximates the real contours with epsilon (note true for close shapes, false for open shapes)

        if len(approximation) == 3:
            # if the approximation finds 3 vertices then a triangle has been found
            shapename = "triangle"
            print "Triangle was found"
        elif len(approximation) == 4:
            # if the approximation finds 4 vertices then a quadrilateral has been found
            shapename = "quadrilateral"
            print "Quadrilateral was found"
        elif len(approximation) == 5:
            # if the approximation finds 5 vertices then a pentagon has been found
            shapename = "pentagon"
            print "Pentagon was found"
        elif len(approximation) == 6:
            # if the approximation finds 4 vertices then a hexagon has been found
            shapename = "hexagon"
            print "Hexagon was found"
        elif len(approximation) == 7:
            # if the approximation finds 5 vertices then a heptagon has been found
            shapename = "heptagon"
            print "Heptagon was found"
        elif len(approximation) == 8:
            # if the approximation finds 4 vertices then a octagon has been found
            shapename = "octagon"
            print "Octagon was found"
        elif len(approximation) == 9:
            # if the approximation finds 5 vertices then a nonagon has been found
            shapename = "nonagon"
            print "Nonagon was found"
        elif len(approximation) == 10:
            # if the approximation finds 4 vertices then a decagon has been found
            shapename = "decagon"
            print "Decagon was found"
        else:
            # if the approximation finds less than 3 vertices or more than 10 then we assume an unknown polygon
            shapename = "unknown"
            print "Unknown Polygon"

        return shapename


driver = CodeDriver()
driver.main()
