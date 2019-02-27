import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    ##converting to grayscale to get a 2D image
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    ##using a gaussian blur to reduce noise. Optional since the Canny edge detector does that anyway
    blur = cv2.GaussianBlur(gray,(5,5),0)

    canny_detected = cv2.Canny(blur, 50, 150)
    return canny_detected

def region_of_interest(image):
    height = image.shape[0]
    polygon_coordinates = np.array([[(200,height),(1100,height),(550,250)]])
    
    ##creates an image with same dimensions as the passed image as argument
    ##except all the pixels are 0. 
    mask = np.zeros_like(image)
    
    ##fills the area determined by polygon_coordinates with pixel value 255
    cv2.fillPoly(mask,polygon_coordinates, 255)

    ##doing bitwise AND operation to extract the region of interest
    masked_image = cv2.bitwise_and(image, mask)
    
    return masked_image

def displayLines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10)
    return line_image


def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
             right_fit.append((slope, intercept))
    left_fit_avg = np.average(left_fit, axis = 0)
    right_fit_avg = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_avg)
    right_line = make_coordinates(image, right_fit_avg)
    ##print("Left line :", left_line)
    ##print("Right line:", right_line)
    return np.array([left_line, right_line])
'''


capture_video = cv2.VideoCapture("test2.mp4/test2.mp4")
while(capture_video.isOpened()):
    _,frame = capture_video.read()
    ##Using canny edge detector
    canny_detected = canny(frame)

    ##extracting region of interest
    roi_image = region_of_interest(canny_detected)

    ##detecting lines
    lines = cv2.HoughLinesP(roi_image,2, np.pi/180, 100, np.array([]), minLineLength = 40, maxLineGap = 5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_displayed = displayLines(frame, averaged_lines)
    final_image = cv2.addWeighted(frame, 0.8, line_displayed, 1, 1)
    cv2.imshow('Detected Image',final_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
capture_video.release()
cv2.destroyAllWindows()
