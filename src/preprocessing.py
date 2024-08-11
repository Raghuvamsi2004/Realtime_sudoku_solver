import cv2
import numpy as np


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
    return thresh


def warp_perspective(image, contour):
    points = contour.reshape(4, 2)
    points = sorted(points, key=lambda x: x[0])
    if points[0][1] > points[1][1]:
        points[0], points[1] = points[1], points[0]
    if points[2][1] < points[3][1]:
        points[2], points[3] = points[3], points[2]

    pts1 = np.float32([points[0], points[1], points[2], points[3]])
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(image, matrix, (450, 450))
    return result
