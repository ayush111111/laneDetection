import cv2 as cv

import cv2 as cv
import numpy as np

video = '/home/ayushlokare/Downloads/lane_vgt.mp4'
cap = cv.VideoCapture(video)

def blank(x):
    pass


def ROI(img, vertices):
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv.fillPoly(mask, vertices, match_mask_color)
    maskedImage = cv.bitwise_and(img, mask)
    return maskedImage


def drawLines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros_like(img)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(blank_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)
    img = cv.addWeighted(img, 0.7, blank_image, 0.3, 0.0, )
    return img


def process(image):
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, 0),
        (0, height),
        (width, height),
        (width, 0)
    ]
    cropped_image = ROI(image,
                        np.array([region_of_interest_vertices], np.int32), )
    lines = cv.HoughLinesP(cropped_image, 10, np.pi / 180, 50, np.array([]), 40, 100)

    imageWithLines = drawLines(cropped_image, lines)

    return imageWithLines


cv.namedWindow('Minimum')
cv.namedWindow('Maximum')
cv.createTrackbar('lh', 'Minimum', 50, 360, blank)
cv.createTrackbar('ls', 'Minimum', 100, 255, blank)
cv.createTrackbar('lv', 'Minimum', 100, 255, blank)

cv.createTrackbar('uh', 'Maximum', 140, 360, blank)
cv.createTrackbar('us', 'Maximum', 160, 255, blank)
cv.createTrackbar('uv', 'Maximum', 255, 255, blank)



while cap.isOpened():
    ret, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_white = np.array([52, 15, 191])
    upper_white = np.array([100, 91, 255])

    # mode 0 is for adjusting hsv values according to video
    mode = 0
    # mode 1 is test mode

    if mode == 0:

        sensitivity = 15

        l_h = cv.getTrackbarPos('lh', 'Minimum')
        l_s = cv.getTrackbarPos('ls', 'Minimum')
        l_v = cv.getTrackbarPos('lv', 'Minimum')

        u_h = cv.getTrackbarPos('uh', 'Maximum')
        u_s = cv.getTrackbarPos('us', 'Maximum')
        u_v = cv.getTrackbarPos('uv', 'Maximum')

        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])

        mask = cv.inRange(hsv, l_b, u_b)

    else:
        mask = cv.inRange(hsv, lower_white, upper_white)

    res = cv.bitwise_and(frame, frame, mask=mask)
    edges = cv.Canny(res, 10, 50)
    median = cv.medianBlur(res, 25)
    final = process(edges)

    cv.imshow('final', final)
    if cv.waitKey(1) & 0xff == ord('q'):
        break

cv.destroyAllWindows()
cap.release()
