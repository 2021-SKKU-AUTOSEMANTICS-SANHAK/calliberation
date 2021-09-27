from utils.transform import four_point_transform
import numpy as np
import argparse
import cv2
from PIL import Image, ImageOps
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coords", help="comma seperated list of source points")
ap.add_argument('--video', type=int, default=0)
ap.add_argument('--resolution', type=int, default=640)
args = ap.parse_args()
image2 = cv2.imread("test/rest_plan3.JPG")
num_video = args.video
close1 = False
pointIndex = 0
AR = (740, 1280)
oppts = np.float32([[0, 0], [AR[1], 0], [0, AR[0]], [AR[1], AR[0]]])
str_video = ['ele', 'en', 'in']
ori_back = cv2.imread("test/rest_plan.JPG")

# function to select four points on a image to capture desired region
def draw_circle(event, x, y, flags, param):
    image = param[0]
    pts = param[1]
    global pointIndex
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        pts[pointIndex] = (x, y)
        # print(pointIndex)
        if pointIndex == 3:
            cv2.line(image, pts[0], pts[1], (0, 255, 0), thickness=2)
            cv2.line(image, pts[0], pts[2], (0, 255, 0), thickness=2)
            cv2.line(image, pts[1], pts[3], (0, 255, 0), thickness=2)
            cv2.line(image, pts[2], pts[3], (0, 255, 0), thickness=2)

        pointIndex = pointIndex + 1

def show_window(image, string):
    global pointIndex
    while True:
        # print(pts,pointIndex-1)
        cv2.imshow(string, image)

        if cv2.waitKey(20) & 0xFF == 27:
            break
        if pointIndex == 4:
            pointIndex = 0
            break

def matrix_test(event, x, y, flags, param):
    image = param[0]
    global pointIndex
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        temp_coor = [x, y]
        param[1].append(temp_coor)
        pointIndex += 1

folder = 'matrix_' + str(args.resolution) + '/'

name = 'test/' + str(args.resolution) + '/' + str_video[num_video] + '_cor.JPG'


cv2.namedWindow("img2")
pts = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
pts2 = [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
image = cv2.imread(name)
cv2.setMouseCallback("img2", draw_circle, param=(image2, pts2))
show_window(image2, "img2")
cv2.namedWindow("img")
cv2.setMouseCallback("img", draw_circle, param=(image, pts))
show_window(image, "img")

cv2.destroyAllWindows()
nparray2 = np.array(pts[:4], dtype="float32")
W = pts2[3][0] - pts2[0][0]
H = pts2[3][1] - pts2[0][1]
nparray1 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])
M = cv2.getPerspectiveTransform(nparray2, nparray1)
warped = cv2.warpPerspective(image, M, (W, H))

str_npy = folder + "coor_" + str_video[num_video] + '_' + str(args.resolution) + ".npy"
np.save(str_npy, M)

str_coor = folder + "coor_" + str_video[num_video] + '_' + str(args.resolution) + ".txt"
f = open(str_coor, 'w')
f.write(str(pts2[0][0]) + ' ' + str(pts2[0][1]))
f.close()

image = cv2.imread(name)
coor_test = []
cv2.namedWindow("matrix_test")
cv2.setMouseCallback("matrix_test", matrix_test, param=(image, coor_test))
show_window(image, "matrix_test")

coor_test2 = cv2.perspectiveTransform(
                np.array(coor_test, dtype=np.float32, ).reshape(1, -1, 2), M,
            )

for point in coor_test2[0]:
    a, b = tuple(point)
    x = (int(a) + int(pts2[0][0]), int(b) + int(pts2[0][1]))
    cv2.circle(ori_back, x, 5, (0, 255, 0), -1)

cv2.imshow("Test", ori_back)
#cv2.imshow("Warped", warped) #Uncommment if you want to see the transformed image
cv2.waitKey(0)

