import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--coords", help="comma seperated list of source points")
ap.add_argument('--cam', type=int, default=0)
ap.add_argument('--resolution', type=int, default=640)
ap.add_argument('--number', type=int, default=4)
ap.add_argument('--image', type=str, default="NULL")
args = ap.parse_args()

str_video = ['cam1_daiso', 'cam2_daiso']
pointIndex = 0
if(args.image=='NULL'):
    image = cv2.imread(str_video[args.cam] + '.jpg')
else:
    image = cv2.imread(args.image)

def draw_circle(event, x, y, flpags, param):
    image = param[0]
    pts = param[1]
    global pointIndex
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        pts.append([x, y])
        # print(pointIndex)

        pointIndex = pointIndex + 1

def show_window(image, string):
    global pointIndex
    while True:
        # print(pts,pointIndex-1)
        cv2.imshow(string, image)

        if cv2.waitKey(20) & 0xFF == 27:
            break
        if pointIndex == args.number:
            pointIndex = 0
            break
pts = []
cv2.namedWindow("img")
cv2.setMouseCallback("img", draw_circle, param=(image, pts))
show_window(image, "img")

filename = 'matrix_640/coor_' + str_video[args.cam] + '_' + str(args.resolution)
M = np.load(filename + '.npy')
M = np.array(M, np.float32)

f = open(filename + '.txt', 'r')
line = f.readline()
coor = line.split(' ')
coor_test2 = cv2.perspectiveTransform(
    np.array(pts, dtype=np.float32, ).reshape(1, -1, 2), M,
)

ori_back = cv2.imread("background.png")

for point in coor_test2[0]:
    a, b = tuple(point)
    x = (int(a) + int(coor[0]), int(b) + int(coor[1]))
    cv2.circle(ori_back, x, 5, (0,0,255), -1)

cv2.imshow("Test", ori_back)
cv2.waitKey(0)