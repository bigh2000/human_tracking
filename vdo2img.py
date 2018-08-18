import cv2
import os

name = '/home/vdo-data3/Downloads/human_tracking/results/180817_174120/kalman_test.mp4'

cap = cv2.VideoCapture(name)
cnt = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cnt += 1
    print(cnt)
    path = '/home/vdo-data3/Pictures/%s_%s' % (os.path.splitext(os.path.basename(name))[0], os.path.splitext(os.path.basename(name))[1][1:])
    if not os.path.exists(path):
        os.mkdir(path)
    cv2.imwrite(os.path.join(path, '%d.jpg' % cnt), frame)