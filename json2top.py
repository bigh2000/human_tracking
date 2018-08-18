import cv2
import json
import os.path
from collections import OrderedDict as od

path = '/home/vdo-ubuntu/Projects/test/human_tracking/results/180817_151720'
json_name = 'alphapose-results.json'
file = open(os.path.join(path, json_name), 'r')
json_list = json.load(file)
file.close()

if not os.path.exists(os.path.join(path, 'top')):
    os.makedirs(os.path.join(path, 'top'))
out_file = os.path.join(path, 'top/beforesorting.top')

cap = cv2.VideoCapture(os.path.join(path, 'AlphaPose_DVR_ch12_main_20180716130000_20180716140000_1.avi'))

frame = 0
while(True):
    _, image = cap.read()
    frame += 1
    if image is None:
        break

with open(out_file, 'w') as f_out:
    for f in range(frame):
        if str(f) in json_list:
            for person in json_list[str(f)]:
                f_out.write('%d,%d,%d,%d,x,x,x,x,%.3f,%.3f,%.3f,%.3f\n' % (person['person_id'], person['frame'], 1, 1, person['bbox'][0], person['bbox'][1], person['bbox'][2], person['bbox'][3]))


