import cv2
import json
import os.path
import numpy as np

from kalman_tracker.sort import Sort

class kalman_rec:

    def __init__(self, video_path, path, w, h):
        self.path = path
        self.video_path = video_path
        self.json_name = 'alphapose-results.json'

        file = open(os.path.join(self.path, self.json_name), 'r')
        self.json_list = json.load(file)
        file.close()

        self.width = w
        self.height = h

        # number of frame
        self.n = len(self.json_list)

    def start(self):

        if not os.path.exists(os.path.join(self.path, 'top')):
            os.makedirs(os.path.join(self.path, 'top'))
        out_file = os.path.join(self.path, 'top/townCentreOut.top')

        # init_tracker
        tracker = Sort(use_dlib=False)  # create instance of the SORT tracker
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        out = cv2.VideoWriter(os.path.join(self.path, 'kalman_test.mp4'), fourcc, 30.0, (self.width, self.height))
        cap = cv2.VideoCapture(self.video_path)
        with open(out_file, 'w') as f_out:
            for frame in range(self.n):
                _, image = cap.read()
                bbox = []
                perid = [] ##
                if str(frame) in self.json_list:
                    for hm_idx in (self.json_list[str(frame)]):
                        part_line = {}
                        kp_preds = hm_idx['kp_coordinates']
                        kp_score = hm_idx['kp_scores']
                        # img_name = os.path.join(self.path, self.ori_dir_name, 'ori_%06d.jpg' % frame)
                        # image = cv2.imread(img_name)
                        # draw keypoints
                        for i in range(17):
                            if kp_score[i] <= 0.3:
                                continue
                            cor_x, cor_y = int(kp_preds[i][0]), int(kp_preds[i][1])
                            part_line[i] = (cor_x, cor_y)
                            # cv2.circle(image, (cor_x, cor_y), 4, (0, 0, 255), -1)

                            # for j, (start_p, end_p) in enumerate(l_pair):
                            #     if start_p in part_line and end_p in part_line:
                            #         start_xy = part_line[start_p]
                            #         end_xy = part_line[end_p]
                            #         cv2.line(image, start_xy, end_xy, line_color[j],
                            #                  int(0.5 * (kp_score[start_p] + kp_score[end_p])) + 1)

                        # draw bounding box and save image
                        if len(part_line) > 5:
                            bbox.append(hm_idx['bbox'])
                            perid.append(hm_idx['person_id']) ##
                    kal_boxes = np.array(bbox)
                    trackers = tracker.update(kal_boxes, image)

                    trackers = [[p]+b for p,b in zip(perid,trackers.tolist())]

                    for d in trackers:
                        f_out.write('%d,%d,%d,%d,x,x,x,x,%.3f,%.3f,%.3f,%.3f\n' % (d[5], frame, 1, 1, d[1], d[2], d[3], d[4]))
                        cv2.rectangle(image, (int(d[1]), int(d[2])), (int(d[3]), int(d[4])), (0, 255, 0), 2)
                        cv2.putText(image, '%d(%d)'%(d[5], d[0]), (int(d[1]), int(d[4])), cv2.FONT_HERSHEY_DUPLEX, 1,
                                    (0, 0, 255))
                out.write(image)