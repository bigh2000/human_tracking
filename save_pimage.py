import cv2
import json
import os

class save_pimage:

    def __init__(self, video_path, path):
        self.ori_dir_name = 'ori_image'
        self.pose_dir_name = 'pose_image'

        self.path = path
        self.video_path = video_path
        self.json_name = 'alphapose-results.json'

        file = open(os.path.join(self.path, self.json_name), 'r')
        self.json_list = json.load(file)
        file.close()

        # number of frame
        self.n = len(self.json_list)

    #save original image
    def ori(self):
        print("[[ Start Save Original Image ]]")
        cam = cv2.VideoCapture(self.video_path)
        if not os.path.exists(os.path.join(self.path, self.ori_dir_name)):
            os.mkdir(os.path.join(self.path, self.ori_dir_name))
        for frame in range(self.n):
            _, image = cam.read()
            cv2.imwrite(os.path.join(self.path, self.ori_dir_name, 'ori_%06d.jpg' % frame), image)
            frame += 1
        cam.release()
        print("[[ Finish Saving Original Image ]]")

    #save pose image
    def pose(self, bbox_save):
        print("[[ Start Save Pose Image ]]")
        # color
        RED = (0, 0, 255)
        GREEN = (0, 255, 0)
        BLUE = (255, 0, 0)
        CYAN = (255, 255, 0)
        YELLOW = (0, 255, 255)
        ORANGE = (0, 165, 255)
        PURPLE = (255, 0, 255)

        # color of pose part
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (5, 11), (6, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [RED, RED, RED, RED, RED, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, YELLOW, GREEN, GREEN, GREEN, GREEN,
                   GREEN, GREEN]
        line_color = [YELLOW, YELLOW, YELLOW, YELLOW, BLUE, BLUE, BLUE, BLUE, BLUE, PURPLE, PURPLE, RED, RED, RED, RED]


        if not os.path.exists(os.path.join(self.path, self.pose_dir_name)):
            os.mkdir(os.path.join(self.path, self.pose_dir_name))

        for frame in range(self.n):
            for hm_idx in (self.json_list[str(frame)]):
                part_line = {}
                kp_preds = hm_idx['kp_coordinates']
                kp_score = hm_idx['kp_scores']
                img_name = os.path.join(self.path, self.ori_dir_name, 'ori_%06d.jpg' % frame)
                image = cv2.imread(img_name)
                # draw keypoints
                for i in range(17):
                    if kp_score[i] <= 0.3:
                        continue
                    cor_x, cor_y = int(kp_preds[i][0]), int(kp_preds[i][1])
                    part_line[i] = (cor_x, cor_y)
                    cv2.circle(image, (cor_x, cor_y), 4, (0, 0, 255), -1)

                    for j, (start_p, end_p) in enumerate(l_pair):
                        if start_p in part_line and end_p in part_line:
                            start_xy = part_line[start_p]
                            end_xy = part_line[end_p]
                            cv2.line(image, start_xy, end_xy, line_color[j],
                                     int(0.5 * (kp_score[start_p] + kp_score[end_p])) + 1)

                # draw bounding box and save image
                if len(part_line) > 5:
                    if (bbox_save == True):
                        bbox = hm_idx['bbox']
                        cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)

                    cv2.imwrite(os.path.join(self.path, self.pose_dir_name, 'pose_%06d_%02d.jpg' % (frame, hm_idx['person_id'])),
                                image)
                else:
                    f = open(os.path.join(self.path, self.pose_dir_name, 'pose_%06d_%02d.txt' % (frame, hm_idx['person_id'])),
                             'w')
                    f.write('Not a person!')
                    f.close()
        print("[[ Finish Saving Pose Image ]]")













