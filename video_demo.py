import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader import VideoLoader, DataWriter, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
import os.path
from tqdm import tqdm
import time
from fn import getTime
import cv2

from pPose_nms import pose_nms, write_json

from save_pimage import save_pimage
from datetime import datetime
from write_top import write_top

from kalman_tracker.sort import Sort

from kalman_rec import kalman_rec

#option settings
def optset(opt):
    # opt.video = '/home/vdo-data3/Videos/pedestrian_test_video.mp4'
    # opt.video = '/home/vdo-data3/Videos/pedestrian_test_video_2.mp4'
    # opt.video = '/home/vdo-data3/Videos/DVR_ch12_main_20180716130000_20180716140000_1.avi'
    opt.video = '/home/vdo-data3/Videos/cut_1.mp4'
    # '/home/vdo-data3/Videos/homeshopping/homeshopping_refrigerator.mp4'
    # '/home/vdo-data3/Videos/TownCentreXVID.avi'
    opt.outputpath = 'results'
    opt.save_video = True
    opt.save_ori = False
    opt.save_pose = False
    opt.draw_bbox = False
    args = opt
    args.dataset = 'coco'

    return args


#main
if __name__ == "__main__":

    # total_time = 0.0 #
    # total_frames = 0 #
    #
    # display = opt.display #
    # use_dlibTracker = opt.use_dlibTracker #
    # saver = opt.saver #

    args = optset(opt)

    videofile = args.video
    mode = args.mode

    args.outputpath = os.path.join(args.outputpath, ('%s'%datetime.today().strftime('%y%m%d_%H%M%S')))

    #init_detector
    # detector = GroundTruthDetections()

    #init_tracker
    # tracker = Sort(use_dlib=use_dlibTracker)  # create instance of the SORT tracker

    if not os.path.exists(args.outputpath):
        os.mkdir(args.outputpath)
    
    if not len(videofile):
        raise IOError('Error: must contain --video')

    # Load input video
    fvs = VideoLoader(videofile).start()
    (fourcc,fps,frameSize) = fvs.videoinfo()

    # Data writer
    save_path = os.path.join(args.outputpath, 'AlphaPose_'+videofile.split('/')[-1].split('.')[0]+'.avi')
    writer = DataWriter(args.save_video, save_path, cv2.VideoWriter_fourcc(*'XVID'), fps, frameSize).start()

    # Load YOLO model
    print('Loading YOLO model..')
    det_model = Darknet("yolo/cfg/yolov3.cfg")
    det_model.load_weights('models/yolo/yolov3.weights')
    det_model.net_info['height'] = args.inp_dim
    det_inp_dim = int(det_model.net_info['height'])
    assert det_inp_dim % 32 == 0
    assert det_inp_dim > 32
    det_model.cuda()
    det_model.eval()

    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

    runtime_profile = {
        'ld': [],
        'dt': [],
        'dn': [],
        'pt': [],
        'pn': []
    }

    im_names_desc = tqdm(range(fvs.length()))
    for i in im_names_desc:
        start_time = getTime()

        (img, orig_img, inp, im_dim_list) = fvs.read()

        ckpt_time, load_time = getTime(start_time)
        runtime_profile['ld'].append(load_time)
        with torch.no_grad():
            # Human Detection
            img = Variable(img).cuda()
            im_dim_list = im_dim_list.cuda()

            prediction = det_model(img, CUDA=True)
            ckpt_time, det_time = getTime(ckpt_time)
            runtime_profile['dt'].append(det_time)
            # NMS process
            dets = dynamic_write_results(prediction, opt.confidence,
                                 opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
            if isinstance(dets, int) or dets.shape[0] == 0:
                writer.save(None, None, None, None, None, orig_img, im_name=int(i))
                continue
            im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
            boxes = dets[:, 1:5].cpu()

            # for bbox in boxes:
            #     cv2.rectangle(orig_img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)

            # kal_boxes = boxes
            # trackers = tracker.update(kal_boxes.numpy(), orig_img)
            #
            # if not os.path.exists('top'):
            #     os.makedirs('top')
            # out_file = 'top/townCentreOut.top'
            #
            # with open(out_file, 'w') as f_out:
            #     for d in trackers:
            #         f_out.write('%d,%d,%d,%d,x,x,x,x,%.3f,%.3f,%.3f,%.3f\n' % (d[4], i, 1, 1, d[0], d[1], d[2], d[3]))
            #         cv2.rectangle(orig_img, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 2)
            #         cv2.putText(orig_img, 'id = %d' % (d[4]), (int(d[0]), int(d[3])), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

            scores = dets[:, 5:6].cpu()
            ckpt_time, detNMS_time = getTime(ckpt_time)
            runtime_profile['dn'].append(detNMS_time)

            # Pose Estimation
            inps, pt1, pt2 = crop_from_dets(inp, boxes)
            inps = Variable(inps.cuda())

            hm = pose_model(inps)
            ckpt_time, pose_time = getTime(ckpt_time)
            runtime_profile['pt'].append(pose_time)

            writer.save(boxes, scores, hm.cpu().data, pt1, pt2, orig_img, im_name=int(i))
            
            ckpt_time, post_time = getTime(ckpt_time)
            runtime_profile['pn'].append(post_time)

        # TQDM
        im_names_desc.set_description(
            'load time: {ld:.4f} | det time: {dt:.4f} | det NMS: {dn:.4f} | pose time: {pt:.4f} | post process: {pn:.4f}'.format(
                ld=np.mean(runtime_profile['ld']), dt=np.mean(runtime_profile['dt']), dn=np.mean(runtime_profile['dn']),
                pt=np.mean(runtime_profile['pt']), pn=np.mean(runtime_profile['pn']))
        )

    if (args.save_img or args.save_video) and not args.vis_fast:
        print('===========================> Rendering remaining images in the queue...')
        print('===========================> If this step takes too long, you can enable the --vis_fast flag to use fast rendering (real-time).')
    while(writer.running()):
        pass
    writer.stop()

    #sort and arrange
    final_result = writer.results()

    #write json
    write_json(final_result, args.outputpath)
    # kal_dets = write_json(final_result, args.outputpath)

    h, w, c = orig_img.shape

    save = save_pimage(args.video, args.outputpath)
    kal = kalman_rec(args.video, args.outputpath, w, h)

    # write_top(final_result, args.outputpath)

    if (opt.save_ori):
        save.ori()
    if (opt.save_pose):
        save.pose(opt.draw_bbox)

    kal.start()

