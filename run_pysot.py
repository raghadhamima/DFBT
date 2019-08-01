from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import argparse

import cv2
import torch
import numpy as np
from glob import glob


env_path = os.path.join(os.path.dirname(__file__), 'pysot')
if env_path not in sys.path:
    sys.path.append(env_path)


from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker


torch.set_num_threads(1)


def get_frames(args):

    images = glob(os.path.join(args.path, args.YT_ID + '_' + str(args.ID), 'frames', '*.pn*'))
    images = sorted(images,
                    key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for img in images:
        frame = cv2.imread(img)
        yield frame


def run_tracker_pysot(args):

    # load config
    config = f'pysot/experiments/{args.tracker_name}/config.yaml'
    snapshot = f'pysot/experiments/{args.tracker_name}/model.pth'

    cfg.merge_from_file(config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model.load_state_dict(torch.load(snapshot,
        map_location=lambda storage, loc: storage.cpu()))
    model.eval().to(device)

    # build tracker
    tracker = build_tracker(model)

    
    first_frame = True
    if args.YT_ID:
        video_name = args.YT_ID.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    # cv2.namedWindow(args.YT_ID, cv2.WND_PROP_FULLSCREEN)
    pred_bboxes = []
    for frame in get_frames(args):
        if first_frame:
            try:
                init_rect = np.loadtxt(str(os.path.join(args.path, args.YT_ID + '_' + str(args.ID), 'initial_BB.txt')), delimiter=',', dtype=np.float64)
                
            except:
                exit()
            tracker.init(frame, init_rect)
            pred_bboxes.append(init_rect)
            first_frame = False
        else:
            outputs = tracker.track(frame)
            bbox = list(map(int, outputs['bbox']))
            pred_bbox = outputs['bbox']
            pred_bboxes.append(pred_bbox)
            # cv2.rectangle(frame, (bbox[0], bbox[1]),
            #                 (bbox[0]+bbox[2], bbox[1]+bbox[3]),
            #                 (0, 255, 0), 3)
            # cv2.imshow(args.YT_ID, frame)

            # cv2.waitKey(40)
    model_path = os.path.join(args.path, args.YT_ID + '_' + str(args.ID), 'results', args.tracker_name)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    result_path = os.path.join(model_path, f'{video_name}.txt')
    with open(result_path, 'w') as f:
        for x in pred_bboxes:
            f.write(','.join([str(i) for i in x])+'\n')
    

import copy

def main():
    parser = argparse.ArgumentParser(description='tracking demo')
    parser.add_argument('--tracker_name', type=str, help='model name')
    parser.add_argument('--YT_ID', default='', type=str,
                        help='videos or image files')
    parser.add_argument('--ID', type=int, default=0, help='ID of the sequence')
    parser.add_argument('--path', type=str, default="/home/hamimart/Documents/Videos",help='where to save the sequence/video/results')


    args = parser.parse_args()

    run_tracker_pysot(args)


if __name__ == '__main__':
    main()

