from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
from youtube_dl import YoutubeDL
import argparse
import cv2
import numpy as np
import os
import torch
from glob import glob


env_path = os.path.join(os.path.dirname(__file__), 'pytracking')
if env_path not in sys.path:
    sys.path.append(env_path)

from pytracking.evaluation import Sequence, Tracker
from pytracking.evaluation.running import run_sequence

env_path = os.path.join(os.path.dirname(__file__), 'pysot')
if env_path not in sys.path:
    sys.path.append(env_path)


from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker



def download_video(args):

    #Download video from youtube:
    print("---------- starting YT_DL ---------- ")
    ydl = YoutubeDL({'outtmpl': args.path + '/%(id)s' + "_" + str(args.ID)})

    with ydl:
        result = ydl.extract_info(
            'https://www.youtube.com/watch?v={0}'.format(args.YT_ID),
            download = True)
    print("---------- YT_DL done ----------")

def cut_video(args):
    print("---------- starting cutting video ---------- ")
    # create the folder
    try:
        os.mkdir( os.path.join(args.path, args.YT_ID + '_' + str(args.ID) ))
    except FileExistsError:
        pass
    
    # cut the video
    os.system(f"ffmpeg -i {os.path.join(args.path, args.YT_ID + '_' + str(args.ID))}.mkv -qscale:v 2 -ss {args.start} -t {args.duration} -async 1 {os.path.join(args.path, args.YT_ID + '_' + str(args.ID), 'video.mkv')}")

    # remove full original video file
    # Handle errors while calling os.remove()
    try:
        os.remove(f"{os.path.join(args.path, args.YT_ID + '_' + str(args.ID))}.mkv")
    except:
        print("Error while deleting file ")
    print("---------- done cutting video ---------- ")

def extract_frames(args): 
    print("---------- start extract frames ---------- ")
    # create the frame folder
    try:
        os.mkdir(os.path.join(args.path, args.YT_ID + '_' + str(args.ID) , "frames"))
    except FileExistsError:
        pass

    # extracting frames
    os.system(f"ffmpeg -i {os.path.join(args.path, args.YT_ID + '_' + str(args.ID), 'video.mkv')}  -qscale:v 2 {os.path.join(args.path, args.YT_ID + '_' + str(args.ID), 'frames')}/%04d.png")

    
    print('---------- done frames extracting ----------')
   
def draw_first_BB(args):
    print("---------- start drawing ibox ---------- ")
    img = cv2.imread(os.path.join(args.path, args.YT_ID + '_' + str(args.ID), "frames","0001.png"))
    display_name = 'Display: Draw Initial Bounding Box' 
    cv2.imshow(display_name, img)
    valid_selection = False
    init_state = [0, 0, 0, 0]

    while not valid_selection:    
        frame_disp = img.copy()
        cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5, (0, 0, 0), 1)

        x, y, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
        init_state = [x, y, w, h]
        valid_selection = np.sum(init_state)

    print(init_state)
    output_path = os.path.join(args.path, args.YT_ID + '_' + str(args.ID), "initial_BB.txt")
    np.savetxt( output_path , np.array([init_state]), delimiter=',', fmt="%d")
    cv2.destroyAllWindows()
    print("---------- done drawing ibox ---------- ")
    return init_state

def run_tracker_pytracking(args):

    frames_path = os.path.join(args.path, args.YT_ID + '_' + str(args.ID),"frames")

    frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".png")]
    frame_list.sort(key=lambda f: int(f[:-4]))
    frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

    anno_path = os.path.join(args.path, args.YT_ID + '_' + str(args.ID),"initial_BB.txt")
    ground_truth_rect = np.loadtxt(str(anno_path), delimiter=',', dtype=np.float64).reshape(-1, 4)
    my_yt_sequence = Sequence(args.YT_ID + '_' + str(args.ID), frames_list, ground_truth_rect)
    os.makedirs(f"/home/hamimart/Documents/Videos/{args.YT_ID + '_' + str(args.ID)}/results", exist_ok=True)    


    for my_trackers in ['atom' , 'eco']:
        my_tracker = Tracker(f"{my_trackers}", "default")
        my_tracker.results_dir = f"/home/hamimart/Documents/Videos/{args.YT_ID + '_' + str(args.ID)}/results/{my_trackers}"
        os.makedirs(my_tracker.results_dir, exist_ok=True)    
        run_sequence(my_yt_sequence, my_tracker)

def run_tracker_pysot(args):

    for my_trackers in ['siamrpn_alex_dwxcorr' , 'siamrpn_r50_l234_dwxcorr' , 'siamrpn_mobilev2_l234_dwxcorr']:
        os.system(f"python pysot_try2.py --YT_ID '{args.YT_ID}' --ID {args.ID} --tracker_name '{my_trackers}'")

    
def result_BB (args):

    results_ATOM = np.loadtxt(os.path.join(args.path, args.YT_ID + '_' + str(args.ID), "results", "atom", f"{args.YT_ID + '_' + str(args.ID)}.txt"), dtype=np.int)
    results_ECO  = np.loadtxt(os.path.join(args.path, args.YT_ID + '_' + str(args.ID), "results", "eco" , f"{args.YT_ID + '_' + str(args.ID)}.txt"), dtype=np.int)
    results_siamrpn_alex_dwxcorr  = np.loadtxt(os.path.join(args.path, args.YT_ID + '_' + str(args.ID), "results", "siamrpn_alex_dwxcorr" , f"{args.YT_ID}.txt"), dtype=np.float, delimiter=',').astype(np.int)
    results_siamrpn_mobilev2_l234_dwxcorr  = np.loadtxt(os.path.join(args.path, args.YT_ID + '_' + str(args.ID), "results", "siamrpn_mobilev2_l234_dwxcorr" , f"{args.YT_ID}.txt"), dtype=np.float, delimiter=',').astype(np.int)
    results_siamrpn_r50_l234_dwxcorr  = np.loadtxt(os.path.join(args.path, args.YT_ID + '_' + str(args.ID), "results", "siamrpn_r50_l234_dwxcorr" , f"{args.YT_ID}.txt"), dtype=np.float, delimiter=',').astype(np.int)


    for i, (ATOM_BB, ECO_BB, siamrpn_alex_BB, siamrpn_mobile_BB, siamrpn_r50_BB ) in enumerate(zip(results_ATOM, results_ECO, results_siamrpn_alex_dwxcorr, results_siamrpn_mobilev2_l234_dwxcorr, results_siamrpn_r50_l234_dwxcorr)):

        frame_file = os.path.join(args.path, args.YT_ID + '_' + str(args.ID), "frames", f"{i+1:04d}.png")
        img = cv2.imread(frame_file)

        cv2.rectangle(img, (ATOM_BB[0], ATOM_BB[1]), (ATOM_BB[0]+ATOM_BB[2], ATOM_BB[1] + ATOM_BB[3]), (255, 255, 00), 2)
        cv2.rectangle(img, (ECO_BB[0], ECO_BB[1]), (ECO_BB[0]+ECO_BB[2], ECO_BB[1] + ECO_BB[3]), (00, 00, 255), 2)
        cv2.rectangle(img, (siamrpn_alex_BB[0], siamrpn_alex_BB[1]), (siamrpn_alex_BB[0]+siamrpn_alex_BB[2], siamrpn_alex_BB[1] + siamrpn_alex_BB[3]), (255, 255, 255), 2)
        cv2.rectangle(img, (siamrpn_mobile_BB[0], siamrpn_mobile_BB[1]), (siamrpn_mobile_BB[0]+siamrpn_mobile_BB[2], siamrpn_mobile_BB[1] + siamrpn_mobile_BB[3]), (255, 00, 255), 2)
        cv2.rectangle(img, (siamrpn_r50_BB[0], siamrpn_r50_BB[1]), (siamrpn_r50_BB[0]+siamrpn_r50_BB[2], siamrpn_r50_BB[1] + siamrpn_r50_BB[3]), (255, 00, 00), 2)

        os.makedirs(os.path.join(args.path, args.YT_ID + '_' + str(args.ID), "frames_BB"), exist_ok=True)    
        frame_BB_file = os.path.join(args.path, args.YT_ID + '_' + str(args.ID), "frames_BB", f"{i+1:04d}.png")
        cv2.imwrite(frame_BB_file, img)


def result_video (args):
    os.system(f"ffmpeg -i {os.path.join(args.path, args.YT_ID + '_' + str(args.ID), 'frames_BB')}/%04d.png -qscale:v 2 {os.path.join(args.path, args.YT_ID + '_' + str(args.ID), 'video_BB.mkv')}")
    os.system(f"xdg-open /home/hamimart/Documents/Videos/{args.YT_ID + '_' + str(args.ID)}/video_BB.mkv")

def main():

    parser = argparse.ArgumentParser(description='Download/Frame/BB.')
    parser.add_argument('--YT_ID', type=str, default=None, help='ID from YT')
    parser.add_argument('--start', type=int, default=0,help='starting time of the frames')
    parser.add_argument('--duration', type=int, default=None,help='duration time of the frames')
    parser.add_argument('--ID', type=int, default=0, help='ID of the sequence')
    parser.add_argument('--path', type=str, default="/home/hamimart/Documents/Videos",help='where to save the sequence/video/results')
            
    args = parser.parse_args()

    folder_name = args.YT_ID + "_" + str(args.ID)
    print(folder_name)

    # download_video(args)
    # cut_video(args)
    # extract_frames(args)
    # draw_first_BB(args)
    # run_tracker_pysot(args)
    # run_tracker_pytracking(args)
    result_BB(args)
    result_video(args)



# my_tracker = Tracker("atom", "default")
#     my_tracker.results_dir = "/home/hamimart/Documents/Videos/QzqIPPmY9gg/results/ATOM"
#     os.makedirs(my_tracker.results_dir, exist_ok=True)    
#     run_sequence(my_yt_sequence, my_tracker)

#     my_tracker = Tracker("eco", "default")
#     my_tracker.results_dir = "/home/hamimart/Documents/Videos/QzqIPPmY9gg/results/ECO"
#     os.makedirs(my_tracker.results_dir, exist_ok=True)    
#     run_sequence(my_yt_sequence, my_tracker)

    # draw BB into Frames
    # results_ATOM = np.loadtxt(os.path.join(path, args.YT_ID, "results", "ATOM", f"{args.YT_ID}.txt"), dtype=np.int)
    # results_ECO  = np.loadtxt(os.path.join(path, args.YT_ID, "results", "ECO" , f"{args.YT_ID}.txt"), dtype=np.int)

    # for i, (ATOM_BB, ECO_BB) in enumerate(zip(results_ATOM, results_ECO)):

    #     frame_file = os.path.join(path, args.YT_ID, "frames", f"{i+1:04d}.png")
    #     img = cv2.imread(frame_file)

    #     cv2.rectangle(img, (ATOM_BB[0], ATOM_BB[1]), (ATOM_BB[0]+ATOM_BB[2], ATOM_BB[1] + ATOM_BB[3]), (255, 255, 00), 2)
    #     cv2.rectangle(img, (ECO_BB[0], ECO_BB[1]), (ECO_BB[0]+ECO_BB[2], ECO_BB[1] + ECO_BB[3]), (00, 00, 255), 2)

    #     os.makedirs(os.path.join(path, args.YT_ID, "frames_BB"), exist_ok=True)    
    #     frame_BB_file = os.path.join(path, args.YT_ID, "frames_BB", f"{i+1:04d}.png")
    #     cv2.imwrite(frame_BB_file, img)


    # create video from frames with BB
    # os.system(f"ffmpeg -i {os.path.join(path, args.YT_ID, 'frames_BB')}/%04d.png {os.path.join(path, args.YT_ID, 'video_BB.mkv')}")
    # os.system(f"xdg-open /home/hamimart/Documents/Videos/{args.YT_ID}/video_BB.mkv")


if __name__ == '__main__':
    main()
