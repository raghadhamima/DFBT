from pysot.tracker.tracker_builder import build_tracker
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
from pytracking.evaluation.running import run_sequence
from pytracking.evaluation import Sequence, Tracker
from tqdm import tqdm

import sys
from youtube_dl import YoutubeDL
import argparse
import cv2
import numpy as np
import os
import torch
from glob import glob
import shutil


env_path = os.path.join(os.path.dirname(__file__), 'pytracking')
if env_path not in sys.path:
    sys.path.append(env_path)


env_path = os.path.join(os.path.dirname(__file__), 'pysot')
if env_path not in sys.path:
    sys.path.append(env_path)


def download_video(YT_ID, output_path):

    #Download video from youtube:
    print("---------- starting YT_DL ---------- ")

    if not os.path.exists(output_path):
        ydl = YoutubeDL({'outtmpl': output_path,
                        'format': 'mp4'})

        with ydl:
            result = ydl.extract_info(
                'https://www.youtube.com/watch?v={0}'.format(YT_ID),
                download=True)
    print("---------- YT_DL done ----------")


def cut_video(video_path, cut_video_path, start, duration, erase_input=False):
    print("---------- starting cutting video ---------- ")

    # create the folder
    os.makedirs(os.path.dirname(cut_video_path), exist_ok=True)

    # cut the video
    if not os.path.exists(cut_video_path):
        os.system(
            f"ffmpeg -y -i {video_path} -qscale:v 2 -ss {start} -t {duration} -async 1 {cut_video_path} -hide_banner")
    print("---------- done cutting video ---------- ")

    if erase_input:
        try:
            os.remove(
                f"{os.path.join(args.path, args.YT_ID + '_' + str(args.ID))}.mp4")
            print("---------- input erased ---------- ")
        except:
            print("Error while deleting file ")


def extract_frames(cut_video_path, frame_path):
    print("---------- start extract frames ---------- ")
    os.makedirs(frame_path, exist_ok=True)

    # extracting frames
    if not os.path.exists(f"{frame_path}/0001.png"):
        os.system(
            f"ffmpeg -n -i {cut_video_path}  -qscale:v 2 {frame_path}/%04d.png -hide_banner")

    print('---------- done frames extracting ----------')


def draw_first_BB(sequence_path, frame_path, first_BB_path, sequence_ID):
    print("---------- start drawing ibox ---------- ")
    img = cv2.imread(os.path.join(frame_path, "0001.png"))
    display_name = f'Display: Draw Initial Bounding Box {sequence_ID}'
    cv2.imshow(display_name, img)
    valid_selection = False
    init_state = [0, 0, 0, 0]

    while not valid_selection:
        frame_disp = img.copy()

        if os.path.exists(first_BB_path):
            annot = np.loadtxt(first_BB_path, dtype=np.int, delimiter=',')
            cv2.rectangle(frame_disp,  (annot[0], annot[1]),
                                (annot[0]+annot[2], annot[1] + annot[3]), 
                                (0, 255, 00), 2)  # Cyan

        cv2.putText(frame_disp, 'Select target ROI and press ENTER [ESC to use previous BB]', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.5, (0, 0, 0), 1)

        x, y, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
        init_state = [x, y, w, h]
        valid_selection = np.sum(init_state)
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
            return

    print(init_state)
    cv2.rectangle(img, (init_state[0], init_state[1]), (init_state[0] +
                                                        init_state[2], init_state[1] + init_state[3]), (00, 00, 255), 2)
    cv2.imwrite(os.path.join(sequence_path, "firstBB.png"), img)
    np.savetxt(first_BB_path, np.array([init_state]), delimiter=',', fmt="%d")
    cv2.destroyAllWindows()
    print("---------- done drawing ibox ---------- ")
    return init_state


def run_tracker_pysot(YT_ID, ID, path, tracking_results_path, overwrite):

    for my_trackers in tqdm(['siamrpn_alex_dwxcorr', 'siamrpn_r50_l234_dwxcorr', 'siamrpn_mobilev2_l234_dwxcorr']):
    # for my_trackers in tqdm(['siamrpn_r50_l234_dwxcorr']):
        if not os.path.exists(os.path.join(tracking_results_path, my_trackers)) or overwrite:
            os.system(
                f"python run_pysot.py --YT_ID '{YT_ID}' --ID {ID} --tracker_name '{my_trackers}' --path {path}")


def run_tracker_pytracking(frame_path, sequence_path, tracking_results_path, sequence_ID, overwrite):



    # frames_path = os.path.join(
    #     args.path, args.YT_ID + '_' + str(args.ID), "frames")

    frame_list = [frame for frame in os.listdir(
        frame_path) if frame.endswith(".png")]
    frame_list.sort(key=lambda f: int(f[:-4]))
    frames_list = [os.path.join(frame_path, frame) for frame in frame_list]

    anno_path = os.path.join(sequence_path, "initial_BB.txt")
    ground_truth_rect = np.loadtxt(
        str(anno_path), delimiter=',', dtype=np.float64).reshape(-1, 4)
    my_yt_sequence = Sequence(sequence_ID, frames_list, ground_truth_rect)
    #Result folder
    os.makedirs(tracking_results_path, exist_ok=True)

    for my_trackers in tqdm(['atom', 'eco']):

    # for my_trackers in tqdm(['atom']):
        my_tracker = Tracker(f"{my_trackers}", "default")
        #Path of the Result folder
        my_tracker.results_dir = os.path.join(tracking_results_path, my_trackers)
        os.makedirs(my_tracker.results_dir, exist_ok=True)
        run_sequence(my_yt_sequence, my_tracker)


def result_BB(tracking_results_path, frame_path, frame_BB_path, sequence_ID, YT_ID, overwrite):
    
    if os.path.exists(frame_BB_path) and not overwrite:
        return

    results_ATOM = np.loadtxt(os.path.join(tracking_results_path,
                              "atom", f"{sequence_ID}.txt"), dtype=np.int)
    results_ECO = np.loadtxt(os.path.join(tracking_results_path,
                                          "eco", f"{sequence_ID}.txt"), dtype=np.int)
    results_siamrpn_alex_dwxcorr = np.loadtxt(os.path.join(tracking_results_path,
                                                           "siamrpn_alex_dwxcorr", f"{YT_ID}.txt"), dtype=np.float, delimiter=',').astype(np.int)
    results_siamrpn_mobilev2_l234_dwxcorr = np.loadtxt(os.path.join(tracking_results_path,
                                                                    "siamrpn_mobilev2_l234_dwxcorr", f"{YT_ID}.txt"), dtype=np.float, delimiter=',').astype(np.int)
    results_siamrpn_r50_l234_dwxcorr = np.loadtxt(os.path.join(tracking_results_path,
                                                               "siamrpn_r50_l234_dwxcorr", f"{YT_ID}.txt"), dtype=np.float, delimiter=',').astype(np.int)

    for i, (ATOM_BB, ECO_BB, siamrpn_alex_BB, siamrpn_mobile_BB, siamrpn_r50_BB) in tqdm(enumerate(zip(results_ATOM, results_ECO, results_siamrpn_alex_dwxcorr, results_siamrpn_mobilev2_l234_dwxcorr, results_siamrpn_r50_l234_dwxcorr))):
    # for i, (ATOM_BB, siamrpn_r50_BB) in tqdm(enumerate(zip(results_ATOM, results_siamrpn_r50_l234_dwxcorr))):

        frame_file = os.path.join(frame_path, f"{i+1:04d}.png")
        img = cv2.imread(frame_file)

        cv2.putText(img, sequence_ID, (0, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)
        
        cv2.rectangle(img,  (ATOM_BB[0], ATOM_BB[1]), 
                            (ATOM_BB[0]+ATOM_BB[2], ATOM_BB[1] + ATOM_BB[3]), 
                            (255, 255, 00), 2)  # Cyan
        cv2.putText(img, 'ATOM', (0, 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 00), 1)

        cv2.rectangle(img,  (ECO_BB[0], ECO_BB[1]), 
                            (ECO_BB[0] + ECO_BB[2], ECO_BB[1] + ECO_BB[3]), 
                            (255, 00, 00), 2)  # Blue
        cv2.putText(img, 'ECO', (0, 60), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 00, 00), 1)

        cv2.rectangle(img,  (siamrpn_mobile_BB[0], siamrpn_mobile_BB[1]), 
                            (siamrpn_mobile_BB[0] + siamrpn_mobile_BB[2], siamrpn_mobile_BB[1] + siamrpn_mobile_BB[3]), 
                            (255, 255, 255), 2)  # White
        cv2.putText(img, 'SiamRPN_Mobile', (0, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1)

        cv2.rectangle(img,  (siamrpn_alex_BB[0], siamrpn_alex_BB[1]),
                            (siamrpn_alex_BB[0] +  siamrpn_alex_BB[2], siamrpn_alex_BB[1] + siamrpn_alex_BB[3]),
                            (255, 00, 255), 2)  # Magente
        cv2.putText(img, 'SiamRPN_AlexNet', (0, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 00, 255), 1)

        cv2.rectangle(img,  (siamrpn_r50_BB[0], siamrpn_r50_BB[1]), 
                            (siamrpn_r50_BB[0] + siamrpn_r50_BB[2], siamrpn_r50_BB[1] + siamrpn_r50_BB[3]), 
                            (00, 00, 255), 2)  # Red
        cv2.putText(img, 'SiamRPN_R50', (0, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (00, 00, 255), 1)

        os.makedirs(frame_BB_path, exist_ok=True)
        frame_BB_file = os.path.join(frame_BB_path, f"{i+1:04d}.png")
        cv2.imwrite(frame_BB_file, img)


def result_video(frame_BB_path, video_BB_path, overwrite):
    if overwrite or not os.path.exists(video_BB_path):
        os.system(f"ffmpeg -y -i {frame_BB_path}/%04d.png -qscale:v 2 {video_BB_path} -hide_banner")
