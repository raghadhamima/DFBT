import os
import sys
import shutil
import argparse
from glob import glob
import pandas as pd
from time import sleep
env_path = os.path.join(os.path.dirname(__file__), 'pytracking')
if env_path not in sys.path:
    sys.path.append(env_path)

env_path = os.path.join(os.path.dirname(__file__), 'pysot')
if env_path not in sys.path:
    sys.path.append(env_path)

from utils import download_video
from utils import cut_video
from utils import extract_frames
from utils import draw_first_BB
from utils import run_tracker_pytracking
from utils import run_tracker_pysot
from utils import result_BB
from utils import result_video


def run_single_sequence(args):

    # define all paths
    sequence_ID = args.YT_ID + "_" + str(args.ID)
    full_video_path = os.path.join(args.path, "Videos", args.YT_ID+'.mp4')
    sequence_path = os.path.join(args.path, "Sequences", sequence_ID)
    cut_video_path = os.path.join(sequence_path, 'video.mp4')
    frame_path = os.path.join(sequence_path, 'frames')
    frame_BB_path = os.path.join(sequence_path, 'frames_BB')
    first_BB_path = os.path.join(sequence_path, "initial_BB.txt")
    tracking_results_path = os.path.join(sequence_path, 'results')
    video_BB_path = os.path.join(sequence_path, 'video_BB.mkv')

    # remove previous results
    if (args.overwrite):
        # if os.path.exists(sequence_path) and os.path.isdir(sequence_path):
        #     # shutil.rmtree(sequence_path)
        if os.path.exists(cut_video_path):
            os.remove(cut_video_path)
        if os.path.exists(video_BB_path):
            os.remove(video_BB_path)
        if os.path.exists(frame_path) and os.path.isdir(frame_path):
            shutil.rmtree(frame_path)
        if os.path.exists(frame_BB_path) and os.path.isdir(frame_BB_path):
            shutil.rmtree(frame_BB_path)
        if os.path.exists(tracking_results_path) and os.path.isdir(tracking_results_path):
            shutil.rmtree(tracking_results_path)

    print(sequence_ID)
    # Download the video
    download_video(args.YT_ID, full_video_path)

    #Cut the video
    cut_video(full_video_path, cut_video_path, args.start, args.duration)

    #extract frames
    extract_frames(cut_video_path, frame_path)


    first_BB_path_shared = os.path.join(
        "/run/user/1001/gvfs/smb-share:server=10.68.74.21,share=tn2", "Sequences", sequence_ID, "initial_BB.txt")

    # draw the first Bounding box
    if not os.path.exists(first_BB_path) and os.path.exists(first_BB_path_shared):

        first_BB_path_shared = os.path.join(
            "/run/user/1001/gvfs/smb-share:server=10.68.74.21,share=tn2", "Sequences", sequence_ID, "initial_BB.txt")
        # first_BB_path_shared = os.path.join(
        #     "/home/giancos/Documents/Videos", "Sequences", sequence_ID, "initial_BB.txt")
        
        # if os.path.exists(first_BB_path_shared):
        shutil.copyfile(first_BB_path_shared, first_BB_path)
            # elif: args.just_BB
        # else:
            # print("please Draw!")
            # draw_first_BB(sequence_path, frame_path, first_BB_path, sequence_ID)

    # if args.just_BB:
    #     draw_first_BB(sequence_path, frame_path,
    #                   first_BB_path, sequence_ID)

    # draw BB if asked to
    if args.draw_BB:
        draw_first_BB(sequence_path, frame_path,
                      first_BB_path, sequence_ID)


    if args.run_trackers:
        # if at that stage, no BB, then need to draw one
        if not os.path.exists(first_BB_path):
            draw_first_BB(sequence_path, frame_path,
                          first_BB_path, sequence_ID)


        # Run trackers bsed on pysot
        run_tracker_pysot(args.YT_ID, args.ID, args.path,
                          tracking_results_path, args.overwrite)

        # Run trackers bsed on pytracking
        run_tracker_pytracking(frame_path, sequence_path,
                               tracking_results_path, sequence_ID, args.overwrite)

        # show result bounding boxes
        result_BB(tracking_results_path, frame_path,
                  frame_BB_path, sequence_ID, args.YT_ID, args.overwrite)


    if args.play_video:
        # create results on video
        result_video(frame_BB_path, video_BB_path, args.overwrite)

        # run video
        os.system(f"xdg-open {video_BB_path}")
        sleep(args.duration+args.sleep_between_videos)




def main():

    parser = argparse.ArgumentParser(
        description='Full pipeline to download Youtube video and infer deep trackers.')
    parser.add_argument('--overwrite', action='store_true',
                        help='remove existing folder')
    parser.add_argument('--draw_BB', action='store_true',
                        help='run the trackers')
    parser.add_argument('--run_trackers', action='store_true',
                        help='run the trackers')
    parser.add_argument('--play_video', action='store_true',
                        help='playthe video afterwards')

    parser.add_argument('--YT_ID', type=str, default=None, 
                        help='ID from YT')
    parser.add_argument('--start', type=int, default=0,
                        help='starting time of the frames')
    parser.add_argument('--duration', type=int, default=0,
                        help='duration time of the frames')
    parser.add_argument('--ID', type=int, default=0, 
                        help='ID of the sequence')
    parser.add_argument('--path', type=str, default="/home/$USER/Documents/Videos",
                        help='where to save the sequence/video/results')

    parser.add_argument('--CSV', type=str, default=None,
                        help='where to save the sequence/video/results')
    parser.add_argument('--first_id', type=int, default=0,
                        help='first line of CSV file')
    parser.add_argument('--last_id', type=int, default=10000,
                        help='last line of CSV file')
    parser.add_argument('--single_id', type=int, default=0,
                        help='first line of CSV file')
    parser.add_argument('--sleep_between_videos', type=int, default=10,
                        help='first line of CSV file')
    # parser.add_argument('--first_BB_folder_pool', type=str, default=0,
    #                     help='first line of CSV file')

    args = parser.parse_args()

    if args.single_id is not 0:
        args.first_id = args.single_id
        args.last_id = args.single_id
        args.sleep_between_videos = 0


    if args.CSV is None:
        run_single_sequence(args)

    else:
        df = pd.read_csv(os.path.join(args.path, args.CSV))
        for i, data in df.iterrows():
            # print("Entry", i)
            # print(data["Youtube_ID"])
            # print(data)
            if data["Duration"] > 0 and i >= args.first_id and i <= args.last_id:
                # if isinstance(data["Approval Silvio"], str):
                #     if "no" in data["Approval Silvio"]:
                #         continue


                # if (isinstance(data["Object_ID"], float)):
                # print(data["Youtube_ID"])
                args.YT_ID = data["Youtube_ID"]
                args.start = int(data["Start Time"]/1000)
                args.duration = int(data["Duration"])
                args.ID = int(data["Object_ID"])

                print(args)
                run_single_sequence(args)
    


if __name__ == '__main__':
    main()
