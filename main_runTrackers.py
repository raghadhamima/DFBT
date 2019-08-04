
import os
import sys
import shutil
import argparse
from glob import glob
import pandas as pd

env_path = os.path.join(os.path.dirname(__file__), 'pytracking')
if env_path not in sys.path:
    sys.path.append(env_path)

env_path = os.path.join(os.path.dirname(__file__), 'pysot')
if env_path not in sys.path:
    sys.path.append(env_path)


from utils import result_video
from utils import result_BB
from utils import run_tracker_pysot
from utils import run_tracker_pytracking
from utils import draw_first_BB
from utils import extract_frames
from utils import cut_video
from utils import download_video

def main():

    parser = argparse.ArgumentParser(
        description='Full pipeline to download Youtube video and infer deep trackers.')
    parser.add_argument('--overwrite', action='store_true',
                        help='remove existing folder')
    # parser.add_argument('--YT_ID', type=str, default=None,
    #                     help='ID from YT')
    # parser.add_argument('--start', type=int, default=0,
    #                     help='starting time of the frames')
    # parser.add_argument('--duration', type=int, default=None,
    #                     help='duration time of the frames')
    # parser.add_argument('--ID', type=int, default=0,
    #                     help='ID of the sequence')
    parser.add_argument('--path', type=str, default="/home/hamimart/TrackingNet2.0",
                        help='where to save the sequence/video/results')

    parser.add_argument('--CSV', type=str, default="TrackingNet 2.0 Test Set Extension - Final TrackingNet2.0.csv",
                    help='where to save the sequence/video/results')

    args = parser.parse_args()



    List_Sequences = os.path.join(args.path, args.CSV)

    df = pd.read_csv(List_Sequences)
    # print(df)

    for i, data in df.iterrows():
        # print(data)
        # print(data["Youtube_ID"])
        if i>6:
        # if (isinstance(data["Object_ID"], float)):
            # print(int(data["Object_ID"]))
            args.YT_ID = data["Youtube_ID"]
            args.start = data["Start Time"]
            args.duration = data["Duration"]
            args.ID = int(data["Object_ID"])
    # # remove previous BB
    # if args.overwrite:
    #     if os.path.exists(first_BB_path):
    #         shutil.rmtree(first_BB_path)

            # define all paths
            sequence_ID = args.YT_ID + "_" + str(args.ID)
            full_video_path = os.path.join(args.path, "Videos", args.YT_ID+'.mp4')
            sequence_path = os.path.join(args.path, "Sequences", sequence_ID)
            cut_video_path = os.path.join(sequence_path, 'video.mp4')
            frame_path = os.path.join(sequence_path, 'frames')
            frame_BB_path = os.path.join(sequence_path, 'frames_BB')
            first_BB_path = os.path.join(sequence_path, "initial_BB.txt")
            tracking_results_path = os.path.join(sequence_path, 'results')

            try:
                # Download the video
                download_video(args.YT_ID, full_video_path)

                #Cut the video
                cut_video(full_video_path, cut_video_path, args.start, args.duration)

                #extract frames
                extract_frames(cut_video_path, frame_path)

                # draw the first Bounding box
                if not os.path.exists(first_BB_path):
                    draw_first_BB(sequence_path, frame_path, first_BB_path)

                # Run trackers bsed on pysot
                run_tracker_pysot(args.YT_ID, args.ID, args.path)

                # Run trackers bsed on pytracking
                run_tracker_pytracking(frame_path, sequence_path, tracking_results_path)

                # show result bounding boxes
                result_BB(tracking_results_path, frame_path,
                          frame_BB_path, sequence_ID, args.YT_ID)

                # create results on video
                result_video(frame_BB_path, sequence_path)

            except:
                print("issue with", sequence_ID)


if __name__ == '__main__':
    main()
