import os
import sys
import shutil
import argparse
from glob import glob

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


def main():

    parser = argparse.ArgumentParser(description='Full pipeline to download Youtube video and infer deep trackers.')
    parser.add_argument('--overwrite', action='store_true', 
                        help='remove existing folder')
    parser.add_argument('--YT_ID', type=str, default=None, 
                        help='ID from YT')
    parser.add_argument('--start', type=int, default=0,
                        help='starting time of the frames')
    parser.add_argument('--duration', type=int, default=None,
                        help='duration time of the frames')
    parser.add_argument('--ID', type=int, default=0, 
                        help='ID of the sequence')
    parser.add_argument('--path', type=str, default="/home/$USER/Documents/Videos",
                        help='where to save the sequence/video/results')
            
    args = parser.parse_args()


    # define all paths
    sequence_ID =args.YT_ID + "_" + str(args.ID)
    full_video_path = os.path.join(args.path, "Videos", args.YT_ID+'.mp4')
    sequence_path = os.path.join(args.path, "Sequences", sequence_ID)
    cut_video_path = os.path.join(sequence_path, 'video.mp4')
    frame_path = os.path.join(sequence_path, 'frames')
    frame_BB_path = os.path.join(sequence_path, 'frames_BB')
    first_BB_path = os.path.join(sequence_path, "initial_BB.txt")
    tracking_results_path = os.path.join(sequence_path, 'results')

    # remove previous results
    if (args.overwrite):
        if os.path.exists(sequence_path) and os.path.isdir(sequence_path):
            shutil.rmtree(sequence_path)

    # Download the video
    download_video(args.YT_ID, full_video_path)

    #Cut the video
    cut_video(full_video_path, cut_video_path, args.start, args.duration)
    
    #extract frames
    extract_frames(cut_video_path, frame_path)

    # draw the first Bounding box
    draw_first_BB(sequence_path, frame_path, first_BB_path)

    # Run trackers bsed on pysot
    run_tracker_pysot(args.YT_ID, args.ID, args.path)

    # Run trackers bsed on pytracking
    run_tracker_pytracking(frame_path, sequence_path, tracking_results_path, sequence_ID)

    # show result bounding boxes
    result_BB(tracking_results_path, frame_path,
              frame_BB_path, sequence_ID, args.YT_ID)

    # create results on video
    result_video(frame_BB_path, sequence_path)
    


if __name__ == '__main__':
    main()
