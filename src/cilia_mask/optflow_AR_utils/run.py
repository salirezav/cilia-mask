import argparse
from joblib import Parallel, delayed
import numpy as np
import os.path
from pathlib import Path
import shutil
import subprocess as sp
from tqdm import tqdm

from nvidia import flo2npy
import ocv

def find_videos(dir):
    """
    Returns a list of absolute paths to videos found, given a parent directory.

    Parameters
    ----------
    dir : string
        A parent directory on the filesystem.

    Returns
    -------
    list[Path]
        A list of Paths to all the .avi files found
        in subdirectories of the parent directory.
    """
    videos = []
    for path in Path(dir).glob("*.avi"):
        # Only interested in the 1000 prefixed videos.
        videos.append(str(path.absolute()))
    return videos

def video2frames(video, outdir, imgtype = "png"):
    """
    Given the absolute path to a video, convert it to individual frames.

    Parameters
    ----------
    video : string
        Absolute path to a video file.
    outdir : string
        Directory for the image frames. If it does not exist, it is created.
    imgtype : string
        Categorical string indicator of the image suffix, i.e. the image type.
        Possible values include [png, jpg]. DEFAULT: png.

    Returns
    -------
    None
    """
    if imgtype not in ["png", "jpg"]:
        print(f"Unrecognized imgtype '{imgtype}', exiting.")
        return None

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    out = os.path.join(outdir, f"%05d.{imgtype}")
    command = f"ffmpeg -i {video} {out}"
    sp.run(command.split(), capture_output = True)

def frames2video(indir, namepattern, outfile):
    """
    Takes a folder of images with a numbering system in the filenames, and converts
    them to an ordered mp4. Something of the reverse of video2frames; not intended
    to be an "undo" but could, in theory, function that way.

    Parameters
    ----------
    indir: string
        Absolute path to the directory where the frames live.
    namepattern:
        Filename glob pattern to indicate the numbering system of the frames.
    outfile: string
        Path and filename to the output video file.

    Returns
    -------
    None
    """
    frames_in = os.path.join(indir, namepattern)
    command = f"ffmpeg -framerate 30 -pattern_type glob -i {frames_in} -c:v libx264 -pix_fmt yuv420p {outfile}"
    sp.run(command.split(), capture_output = True)

def optflow_nvidia(video_input,
                   frames_dir,
                   flow_output,
                   visual_flow_output = None,
    nvidia_bin = "/home/myid/spq/Optical_Flow_SDK_5.0.7/NvOFBasicSamples/build/bin/x64/AppOFCuda"):
    """
    Runs the NVIDIA optical flow.

    Parameters
    ----------
    video_input : string
        Absolute path to the video on which we're computing NVIDIA optical flow.
    frames_dir : string
        Absolute path to the directory where the video's frames should go.
    flow_output : string
        Directory where the output optical flow should go. If it does not exist, it is created.
    visual_flow_output : string
        If specified, the output directory where the flow images should go. DEFAULT: None.
    nvidia_bin : string
        Full path to the NVIDIA optical flow binary. Probably stick with the default.

    Returns
    -------
    vid_prefix : string
        The prefix identifier to a given video.
    """
    imgtype = "png"
    if not os.path.exists(flow_output):
        os.makedirs(flow_output)

    # First, we need to convert the input video to individual numbered frames.
    video2frames(video_input, frames_dir, imgtype = imgtype)

    # Second, configure the optical flow.
    flow_input = os.path.join(frames_dir, f"*.{imgtype}")
    visual_flow = "" if visual_flow_output is None else " --visualFlow=true"
    flow_output_tpl = os.path.join(flow_output, "flow")

    # Third, run it!
    command = f"{nvidia_bin} --input={flow_input} --output={flow_output_tpl} --preset=slow{visual_flow}"
    sp.run(command.split(), capture_output = True)

    # Fourth, clean up a bit. Like move the viz where it belongs.
    if visual_flow_output is not None:
        if not os.path.exists(visual_flow_output):
            os.makedirs(visual_flow_output)
        for viz in Path(flow_output).glob("*.png"):
            shutil.move(viz, visual_flow_output)

    # Fifth, convert the flo files into NumPy arrays.
    outnpy = []
    for flo in sorted(Path(flow_output).glob("*.flo")):
        npy = flo2npy(str(flo))
        outnpy.append(npy)
    outnpy = np.array(outnpy)
    vid_prefix = video_input.split("/")[-1].split(".")[0]
    np.save(os.path.join(flow_output, f"{vid_prefix}_flow.npy"), outnpy)

    # Sixth, and final, convert the visuals into movies.
    frames2video(visual_flow_output, "flow_00*_viz.png", 
                 os.path.join(visual_flow_output, f"{vid_prefix}_viz.mp4"))

    # Whew.
    return vid_prefix

def optflow_opencv(video_input,
                   flow_output,
                   visual_flow_output = None):
    """
    Compute OpenCV Farneback optical flow.

    Parameters
    ----------
    video_input : string
        Absolute path to the video on which we're computing OpenCV optical flow.
    flow_output : string
        Directory where the output optical flow should go. If it does not exist, it is created.
    visual_flow_output : string
        If specified, the output directory where the flow images should go. DEFAULT: None.

    Returns
    -------
    vid_prefix : string
        The prefix identifier to a given video.
    """
    if not os.path.exists(flow_output):
        os.makedirs(flow_output)
    vid_prefix = video_input.split("/")[-1].split(".")[0]

    # Read video, compute optical flow, and write it out.
    video = ocv.read_video(video_input)
    flow = ocv.optical_flow(video)
    np.save(os.path.join(flow_output, f"{vid_prefix}_flow.npy"), flow)

    # Do we need a visual representation of the flow?
    # if visual_flow_output is not None:
    #     if not os.path.exists(visual_flow_output):
    #         os.makedirs(visual_flow_output)
    return vid_prefix

def path_func(video, output, flow_type):
    prefix = video.split("/")[-1].split(".")[0]
    output_prefix = os.path.join(output, prefix)
    flow_dir = os.path.join(output_prefix, flow_type, "flow")
    viz_dir = os.path.join(output_prefix, flow_type, "viz")

    # What flow type are we computing?
    if flow_type == 'nvidia':
        frame_dir = os.path.join(output_prefix, "frames")
        out = optflow_nvidia(video, frame_dir, flow_dir, viz_dir)
    elif flow_type == 'opencv':
        out = optflow_opencv(video, flow_dir, viz_dir)
    
    # Return the prefix computed.
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Optical Flow',
        epilog = 'lol moar cilia', add_help = 'How to use',
        prog = 'python run.py <args>')
    parser.add_argument('-i', '--input', required = True,
        help = 'Path to AVI files for calculating optical flow.')
    parser.add_argument('-o', '--output', required = True,
        help = 'Path where output directory where stuff will be written.')
    parser.add_argument('-f', '--flow_type', choices = ['nvidia', 'opencv'], 
        required = True, default = "nvidia", help = "Type of optical flow to analyze.")
    
    parser.add_argument("--n_jobs", required = False, default = 1, type = int,
        help = "Number of parallel jobs to run. -1 uses all cores. [DEFAULT: 1]")

    args = vars(parser.parse_args())

    # First, find the videos.
    videos_to_convert = find_videos(args['input'])

    # Now, go through each video in turn.
    p = Parallel(n_jobs = args['n_jobs'], verbose = 30)
    output = p(delayed(path_func)(video, args['output'], args['flow_type'])
               for video in videos_to_convert)