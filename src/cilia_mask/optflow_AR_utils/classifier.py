import argparse
from joblib import Parallel, delayed
import json
import numpy as np
import os.path
from sklearn.feature_extraction.image import extract_patches_2d
from tqdm import tqdm

import run
from ocv import read_video

def compute_features(vidfile, flowdir, flow_type, window_size, percentile):
    """
    Computes feature vectors for the given video file.
    """
    # First load the frame and the optical flow target.
    prefix = vidfile.split("/")[-1].split(".")[0]
    x = read_video(vidfile, max_frames = 1)[0] # just frame 0 is all we need
    # ALSO, should any additional normalization happen with x?
    # - 0-centering pixel value mean
    # - unit variance pixel values
    # - etc?
    xx = np.array(x, dtype = np.float32)
    xx = (xx - xx.mean()) / xx.std()

    # Read in the optical flow, aka the "target".
    y = np.load(os.path.join(flowdir, prefix, flow_type, "flow", f"{prefix}_z.npy"))
    yy = np.zeros(y.shape, dtype = np.uint8)
    yy[y >= np.percentile(y, percentile)] = 1

    # Now pull out patches, reshape, and align with targets.
    x_patches = extract_patches_2d(xx, patch_size = (window_size, window_size))
    X_img = x_patches.reshape(x_patches.shape[0], -1)
    ind = window_size // 2
    y_img = yy[ind:-ind, ind:-ind].flatten()
    
    # All done...?
    return (prefix, X_img, y_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = '#3: Classification',
        epilog = 'lol moar cilia', add_help = 'How to use',
        prog = 'python classifier.py <args>')
    parser.add_argument('-x', '--data', required = True,
        help = 'Path to parent directory containing AVI videos.')
    parser.add_argument('-y', '--target', required = True,
        help = 'Path to parent directory with optical flow output folders.')
    parser.add_argument('-f', '--flow_type', choices = ['nvidia', 'opencv'], 
        required = True, help = "Type of optical flow features to analyze.")
    parser.add_argument('-o', '--output', required = True,
        help = "Output directory to save features and targets.")
    
    parser.add_argument('-t', '--as_test', required = False, action = "store_true",
        help = "If specified, this is a testing dataset. Defaults to training.")
    parser.add_argument('-w', '--win_size', type = int, default = 5,
        help = "Size of the windows to pull from the images. [DEFAULT: 5]")
    parser.add_argument('-s', '--sigma', type = float, default = 2.5,
        help = "Size of the 2D gaussian gradient magnitude filter. [DEFAULT: 2.5]")
    parser.add_argument('-p', '--percentile', type = int, default = 97,
        help = "Percentile threshold for optical flow. [DEFAULT: 97]")
    parser.add_argument("--n_jobs", required = False, default = 1, type = int,
        help = "Number of parallel jobs to run. -1 uses all cores. [DEFAULT: 1]")

    args = vars(parser.parse_args())

    # First, find the videos.
    # Filter out anything that does not start with "1" for training. If
    # testing, filter anything that does not start with a "7".
    traintest = "1" if not args['as_test'] else "7"
    videos = filter(lambda x: 
                             x.split("/")[-1].startswith(traintest), 
                             run.find_videos(args['data']))
    
    # Second, compute features from each video. This may take awhile.
    p = Parallel(n_jobs = args['n_jobs'], verbose = 30)
    data = p(delayed(compute_features)(
            video, args['target'], args['flow_type'], args['win_size'],
            args['percentile'])
        for video in videos)
    X = None
    Y = None
    print("Processing results into feature vectors...")
    mappings = {}
    for (pf, x, y) in tqdm(data):
        if X is None:
            X = x
            Y = y
            mappings[pf] = (0, y.shape[0])
            continue
        mappings[pf] = (Y.shape[0], y.shape[0])
        X = np.vstack([X, x])
        Y = np.append(Y, y)
    
    # SAVE IT OUT.
    print("Saving...")

    fname = f"{args['flow_type']}_{'test' if args['as_test'] else 'train'}"
    if not os.path.exists(args['output']):
        os.mkdir(args['output'])
    with open(os.path.join(args['output'], f"i_{fname}.json"), "w") as fp:
        json.dump(mappings, fp)
    np.save(os.path.join(args['output'], f"X_{fname}.npy"), X)
    np.save(os.path.join(args['output'], f"y_{fname}.npy"), Y)
    print("DONE.")