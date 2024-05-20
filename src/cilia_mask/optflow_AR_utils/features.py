import argparse
from joblib import Parallel, delayed
import numpy as np
import os.path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_flow(videodir, method):
    """
    Reads optical flow computed from previous methods.
    
    Parameters
    ----------
    videodir : string
        Path to the video directory containing optical flow.
    method : string
        String indicating the optical flow method used (e.g., nvidia).

    Returns
    -------
    video : array, shape (F, H, W, 2)
        NumPy array of computed optical flow.
    """
    vid_prefix = videodir.split("/")[-1]
    flow_path = os.path.join(videodir, method, "flow", f"{vid_prefix}_flow.npy")
    video = np.load(flow_path)
    return video

def save_flow(z, videodir, method):
    """
    Saves the new flow stack to the filesystem.
    
    Parameters
    ----------
    z : array, shape (W, H)
        NumPy array containing the optical flow features.
    videodir : string
        Path to the video directory containing optical flow.
    method : string
        String indicating the optical flow method used (e.g., nvidia).

    Returns
    -------
    None
    """
    vid_prefix = videodir.split("/")[-1]
    flow_path = os.path.join(videodir, method, "flow", f"{vid_prefix}_z.npy")
    np.save(flow_path, z)

    # Save some plots.
    img_path = os.path.join(videodir, method, "viz")
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    plt.imshow(z, cmap = "Blues", vmin = 0.0, vmax = 1.0)
    heatmap = os.path.join(img_path, f"{vid_prefix}_heat.png")
    plt.savefig(heatmap)
    plt.close()
    plt.hist(z.flatten(), bins = 100, range = (0.0, 1.0))
    hist = os.path.join(img_path, f"{vid_prefix}_hist.png")
    plt.savefig(hist)
    plt.close()

def joblib_func(child, flow_type):
    # Ok, pull out the NumPy optical flow.
    flow = read_flow(child, flow_type)

    # Compute vector amplitudes for each pixel.
    x, y = flow[:, :, :, 0], flow[:, :, :, 1]
    z = np.sqrt(x**2 + y**2).sum(axis = 0)
    z -= z.min() # Make the minimum 0.
    z /= z.max() # normalize everything to 1

    # Save it back.
    save_flow(z, child, flow_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = '#2: Compute Features',
        epilog = 'lol moar cilia', add_help = 'How to use',
        prog = 'python features.py <args>')
    parser.add_argument('-i', '--input', required = True,
        help = 'Path to parent directory with optical flow output folders.')
    parser.add_argument('-f', '--flow_type', choices = ['nvidia', 'opencv'],
         required = True, help = "Type of optical flow to analyze.")
    
    parser.add_argument("--n_jobs", required = False, default = 1, type = int,
        help = "Number of parallel jobs to run. -1 uses all cores. [DEFAULT: 1]")
    
    args = vars(parser.parse_args())

    # Get the list of folders. Keep only those that are actually directories.
    abs_paths = map(lambda x: os.path.join(args['input'], x), os.listdir(args['input']))
    children = filter(lambda x: os.path.isdir(x), abs_paths)

    # Run parallel tasks for each child.
    p = Parallel(n_jobs = args['n_jobs'], verbose = 30)
    output = p(delayed(joblib_func)(child, args['flow_type'])
               for child in children)