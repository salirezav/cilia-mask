import os.path

import numpy as np
from PIL import Image
from tqdm import tqdm

# This script is not meant to be beautiful. It's meant to get the job done.

data_dir = '/space/cilia/optflow'   # Where the data currently are.
new_dir = '/space/cilia/scipy23'    # Where the data are going.
window_size = 128                   # Square window size for tiling.
min_cilia = 0.025     # Required minimum ratio of cilia pixels to keep tile.
flow_types = ['nvidia', 'opencv']
percentile = 97

def tile(mat, winsize):
    """
    Returns the *index pairs* of where each tile starts.
    """
    indices = []
    for i in range(mat.shape[0] // winsize):
        for j in range(mat.shape[1] // winsize):
            # Make the tiles overlap if the dimensions don't
            # divide evenly into the window size. Equal sized
            # patches are more important than nonoverlapping.
            start_i = i * winsize
            if start_i + winsize > mat.shape[0]:
                start_i = mat.shape[0] - winsize
            start_j = j * winsize
            if start_j + winsize > mat.shape[1]:
                start_j = mat.shape[1] - winsize
            indices.append((start_i, start_j))
    return indices

n_tiles = 0
video_dirs = list(filter(lambda x: x.startswith("1") or x.startswith("7"), os.listdir(data_dir)))
for video in tqdm(video_dirs):
    abs_path = os.path.join(data_dir, video)
    image = np.asarray(Image.open(os.path.join(abs_path, "frames", "00001.png")), dtype = np.uint8)
    masks = {}
    for flow_type in flow_types:
        mask_path = os.path.join(abs_path, flow_type, "flow", f"{video}_z.npy")
        mask = np.load(mask_path)
        mask_mask = mask > np.percentile(mask.flatten(), percentile)
        mask_bin = np.zeros(shape = mask.shape, dtype = np.uint8)
        mask_bin[mask_mask] = 1
        masks[flow_type] = mask_bin
    
    outpath = os.path.join(new_dir, video)
    imagedir = os.path.join(outpath, "images")
    maskdir = os.path.join(outpath, "masks")
    if not os.path.exists(outpath):
        os.mkdir(outpath)
    if not os.path.exists(imagedir):
        os.mkdir(imagedir)
    if not os.path.exists(maskdir):
        os.mkdir(maskdir)
    for ft in flow_types:
        ftp = os.path.join(maskdir, ft)
        if not os.path.exists(ftp):
            os.mkdir(ftp)
        
    # We have the images and masks loaded. Let's tile them.
    for num, (i, j) in enumerate(tile(image, window_size)):
        skip_tile = False
        for ft, mask in masks.items():
            perc = mask[i:i + window_size, j:j + window_size].sum() / (window_size ** 2)
            if perc < min_cilia:
                skip_tile = True
        if skip_tile is True:
            continue
        
        # If we've reached this point, we're good to go.
        for ft, mask in masks.items():
            mask_tile = mask[i:i + window_size, j:j + window_size]
            np.save(os.path.join(maskdir, ft, f"{video}_f_{num}.npy"), mask_tile)

        image_tile = image[i:i + window_size, j:j + window_size]
        np.save(os.path.join(imagedir, f"{video}_i_{num}.npy"), image_tile)
        n_tiles += 1

print(f"{n_tiles} saved out.")

        

        