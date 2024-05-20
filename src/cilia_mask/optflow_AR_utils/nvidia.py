import numpy as np

def flo2npy(flofile):
    """
    Converts a binary .flo file to a NumPy array.

    Arguments
    =========
    flofile : string
        Path to a .flo file on the filesystem.

    Returns
    =======
    flow : array, H x W x 2
        Optical flow array for the given file.
    """
    with open(flofile, "rb") as fp:
        x = np.fromfile(fp, np.float32, count = 1)[0]
        if x != 202021.25:
            print(f"ERROR: Incorrect flow number '{x}' found.")
            return None
        w = np.fromfile(fp, np.int32, count = 1)[0]
        h = np.fromfile(fp, np.int32, count = 1)[0]
        data = np.fromfile(fp, np.float32, count = (2 * h * w))
        return np.resize(data, (h, w, 2))
