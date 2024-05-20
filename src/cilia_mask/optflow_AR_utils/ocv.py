import numpy as np

import cv2

def read_video(vidfile, max_frames = -1):
    """
    Reads an AVI video file as a NumPy array.

    Parameters
    ----------
    vidfile : string
        Absolute string path to the AVI file.
    max_frames : int
        Maximum number of frames to read. Reads all frames in the video
        if -1 (default), or if exceeds actual number of frames.

    Returns
    -------
    video : array, shape (F, H, W) np.int32
        Grayscale video with F frames, H rows, and W columns.
    """
    vp = cv2.VideoCapture(vidfile)
    F = int(vp.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0 and max_frames < F:
        F = max_frames
    H = int(vp.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(vp.get(cv2.CAP_PROP_FRAME_WIDTH))
    video = np.zeros(shape = (F, H, W), dtype = np.int32)

    i = 0
    while vp.isOpened() and i < F:
        ret, frame = vp.read()
        if not ret:
            print(f"Unable to read frame {i}, exiting.")
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        video[i] = np.array(gray, dtype = np.int32)
        i += 1
    vp.release()
    return video

def optical_flow(video):
    """
    Computes OpenCV Farneback optical flow on a NumPy grayscale video. NOTE: uses
    a lot of hard-coded parameters!

    Parameters
    ----------
    video : array, shape (F, H, W)
        The grayscale video.

    Returns
    -------
    flow : array, shape (F - 1, H, W, 2)
        The optical flow.
    """
    n_frames = video.shape[0]
    prev_flow = None
    flow_vid = []
    for i in range(1, n_frames):
        curr = video[i]
        prev = video[i - 1]
    
        # Calculate the optical flow for a pair of frames.
        opt = cv2.calcOpticalFlowFarneback(
            prev,
            curr,
            prev_flow,
            0.85,   # pyr_scale
            7,      # levels
            15,     # winsize
            20,     # iterations
            7,      # poly_n
            0.7,    # poly_sigma
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN & cv2.OPTFLOW_USE_INITIAL_FLOW
        )
        # Convert to float32 format.
        flow = np.array(opt, dtype = np.float32)
        flow_vid.append(flow)
        prev_flow = flow
    return np.array(flow_vid)
