from .optflow_AR_utils import ar, invariants, raster_scan
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


class OF_AR:

    def draw_flow(self, img, flow, step=6):
        h, w = img.shape[:2]
        y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (_x2, _y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        return vis

    def calc_OF(self, vid: np.array):
        flow_vid = []
        self.rot = []

        prev_flow = None
        for i in tqdm(range(1, vid.shape[0])):
            curr = vid[i]
            prev = vid[i - 1]

            opt = cv2.calcOpticalFlowFarneback(prev, curr, prev_flow, 0.75, 7, 15, 20, 7, 0.9, cv2.OPTFLOW_FARNEBACK_GAUSSIAN & cv2.OPTFLOW_USE_INITIAL_FLOW)  # pyr_scale  # levels  # winsize  # iterations  # poly_n  # poly_sigma
            opt = np.array(opt, dtype=np.float64)
            self.rot.append(invariants.curl(opt[:, :, 0], opt[:, :, 1]))
            flow_vid.append(self.draw_flow(curr, opt))
            prev_flow = opt

        flow_vid = np.array(flow_vid)
        self.rot = np.array(self.rot)

        bwrot = self.rot.copy()
        bwrot += np.abs(bwrot.min())
        bwrot /= bwrot.max()
        bwrot *= 256
        bwrot = np.array(bwrot, np.uint8)

        return flow_vid, self.rot, bwrot

    def AR(self, rot: np.array, order: int = 5):
        # order = 5

        image = np.zeros(shape=(order, rot.shape[1], rot.shape[2]))
        for row in tqdm(range(image.shape[1])):
            for col in range(image.shape[2]):
                a = ar.train(rot[:, row, col], order)
                for i in range(image.shape[0]):
                    image[i, row, col] = a[i][0][0]
        return image

    # def do_raster(rot):
    #     X = raster_scan(rot).T
    #     X.shape


def calculate_iou(layer1: np.ndarray, layer2: np.ndarray) -> float:
    """Calculate Intersection over Union (IoU) between two layers."""
    intersection = np.logical_and(layer1, layer2).sum()
    union = np.logical_or(layer1, layer2).sum()
    iou = intersection / union if union != 0 else 0
    return iou
