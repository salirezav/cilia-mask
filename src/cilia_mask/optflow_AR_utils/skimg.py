import numpy as np
import skimage.registration as skreg

def optical_flow(img1, img2):
    flow1 = skreg.optical_flow_ilk(img1, img2)
    flow2 = skreg.optical_flow_tvl1(img1, img2)