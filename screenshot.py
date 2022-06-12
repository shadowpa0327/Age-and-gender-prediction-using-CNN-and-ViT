from mss import mss
import numpy as np

sct = mss()
def screenshot(left, top, width, height):
    mon = {'left': left, 'top': top, 'width': width, 'height': height}
    img = np.asarray(sct.grab(mon))

    return img