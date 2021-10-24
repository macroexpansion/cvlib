
import numpy as np
import cv2
import time
from os.path import splitext

from alpr_unconstrained.src.label import Label
from alpr_unconstrained.src.utils import getWH, nms
from alpr_unconstrained.src.projection_utils import getRectPts, find_T_matrix
from alpr_unconstrained.src.utils import im2single


class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, 1)
        br = np.amax(pts, 1)
        Label.__init__(self, cl, tl, br, prob)

    def add_pad(self, pad_percent=0.1):
        w, h = self.wh()
        pad_w = w*pad_percent/2
        pad_h = 0
        pad_matrix = [[-pad_w, pad_w, pad_w, -pad_w], [-pad_h, -pad_h, pad_h, pad_h]]
        self.pts += pad_matrix

def decode_predict(Y, I_resized_shape, threshold=.9):
    try:
        net_stride = 2**4
        side = ((288. + 40.)/2.)/net_stride  # 7.75

        Probs = Y[..., 0]
        Affines = Y[..., 2:]
        rx, ry = Y.shape[:2]

#     print("prob maximum:",np.max(Probs))
#     xx, yy = np.where(Probs > threshold)
        max_prob = np.amax(Probs) 
    
        xx,yy = np.where((Probs > threshold) & (Probs == max_prob))

        WH = getWH(I_resized_shape)
        MN = WH/net_stride

        vxx = vyy = 0.5  # alpha

        base = lambda vx, vy: np.matrix(
            [[-vx, -vy, 1.], [vx, -vy, 1.], [vx, vy, 1.], [-vx, vy, 1.]]).T
        labels = []

        for i in range(len(xx)):
            y, x = xx[i], yy[i]
            affine = Affines[y, x]
            prob = Probs[y, x]

            mn = np.array([float(x) + .5, float(y) + .5])

            A = np.reshape(affine, (2, 3))
            A[0, 0] = max(A[0, 0], 0.)
            A[1, 1] = max(A[1, 1], 0.)

            pts = np.array(A*base(vxx, vyy))  # *alpha
            pts_MN_center_mn = pts*side
            pts_MN = pts_MN_center_mn + mn.reshape((2, 1))

            pts_prop = pts_MN/MN.reshape((2, 1))

            labels.append(DLabel(0, pts_prop, prob))

#     print("number of prediction: ",len(labels))
        final_labels = nms(labels, .1)
#     print("the number after nms:",len(final_labels))
    
        return final_labels
    except:
        return  []

def point_distance(vx1,vy1,vx2,vy2):
    distance = np.sqrt(np.square(vx1-vx2)+np.square(vy1-vy2))
    return distance
    
def reconstruct(Iorig, final_labels):
    h, w, _ = Iorig.shape 
    final_labels.sort(key=lambda x: x.prob(), reverse=True)
    # One plate per vehicle
    label = final_labels[0]
    out_size = (np.int(label.wh()[0]*w),np.int(label.wh()[1]*h)) 
    t_ptsh 	= getRectPts(0, 0, out_size[0], out_size[1])

    ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)), np.ones((1,4))))
    H 		= find_T_matrix(ptsh, t_ptsh)
    Ilp 	= cv2.warpPerspective(Iorig, H, out_size, borderValue=.0,flags=cv2.INTER_CUBIC)
    return Ilp, label


def detect_lp_on_batch(sess, image_resized_numpy, threshold, input_name, output_name):
    start 	= time.time()
    batch_size = 256
    n_img = image_resized_numpy.shape[0]
    n_left = n_img
    Yr = []
    while n_left > 0:
        n_samples = min(batch_size, n_left)
        start_index = n_img-n_left
        end_index = start_index + n_samples
        Yr_batch = sess.run([output_name], {input_name: image_resized_numpy[start_index:end_index]})[0]
        for i in range(n_samples):
            Yr.append(Yr_batch[i])
        n_left -= n_samples
    
    elapsed = time.time() - start
    #print("Inference time of detect plate: ", elapsed)
    return np.array(Yr)