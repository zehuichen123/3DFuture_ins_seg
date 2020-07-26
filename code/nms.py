import numpy as np
from cython_nms.cpu_nms import greedy_nms, soft_nms
import pycocotools.mask as mutils

def cython_soft_nms_wrapper(thresh, sigma=0.5, score_thresh=0.001, method='linear'):
    methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    assert method in methods, 'Unknown soft_nms method: {}'.format(method)
    def _nms(dets):
        dets, inds = soft_nms(
                    np.ascontiguousarray(dets, dtype=np.float32),
                    np.float32(sigma),
                    np.float32(thresh),
                    np.float32(score_thresh),
                    np.uint8(methods[method]))
        return dets, inds
    return _nms

def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms

def py_matrix_nms_wrapper():
    def _nms(dets):
        return matrix_nms(dets)
    return _nms

def cpu_nms_wrapper(thresh):
    def _nms(dets):
        return greedy_nms(dets, thresh)[0]
    return _nms

def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep], keep

def matrix_nms(dets, kernel='gaussian', sigma=0.5):
    N = len(dets)
    if N == 0:
        return dets, np.where(dets[:, 0] == 0)[0]
    dets = np.array(dets)
    rles = dets[:, 0]
    scores = dets[:, 1]
    order = scores.argsort()[::-1]
    # sort in decending order
    sorted_scores = scores[order]
    sorted_rles = rles[order]
    ious = mutils.iou(rles.tolist(), rles.tolist(), [False])
    ious = np.triu(ious, k=1)
    
    ious_cmax = ious.max(0)
    ious_cmax = np.tile(ious_cmax, reps=(N, 1)).T
    if kernel == 'gaussian':
        decay = np.exp(-(ious ** 2 - ious_cmax ** 2) / sigma)
    else: # linear
        decay = (1 - ious) / (1 - ious_cmax)
    # decay factor: N
    decay = decay.min(axis=0)
    sorted_scores *= decay
    dets = np.concatenate([sorted_rles.reshape(-1, 1), sorted_scores.reshape(-1, 1)], axis=-1)
    valid_ind = np.where(sorted_scores >= 0.05)[0]
    return dets[valid_ind], order[valid_ind]