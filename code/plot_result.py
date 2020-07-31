import mmcv
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mutils
from pycocotools.coco import COCO
import cv2
import argparse
import os

color_list = [[0, 153, 51], [0, 153, 255], [255, 255, 0], [204, 102, 255],\
              [255, 0, 0], [204, 0, 255], [102, 255, 255], [255, 102, 102]]
ratio = 1

anno_path = '/mnt/truenas/scratch/czh/data/future/annotations/val_set.json'
img_path = '/mnt/truenas/scratch/czh/data/future/images/val/'
num_class = 34

def parse_args():
    parser = argparse.ArgumentParser(description='Plot results on 3D Future')
    parser.add_argument('--dir', help='path to pkl, (/xxx/xxx/xxx.pkl)', type=str)
    parser.add_argument('--num-images', help='number of images to plot', default=1, type=int)
    parser.add_argument('--score-thr', help='score to visualize results', default=0.5, type=float)
    args = parser.parse_args()
    return args.dir, args.num_images, args.score_thr

def decode_mask(mask_encode):
    mask_decode = mutils.decode(mask_encode)
    return mask_decode

def apply_mask(img, mask, color, alpha=.5):
    for c in range(3):
        img[:, :, c] = np.where(mask==1, img[:, :, c] *
                                (1 - alpha) + alpha * color[c],
                                img[:, :, c])
    return img

def draw_mask(img, mask_data, index):
    global color_list
    index %= 5
    mask_mat = decode_mask(mask_data)
    img = apply_mask(img, mask_mat, color_list[index], .5)
    return img

def draw_bbox(img, bbox_score_data, index, show_score=False):
    global color_list
    index %= 5
    test = list(map(lambda x: int(x), bbox_score_data[:4]))
    img = cv2.rectangle(img, (test[0], test[1]), (test[2], test[3]), color_list[index], 2)
    if show_score:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = ""
        if show_score:
            text += str(bbox_score_data[4])[:4]
        rec_width = len(text) * 6
        img = cv2.rectangle(img, (test[0], test[1]), (test[0]+rec_width, test[1]+10), color_list[index], -1)
        cv2.putText(img, text, (test[0], test[1]+8), font, 0.35, (0, 0, 0), 1)
    return img

def parse_img(img, anno, vis_thr):
    is_solo = True if len(anno) == num_class else False
    if is_solo:
        bbox_data = [[] for ii in range(num_class)]; mask_data = anno    # SOLO
    else:
        bbox_data = anno[0]; mask_data = anno[1]        # HTC & PointRend
    if len(mask_data) == 2:                             # HTC
        mask_data = mask_data[0]
    for cid in range(num_class):
        num_inst = len(mask_data[cid])
        for ii in range(num_inst):
            mask = mask_data[cid][ii]
            if is_solo and mask[1] > vis_thr:           # SOLO
                img = draw_mask(img, mask[0], cid)
            elif not is_solo and bbox_data[cid][ii][4] > vis_thr:   # HTC & Pointrend
                img = draw_mask(img, mask, cid)
            else:
                pass
    return img

if __name__ == '__main__':
    pred_path, num_img, vis_thr = parse_args()
    model_name = pred_path.split('/')[-1][:-4]
    print("Load [%s] Model Pickle..." % model_name)

    pred_data = mmcv.load(pred_path)
    anno_data = mmcv.load(anno_path)
    coco = COCO(anno_path)

    os.makedirs('results/plot/%s' % model_name, exist_ok=True)

    cnt = 0
    for index, (anno, pred) in enumerate(zip(anno_data['images'], pred_data)):
        if index < 500:
            continue
        cnt += 1
        img_name = anno['file_name'] + '.jpg'
        img = cv2.imread(img_path + img_name)[:, :, ::-1]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = parse_img(img, pred, vis_thr)
        save_path = 'results/plot/%s/%s' % (model_name, img_name)
        plt.imsave(save_path, img)
        if cnt >= num_img:
            break
    print("Saving Results Done!")