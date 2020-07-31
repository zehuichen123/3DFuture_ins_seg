import mmcv
import os.path as osp
import argparse


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, help='absolute path for project mmdet_furniture')
    parser.add_argument('--config_name', type=str, help='config file name for PointRend model')
    return parser

def pkl2json(args):
    root, config_name = args.root, args.config_name
    pklpath = osp.join(root, 'out_results', config_name, 'segmentation_resutls.pkl')
    pred_data = mmcv.load(pklpath)
    anno_path = osp.join(root, 'data/future/annotations', 'test_set.json')
    test_anno = mmcv.load(anno_path)

    anno_list = []
    for anno, pred in zip(test_anno['images'], pred_data):
        bbox_data = pred[0]; mask_data = pred[1]
        for cid in range(34):
            bboxes = bbox_data[cid]
            masks = mask_data[cid]
            for ii in range(bboxes.shape[0]):
                x1, y1, x2, y2, score = bboxes[ii]
                x = (x1 + x2) / 2; y = (y1 + y2) / 2
                w = x2 - x1 + 1; h = y2 - y1 + 1
                mask = masks[ii]
                res = {
                    'image_id': anno['id'],
                    'category_id': cid + 1,
                    'segmentation': {'size': mask['size'], 'counts': mask['counts'].decode('utf-8')},
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'score': float(score),
                }
                anno_list.append(res)

    img_list = []
    for img_info in test_anno['images']:
        res = {
            'image_id': img_info['id'],
            'width': img_info['width'],
            'height': img_info['height'],
            'file_name': img_info['file_name']
        }
        img_list.append(res)

    submit = {}
    submit['images'] = img_list
    submit['annotations'] = anno_list
    save_path = osp.join(root, 'out_results', config_name, 'segmentation_resutls.json')
    mmcv.dump(submit, save_path)


if __name__ == '__main__':
    root = '/mnt/truenas/scratch/lqf/code/github/3DFuture_ins_seg/mmdet_furniture/'  # absolute path for mmdet_furniture
    config_name = 'pointrend_x101_64x4d_dcn_fpn_fp16_p2p6'
    parser = arg_parse()
    args = parser.parse_args()
    pkl2json(args)
