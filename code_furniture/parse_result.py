import numpy as np
def parse_pred_2_json(pred_data, val_anno, num_class=34, add_bbox=False):
    anno_list = []
    for anno, pred in zip(val_anno['images'], pred_data):
        for cid in range(num_class):
            keep_cid = np.where(pred[:, -1] == cid)[0]
            pred_res = pred[keep_cid]
            for ii in range(pred_res.shape[0]):
                x1, y1, x2, y2, score, mask, ms_score, cls = pred_res[ii]
                # x = (x1 + x2) / 2; y = (y1 + y2) / 2
                w = x2 - x1 + 1; h = y2 - y1 + 1
                final_score = ms_score * score
                res = {
                    'image_id': anno['id'],       # special for cocoeval
                    'category_id': cid + 1,
                    'segmentation': {'size': mask['size'], 'counts': mask['counts'].decode('utf-8')},
                    'score': float('%.3f' % final_score),
                }
                if add_bbox:
                    res['bbox'] = [float(ii) for ii in [x1, y1, w, h]]
                anno_list.append(res)

    img_list = []
    for img_info in val_anno['images']:
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
    return submit

# mmcv.dump(submit, '../segmentation_results.json')