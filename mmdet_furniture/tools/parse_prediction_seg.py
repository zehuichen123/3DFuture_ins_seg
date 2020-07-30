import mmcv
data_path = '/mnt/truenas/scratch/czh/data/future/'
pred_path = '/mnt/truenas/scratch/czh/mmdet/work_dirs/solo_x101_dcn_4x_mst'

pred_data = mmcv.load(pred_path + '/results.pkl')
val_anno = mmcv.load(data_path + '/annotations/val_set.json')

num_class = 34
segm_only = True

anno_list = []
for anno, pred in zip(val_anno['images'], pred_data):
    if segm_only:
        mask_data = pred; bbox_data = [[] for ii in range(num_class)]
    else:
        mask_data = pred[1]; bbox_data = pred[0]
    for cid in range(num_class):
        bboxs = bbox_data[cid]
        masks = mask_data[cid]
        for ii in range(len(masks)):
            if segm_only:
                mask = masks[ii][0]; score = masks[ii][1]
            else:
                x1, y1, x2, y2, score = bboxes[ii]
                # x = (x1 + x2) / 2; y = (y1 + y2) / 2
                # w = x2 - x1 + 1; h = y2 - y1 + 1
                mask = masks[ii]
            res = {
                'image_id': anno['id'],
                'category_id': cid + 1,
                'segmentation': {'size': mask['size'], 'counts': mask['counts'].decode('utf-8')},
                'score': float(score),
            }
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

mmcv.dump(submit, 'segmentation_results.json')