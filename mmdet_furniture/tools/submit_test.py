import mmcv

root='/mnt/truenas/scratch/lqf/code/mmdet_lqf/whole_pkl/test/'
filepath='pointrend_res2net101_dcn_fpnbfp_large_coarsehead_fp16_larger_img_p2p6_1200_1400_enrichfeat_largeboxalign/'
filename='120013001400test.pkl'
pred_data = mmcv.load(root+filepath+filename)
data_path = '/mnt/truenas/scratch/lqf/code/czh/data/future/'
val_anno = data_path + 'annotations/test_set.json'  # NOTE
val_anno = mmcv.load(val_anno)

anno_list = []
for anno, pred in zip(val_anno['images'], pred_data):
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
outname='segmentation_resutls_{}'.format(filename.replace('.pkl','.json'))
mmcv.dump(submit, root+filepath+outname)
