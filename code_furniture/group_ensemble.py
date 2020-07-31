from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mutils

import concurrent.futures
import time
import mmcv
import numpy as np

data_dir = 'infer_results/'
anno_dir = 'data/future/annotations/test_set.json'
ensemble_list = [
    'pointrend_res2net101_dcn_fpnbfp_fp16_p2p6_enrichfeat.pkl',    # 77.21
    'pointrend_x101_64x4d_dcn_fpnbfp_fp16_p2p6_lr001.pkl',  # 77.32
    'pointrend_x101_64x4d_dcn_fpn_fp16_p2p6.pkl',   # 77.38
    'pointrend_x101_64x4d_dcn_fpnbfp_fp16_p2p6_enrichfeat.pkl',   # 77.37
    'pointrend_res2net101_dcn_fpnbfp2repeat_p2p6_fp16_enrichfeat_largeboxalign.pkl'    # 76.95
]
val_score_list = [77.21, 77.32, 77.38, 77.37, 76.95]
num_ensemble = len(ensemble_list)
model_type_list = []
nms_type = 'soft_nms'

def generate_reweight_list(val_score_list, min_val=0.6, max_val=1.0):
    min_score = min(val_score_list)
    max_score = max(val_score_list)
    interval = max_score - min_score
    reweight_list = []
    for each_score in val_score_list:
        reweight_score = min_val + (each_score - min_score) * (max_val - min_val) / interval
        reweight_list.append(reweight_score)
    return reweight_list

def check_valid_ensemble_list(model_type_list):
    for each_model_type in model_type_list:
        if each_model_type == "SOLO":
            if set(model_type_list) != set("SOLO"):
                print("Only Support All SOLO ensembles")
                raise Exception
    return

def generate_model_type_list(ensemble_list):
    model_type_list = []
    for each_dir in ensemble_list:
        if 'htc' in each_dir:
            model_type_list.append("HTC")
            continue
        elif 'pointrend' in each_dir:
            model_type_list.append("PointRend")
            continue
        elif 'solo' in each_dir:
            model_type_list.append('SOLO')
            continue
        else:
            print("Can NOT determine model type: %s" % each_dir)
            raise Exception
    check_valid_ensemble_list(model_type_list)
    return model_type_list

def group_results(res_list, model_type_list, score_list):
    num_images = len(res_list[0])
    num_classes = 34
    group_res = []
    for img_id in range(num_images):
        group_res_per_img = []
        for model_id in range(num_ensemble):
            res_per_img = res_list[model_id][img_id]
            model_type = model_type_list[model_id]
            model_score = score_list[model_id]
            # determine the type of model
            for cid in range(num_classes):
                if model_type == 'SOLO':
                    mask_data = res_per_img
                    for det_id in range(len(mask_data[cid])):
                        each_det = [cid] * 8  # x1,y1,x2,y2,score,RLE,mask_score,cid
                        each_det[4] = mask_data[cid][det_id][1]
                        each_det[5] = mask_data[cid][det_id][0]
                        each_det[6] = 1
                        group_res_per_img.append(each_det)
                elif model_type == 'HTC':
                    bbox_data, mask_info = res_per_img
                    mask_data, mask_scores = mask_info
                    det_bbox = bbox_data[cid]; det_mask = mask_data[cid]; mask_score = mask_scores[cid]
                    for det_id in range(det_bbox.shape[0]):
                        each_det = [cid] * 8
                        each_det[:5] = det_bbox[det_id]
                        each_det[4] *= model_score
                        each_det[5] = det_mask[det_id]
                        each_det[6] = mask_score[det_id]
                        group_res_per_img.append(each_det)
                elif model_type == 'PointRend':
                    bbox_data, mask_info = res_per_img
                    det_bbox = bbox_data[cid]; det_mask = mask_info[cid]
                    for det_id in range(det_bbox.shape[0]):
                        each_det = [cid] * 8
                        each_det[:5] = det_bbox[det_id]
                        each_det[4] *= model_score
                        each_det[5] = det_mask[det_id]
                        each_det[6] = 1
                        group_res_per_img.append(each_det)
                else:
                    print("No Model Type named", model_type)
                    raise Exception
        group_res_per_img = np.array(group_res_per_img)
        order = group_res_per_img[:, 4].argsort()[::-1]
        if nms_type == 'matrix_nms':
            group_res_per_img = group_res_per_img[order[:500 * num_ensemble]]
        group_res.append(group_res_per_img)
    return group_res

def nms_per_img(group_res_per_img):
    num_classes = 34
    group_res = []
    for cid in range(num_classes):
        cid_ind = np.where(group_res_per_img[:, -1] == cid)[0]
        if nms_type != 'matrix_nms':
            if nms_type == 'soft_nms':
                from nms import cython_soft_nms_wrapper
                nms = cython_soft_nms_wrapper(0.5)
            elif nms_type == 'nms':
                from nms import py_nms_wrapper
                nms = py_nms_wrapper(0.5)
            dets = group_res_per_img[cid_ind][:, :5]
        elif nms_type == 'matrix_nms':
            from nms import py_matrix_nms_wrapper
            nms = py_matrix_nms_wrapper()
            dets = group_res_per_img[cid_ind][:, 4:6][:, ::-1]
            dets[:, 1] *= group_res_per_img[cid_ind][:, 6]  # multiply with mask score
            group_res_per_img[cid_ind][:, 6] = 1
        dets, keep = nms(dets)
        keep_res = group_res_per_img[cid_ind][keep]
        keep_res[:, 4] = dets[:, -1]  # for softNMS update score
        group_res.append(keep_res)
    group_res = np.vstack(group_res)
    order = group_res[:, 4].argsort()[::-1]
    return group_res[order[:100]]
            
def ensemble_group_det(group_res, num_threads=45):
    new_dets = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        for index, res_data in enumerate(executor.map(nms_per_img, group_res)):
            new_dets.append(res_data)
    return new_dets

if __name__ == '__main__':
    # Loading results and annotations
    checkpoint = time.time()

    model_type_list = generate_model_type_list(ensemble_list)
    ensemble_list = [data_dir + each_dir for each_dir in ensemble_list]

    coco = COCO(anno_dir)
    val_anno = mmcv.load(anno_dir)

    ensemble_res_list = [mmcv.load(model_dir) for model_dir in ensemble_list]
    num_ensemble = len(ensemble_list)

    print("[*] Loading Result in %g s" % (time.time() - checkpoint))
    checkpoint = time.time()

    # determine score list
    score_list = generate_reweight_list(val_score_list)
    print("[*] Original Score List is ", val_score_list)
    print("[*] Score List is ", score_list)

    # Group results with multiple models
    group_res = group_results(ensemble_res_list, model_type_list, score_list)

    print("[*] Group Result in %g s" % (time.time() - checkpoint))
    checkpoint = time.time()

    # NMS among group results
    new_dets = ensemble_group_det(group_res)
    print("[*] NMS Result in %g s" % (time.time() - checkpoint))
    checkpoint = time.time()

    from parse_result import parse_pred_2_json
    tmp_res = parse_pred_2_json(new_dets, val_anno)

    mmcv.dump(tmp_res, 'segmentation_resutls.json')
    print("[*] Saving Result in %g s" % (time.time() - checkpoint))



        

                

