# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn
from ..registry import HEADS
from ..builder import build_head
import torch.nn.functional as F
import mmcv
import pycocotools.mask as mask_util
from mmdet.core import auto_fp16, force_fp32


from .point_features import (
    generate_regular_grid_point_coords,
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
    point_sample_fine_grained_features,
)


def calculate_uncertainty(logits, classes):
    if logits.shape[1] == 1:
        gt_class_logits = logits.clone()
        raise ValueError
    else:
        gt_class_logits = logits[
            torch.arange(logits.shape[0], device=logits.device), classes
        ].unsqueeze(1)
    return -(torch.abs(gt_class_logits))


@HEADS.register_module
class PointRendROIHeads(nn.Module):
    def __init__(self, coarse_mask_head, point_head):
        super(PointRendROIHeads, self).__init__()
        self._init_mask_head(coarse_mask_head)
        self._init_point_head(point_head)
        assert point_head.num_classes==coarse_mask_head.num_classes
        self.num_classes=point_head.num_classes
        self.init_weights()
        self.fp16_enabled = False

    def _init_mask_head(self, coarse_mask_head_cfg):
        self.mask_on = True
        strides, mask_coarse_in_features, mask_coarse_in_channels = coarse_mask_head_cfg.strides, coarse_mask_head_cfg.in_features, coarse_mask_head_cfg.in_channels
        assert len(strides)==len(mask_coarse_in_features)==len(mask_coarse_in_channels)
        self.mask_coarse_in_features = mask_coarse_in_features
        self.mask_coarse_side_size = coarse_mask_head_cfg.mask_coarse_side_size
        self._mask_coarse_feature_scales = [1./stride for stride in strides]

        in_channels_ = sum(mask_coarse_in_channels)
        coarse_mask_head_cfg.update({'input_channels':in_channels_, 'input_h':self.mask_coarse_side_size, 'input_w':self.mask_coarse_side_size})
        self.mask_coarse_head = build_head(coarse_mask_head_cfg)
        self.output_side_resolution=coarse_mask_head_cfg.output_side_resolution

    def _init_point_head(self, point_head_cfg):
        self.mask_point_on = True
        self.mask_point_in_features = point_head_cfg.mask_point_in_features
        self._mask_point_feature_scales = [1./stride for stride in point_head_cfg.strides]
        self.mask_point_train_num_points        = point_head_cfg.train_num_points
        self.mask_point_oversample_ratio        = point_head_cfg.oversample_ratio
        self.mask_point_importance_sample_ratio = point_head_cfg.importance_sample_ratio
        self.mask_point_subdivision_steps       = point_head_cfg.subdivision_steps
        self.mask_point_subdivision_num_points  = point_head_cfg.subdivision_num_points

        in_channels = sum(point_head_cfg.mask_point_in_channels)
        point_head_cfg.update({'input_channels': in_channels})
        self.mask_point_head = build_head(point_head_cfg)
    
    def init_weights(self,):
        self.mask_coarse_head.init_weights()
        self.mask_point_head.init_weights()
    
    @auto_fp16()
    def _forward_mask_train(self, features, sampling_results, gt_masks, return_coarse_features=False, return_indices=False):
        if return_coarse_features:
            coarse_feats, coarse_pred = self._forward_mask_coarse(features, [res.pos_bboxes for res in sampling_results], return_coarse_features=True)  
        else:
            coarse_pred = self._forward_mask_coarse(features, [res.pos_bboxes for res in sampling_results])
        mask_targets = self.mask_coarse_head.get_target(sampling_results, gt_masks)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_coarse_head.loss(coarse_pred, mask_targets, pos_labels)
        torch.cuda.empty_cache()
        if return_indices:
            loss_mask_point, point_indices_,point_logits_ = self._forward_mask_point_train(
                features, coarse_pred, sampling_results, gt_masks, return_indices=True)
            fine_grained_mask_pred = coarse_pred
            fine_grained_mask_pred = F.interpolate(fine_grained_mask_pred, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            loss_mask_point = self._forward_mask_point_train(features, coarse_pred, sampling_results, gt_masks)
        loss_mask.update(loss_mask_point)
        if return_coarse_features:
            mask_targets = self.mask_coarse_head.get_target(sampling_results, gt_masks, mask_size=self.output_side_resolution*2)
            return loss_mask, (coarse_feats,fine_grained_mask_pred, mask_targets)
        else:
            return loss_mask

    @auto_fp16()
    def _forward_mask_coarse(self, features, boxes, return_coarse_features=False):
        point_coords = generate_regular_grid_point_coords(
            np.sum(len(x) for x in boxes) if isinstance(boxes, list) else boxes.size(0), 
            self.mask_coarse_side_size, features[0].device, features[0].dtype)
        mask_coarse_features_list = [features[k] for k in self.mask_coarse_in_features]
        mask_features, _ = point_sample_fine_grained_features(
            mask_coarse_features_list, self._mask_coarse_feature_scales, boxes, point_coords
        )
        if return_coarse_features:
            return mask_features.view(mask_features.size(0),mask_features.size(1),self.mask_coarse_side_size, 
                self.mask_coarse_side_size), self.mask_coarse_head(mask_features)
        else:
            return self.mask_coarse_head(mask_features)

    @auto_fp16()
    def _forward_mask_point_train(self, features, mask_coarse_logits, sampling_results, gt_mask_list, return_indices=False):
        mask_features_list = [features[k] for k in self.mask_point_in_features]
        proposal_boxes = [res.pos_bboxes for res in sampling_results]
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        assert pos_labels.size(0) == sum([proposal_.size(0) for proposal_ in proposal_boxes])
        with torch.no_grad():
            point_coords_outs = get_uncertain_point_coords_with_randomness(
                mask_coarse_logits,
                lambda logits: calculate_uncertainty(logits, pos_labels), 
                self.mask_point_train_num_points,
                self.mask_point_oversample_ratio,
                self.mask_point_importance_sample_ratio,
                return_indices=return_indices,
            )
        point_coords=point_coords_outs[0]
        fine_grained_features, point_coords_wrt_image = point_sample_fine_grained_features(
            mask_features_list, self._mask_point_feature_scales, proposal_boxes, point_coords)
        coarse_features = point_sample(mask_coarse_logits, point_coords, align_corners=False)
        point_logits = self.mask_point_head(fine_grained_features, coarse_features)
        loss_mask_point={"loss_mask_point": self.mask_point_head.roi_mask_point_loss(
                point_logits, point_coords_wrt_image, sampling_results, gt_mask_list)} 
        if not return_indices:
            return loss_mask_point
        else:
            point_indices = point_coords_outs[1]
            return loss_mask_point, point_indices,point_logits

    def _forward_mask_test(self, features, det_bboxes, det_labels, return_coarse_features=False):
        if return_coarse_features:
            coarse_feats, mask_coarse_logits = self._forward_mask_coarse(features, det_bboxes, return_coarse_features=True)
        else:
            mask_coarse_logits = self._forward_mask_coarse(features, det_bboxes,)
        mask_logits = self._forward_mask_point_test(features, mask_coarse_logits, det_bboxes, det_labels)
        if return_coarse_features:
            mask_logits_for_maskiou = F.interpolate(mask_logits,  size=coarse_feats.size()[-2:], mode='bilinear', align_corners=False)
            return mask_logits, (coarse_feats, mask_logits_for_maskiou)
        else:
            return mask_logits

    def _forward_mask_point_test(self, features, mask_coarse_logits, det_bboxes, det_labels):
        mask_features_list = [features[k] for k in self.mask_point_in_features]
        features_scales = [self._mask_point_feature_scales[k] for k in self.mask_point_in_features]

        mask_logits = mask_coarse_logits.clone()
        for subdivions_step in range(self.mask_point_subdivision_steps):
            mask_logits = F.interpolate(mask_logits, scale_factor=2, mode="bilinear", align_corners=False)
            H, W = mask_logits.shape[-2:]
            if (self.mask_point_subdivision_num_points >= 4 * H * W
                and subdivions_step < self.mask_point_subdivision_steps - 1):
                continue
            uncertainty_map = calculate_uncertainty(mask_logits, det_labels+1)
            point_indices, point_coords = get_uncertain_point_coords_on_grid(
                uncertainty_map, self.mask_point_subdivision_num_points)
            fine_grained_features, _ = point_sample_fine_grained_features(
                mask_features_list, features_scales, det_bboxes, point_coords)
            coarse_features = point_sample(
                mask_coarse_logits, point_coords, align_corners=False)
            point_logits = self.mask_point_head(fine_grained_features, coarse_features)

            R, C, H, W = mask_logits.shape
            point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
            mask_logits = (
                mask_logits.reshape(R, C, H * W)
                .scatter_(2, point_indices, point_logits)
                .view(R, C, H, W)
            )
        return mask_logits   

    def scatter_maskpred(self, coarse_mask_pred, point_indices, point_logits):
        R,C,H,W=coarse_mask_pred.shape
        point_indices = point_indices.unsqueeze(1).expand(-1,C,-1)
        point_indices = point_indices.to(torch.long)
        fine_grained_mask_pred = coarse_mask_pred.reshape(R,C,H*W).scatter(2,point_indices,point_logits).view(R,C,H,W)
        return fine_grained_mask_pred

    @force_fp32(apply_to=('mask_pred'))
    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        mask_pred = mask_pred.astype(np.float32)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            if not isinstance(scale_factor, (float, np.ndarray)):
                scale_factor = scale_factor.cpu().numpy()
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)
            mask_pred_ = mask_pred[i, label, :, :]
            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(np.uint8)

            if rcnn_test_cfg.get('crop_mask', False):
                im_mask = bbox_mask
            else:
                im_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask

            if rcnn_test_cfg.get('rle_mask_encode', True):
                rle = mask_util.encode(
                    np.array(im_mask[:, :, np.newaxis], order='F'))[0]
                cls_segms[label - 1].append(rle)
            else:
                cls_segms[label - 1].append(im_mask)
        return cls_segms