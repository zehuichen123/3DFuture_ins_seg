# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from .point_features import point_sample
from mmdet.core import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair
from ..registry import HEADS
import numpy as np


@HEADS.register_module
class StandardPointHead(nn.Module):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self, input_channels, num_classes=35, fc_dim=256, num_fc=3, cls_agnostic_mask=False, coarse_pred_each_layer=True, **kwargs):
        super(StandardPointHead, self).__init__()
        self.fp16_enabled = False
        self.coarse_pred_each_layer = coarse_pred_each_layer
        if cls_agnostic_mask:
            raise NotImplementedError
        fc_dim_in = input_channels + num_classes
        self.fc_layers = []
        for k in range(num_fc):
            fc = nn.Conv1d(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias=True)
            self.add_module("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv1d(fc_dim_in, num_mask_classes, kernel_size=1, stride=1, padding=0)
        self.init_weights()

    def init_weights(self,):
        for layer in self.fc_layers:
            weight_init.c2_msra_fill(layer)
        nn.init.normal_(self.predictor.weight, std=0.001)
        if self.predictor.bias is not None:
            nn.init.constant_(self.predictor.bias, 0)

    @auto_fp16()
    def forward(self, fine_grained_features, coarse_features):
        x = torch.cat((fine_grained_features, coarse_features), dim=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_features), dim=1)
        return self.predictor(x)

    @force_fp32(apply_to=('mask_logits', ))
    def roi_mask_point_loss(self, mask_logits, points_coord, sampling_results, gt_mask_list):  # instances
        with torch.no_grad():
            total_num_masks = mask_logits.size(0)

            gt_mask_logits = []
            assigned_gt_masks_list = []
            idx = 0
            gt_classes = [res.pos_gt_labels for res in sampling_results]

            pos_assigned_gt_inds_list = [res.pos_assigned_gt_inds for res in sampling_results]
            for pos_assigned_gt_inds, gt_mask_perimg in zip(pos_assigned_gt_inds_list, gt_mask_list):
                num_pos = pos_assigned_gt_inds.size(0)
                if num_pos > 0 and len(gt_mask_perimg)>0:
                    pos_assigned_gt_inds_np = pos_assigned_gt_inds.cpu().numpy()
                    assigned_gt_masks_list.append(torch.from_numpy(np.stack([gt_mask_perimg[pos_assigned_gt_inds_np[i]] 
                        for i in range(num_pos)])).to(mask_logits.dtype).to(mask_logits.device))
                else:
                    continue

            for assigned_gt_masks in assigned_gt_masks_list:
                if len(assigned_gt_masks) == 0:
                    continue
                h, w = assigned_gt_masks.size(-2), assigned_gt_masks.size(-1)
                scale = torch.tensor([w, h], dtype=torch.float, device=assigned_gt_masks.device)
                points_coord_grid_sample_format = (
                    points_coord[idx : idx + len(assigned_gt_masks)] / scale
                )

                idx += len(assigned_gt_masks)
                gt_mask_logits.append(
                    point_sample(
                        assigned_gt_masks.to(torch.float32).unsqueeze(1) if len(assigned_gt_masks.size())==3 else assigned_gt_masks.to(torch.float32),
                        points_coord_grid_sample_format,
                        align_corners=False,
                    ).squeeze(1)
                )

        if len(gt_mask_logits) == 0:
            return mask_logits.sum() * 0

        gt_mask_logits = torch.cat(gt_mask_logits, 0)
        assert gt_mask_logits.numel() > 0, gt_mask_logits.size()
        
        indices = torch.arange(total_num_masks)
        gt_classes = torch.cat(gt_classes, dim=0)
        assert gt_mask_logits.size(0)==total_num_masks == gt_classes.size(0)
        mask_logits = mask_logits[indices, gt_classes]

        point_loss = F.binary_cross_entropy_with_logits(
            mask_logits, gt_mask_logits.to(dtype=mask_logits.dtype), reduction="mean"
        )
        return point_loss
