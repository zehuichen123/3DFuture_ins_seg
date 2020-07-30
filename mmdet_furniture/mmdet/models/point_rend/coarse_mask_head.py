# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from mmdet.core import auto_fp16, force_fp32
from ..builder import build_loss
from ..registry import HEADS
from torch.nn.modules.utils import _pair
import numpy as np
import mmcv


@HEADS.register_module
class CoarseMaskHead(nn.Module):
    """
    A mask head with fully connected layers. Given pooled features it first reduces channels and
    spatial dimensions with conv layers and then uses FC layers to predict coarse masks analogously
    to the standard box head.
    """

    def __init__(self, input_channels, conv_dim, input_h, input_w, num_classes=35, fc_dim=1024, num_fc=2, output_side_resolution=7, 
                    loss_mask=dict(type='CrossEntropyLoss'), **kwargs):
        super(CoarseMaskHead, self).__init__()
        self.num_classes            = num_classes  # NOTE include bg
        self.fc_dim                 = fc_dim
        self.output_side_resolution = output_side_resolution
        self.input_channels         = input_channels
        self.input_h                = input_h
        self.input_w                = input_w
        self.loss_mask = build_loss(loss_mask)
        self.fp16_enabled = False

        self.conv_layers = []
        if self.input_channels > conv_dim:
            self.reduce_channel_dim_conv = nn.Sequential(nn.Conv2d(
                self.input_channels,
                conv_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.ReLU())
            self.conv_layers.append(self.reduce_channel_dim_conv)

        self.reduce_spatial_dim_conv = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU())
        self.conv_layers.append(self.reduce_spatial_dim_conv)

        input_dim = conv_dim * self.input_h * self.input_w
        input_dim //= 4

        self.fcs = []
        for k in range(num_fc):
            fc = nn.Linear(input_dim, self.fc_dim)
            self.add_module("coarse_mask_fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            input_dim = self.fc_dim

        output_dim = self.num_classes * self.output_side_resolution * self.output_side_resolution
        self.prediction = nn.Linear(self.fc_dim, output_dim)
        self.init_weights()

    def init_weights(self,):
        # use normal distribution initialization for mask prediction layer
        nn.init.normal_(self.prediction.weight, std=0.001)
        nn.init.constant_(self.prediction.bias, 0)

        for layer in self.conv_layers:
            if isinstance(layer, nn.Sequential):
                assert len(layer)==2
                weight_init.c2_msra_fill(layer[0])
            else:
                weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)
            
    @auto_fp16()
    def forward(self, x):
        N = x.shape[0]
        x = x.view(N, self.input_channels, self.input_h, self.input_w)
        for layer in self.conv_layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        for layer in self.fcs:
            x = F.relu(layer(x))
        return self.prediction(x).view(
            N, self.num_classes, self.output_side_resolution, self.output_side_resolution
        )

    def get_target(self, sampling_results, gt_masks, mask_size=-1):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        mask_targets = self.mask_target(pos_proposals, pos_assigned_gt_inds, gt_masks, self.output_side_resolution if mask_size<=0 else mask_size)
        return mask_targets

    def mask_target(self, pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, mask_size):
        mask_size_list = [mask_size for _ in range(len(pos_proposals_list))]
        mask_targets = map(self.mask_target_single, pos_proposals_list,
                        pos_assigned_gt_inds_list, gt_masks_list, mask_size_list)
        mask_targets = torch.cat(list(mask_targets))
        return mask_targets

    def mask_target_single(self, pos_proposals, pos_assigned_gt_inds, gt_masks, mask_size_):
        mask_size = _pair(mask_size_)
        num_pos = pos_proposals.size(0)
        mask_targets = []
        if num_pos > 0:
            proposals_np = pos_proposals.cpu().numpy()
            _, maxh, maxw = gt_masks.shape
            proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw - 1)
            proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh - 1)
            pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()
            for i in range(num_pos):
                gt_mask = gt_masks[pos_assigned_gt_inds[i]]
                bbox = proposals_np[i, :].astype(np.int32)
                x1, y1, x2, y2 = bbox
                w = np.maximum(x2 - x1 + 1, 1)
                h = np.maximum(y2 - y1 + 1, 1)
                target = mmcv.imresize(gt_mask[y1:y1 + h, x1:x1 + w], mask_size[::-1])
                mask_targets.append(target)
            mask_targets = torch.from_numpy(np.stack(mask_targets)).float().to(pos_proposals.device)
        else:
            mask_targets = pos_proposals.new_zeros((0, ) + mask_size)
        return mask_targets

    @force_fp32(apply_to=('mask_pred', ))
    def loss(self, mask_pred, mask_targets, labels):
        loss = dict()
        loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask_coarse'] = loss_mask
        return loss