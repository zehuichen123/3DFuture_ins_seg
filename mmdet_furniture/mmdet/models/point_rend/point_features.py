# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch.nn import functional as F
from mmdet.core import auto_fp16, force_fp32


def point_sample(input, point_coords, **kwargs):
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def generate_regular_grid_point_coords(R, side_size, device, dtype):
    aff = torch.tensor([[[0.5, 0, 0.5], [0, 0.5, 0.5]]], device=device,dtype=dtype)
    r = F.affine_grid(aff, torch.Size((1, 1, side_size, side_size)), align_corners=False)
    return r.view(1, -1, 2).expand(R, -1, -1)


def get_uncertain_point_coords_with_randomness(coarse_logits, uncertainty_func, num_points, 
        oversample_ratio, importance_sample_ratio, return_indices=False):
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device, dtype=coarse_logits.dtype)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)

    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2)
    if num_random_points > 0:
        point_coords = torch.cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device, dtype=coarse_logits.dtype),
            ],
            dim=1,
        )
    if not return_indices:
        return [point_coords]    
    hh,ww=coarse_logits.size(-2),coarse_logits.size(-1)
    point_indices = torch.floor(torch.floor(point_coords[:,:,1]*hh)*ww+ww*point_coords[:,:,0])
    point_indices = point_indices.clamp(max=hh*ww-1)
    return [point_coords, point_indices]


def get_uncertain_point_coords_on_grid(uncertainty_map, num_points):
    R, _, H, W = uncertainty_map.shape
    h_step = 1.0 / float(H)
    w_step = 1.0 / float(W)

    num_points = min(H * W, num_points)
    point_indices = torch.topk(uncertainty_map.view(R, H * W), k=num_points, dim=1)[1]
    point_coords = torch.zeros(R, num_points, 2, dtype=uncertainty_map.dtype, device=uncertainty_map.device)
    point_coords[:, :, 0] = w_step / 2.0 + (point_indices % W).to(uncertainty_map.dtype) * w_step
    point_coords[:, :, 1] = h_step / 2.0 + (point_indices // W).to(uncertainty_map.dtype) * h_step
    return point_indices, point_coords


def point_sample_fine_grained_features(features_list, feature_scales, boxes, point_coords):
    if isinstance(boxes, list):
        cat_boxes = torch.cat(boxes, 0)
    else:
        cat_boxes = boxes
    num_boxes = [len(b) for b in boxes] if isinstance(boxes, list) else [boxes.size(0)]

    point_coords_wrt_image = get_point_coords_wrt_image(cat_boxes, point_coords)
    split_point_coords_wrt_image = torch.split(point_coords_wrt_image, num_boxes)  

    point_features = []
    for idx_img, point_coords_wrt_image_per_image in enumerate(split_point_coords_wrt_image):
        point_features_per_image = []
        for idx_feature, feature_map in enumerate(features_list):
            h, w = feature_map.shape[-2:]
            scale = torch.tensor([w, h], device=feature_map.device, dtype=feature_map[0].dtype) / feature_scales[idx_feature]
            point_coords_scaled = point_coords_wrt_image_per_image / scale
            point_features_per_image.append(
                point_sample(
                    feature_map[idx_img].unsqueeze(0),
                    point_coords_scaled.unsqueeze(0),
                    align_corners=False,
                )
                .squeeze(0)
                .transpose(1, 0)
            )
        point_features.append(torch.cat(point_features_per_image, dim=1))

    return torch.cat(point_features, dim=0), point_coords_wrt_image


def get_point_coords_wrt_image(boxes_coords, point_coords):
    with torch.no_grad():
        point_coords_wrt_image = point_coords.clone()
        point_coords_wrt_image[:, :, 0] = point_coords_wrt_image[:, :, 0] * (
            boxes_coords[:, None, 2] - boxes_coords[:, None, 0]
        )
        point_coords_wrt_image[:, :, 1] = point_coords_wrt_image[:, :, 1] * (
            boxes_coords[:, None, 3] - boxes_coords[:, None, 1]
        )
        point_coords_wrt_image[:, :, 0] += boxes_coords[:, None, 0]
        point_coords_wrt_image[:, :, 1] += boxes_coords[:, None, 1]

    return point_coords_wrt_image
