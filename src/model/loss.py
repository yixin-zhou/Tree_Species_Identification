import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

INF = 100000000
RANGE_ON_LEVELS = [
    [-1, 64],
    [64, 128],
    [128, INF],
]

def coords_fmap2orig(feature, stride):
    h, w = feature.shape[-2:]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32, device=feature.device)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32, device=feature.device)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x + stride / 2, shift_y + stride / 2), dim=1)
    return locations

class IOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect

        ious = (area_intersect + 1.0) / (area_union + 1.0)

        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion
        
        losses = 1 - gious
        if weight is not None:
            losses = losses * weight
        
        return losses.mean()

class FCOSLoss(nn.Module):
    def __init__(self, strides=[4, 8, 16], sparse_ignore_threshold=0.3):
        super().__init__()
        self.strides = strides

        self.sparse_ignore_threshold = sparse_ignore_threshold
        
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss_func = IOULoss()

    def forward(self, preds, targets):
        cls_logits = preds['cls_logits']
        bbox_regs = preds['bbox_reg']
        centerness = preds['centerness']

        target_cls, target_reg, target_center = self.prepare_targets(cls_logits, targets)

        return self.compute_loss(
            cls_logits, bbox_regs, centerness, 
            target_cls, target_reg, target_center
        )

    def prepare_targets(self, cls_logits, targets):
        all_level_points = []
        for i, stride in enumerate(self.strides):
            points = coords_fmap2orig(cls_logits[i], stride)
            all_level_points.append(points)
        all_points = torch.cat(all_level_points, dim=0) 

        labels, reg_targets, centerness = self.compute_targets_for_locations(
            all_points, targets, cls_logits
        )
        return labels, reg_targets, centerness

    def compute_targets_for_locations(self, locations, targets, cls_logits):
        num_points = locations.shape[0]
        num_targets = len(targets)
        labels_list = []
        reg_targets_list = []
        centerness_list = []

        num_points_per_level = [l.shape[-1]*l.shape[-2] for l in cls_logits]

        for i in range(num_targets):
            target = targets[i]
            boxes = target['boxes']
            
            classes = target['labels'] 

            if boxes.shape[0] == 0:
                labels_list.append(torch.full((num_points,), -1, device=locations.device))
                reg_targets_list.append(torch.zeros((num_points, 4), device=locations.device))
                centerness_list.append(torch.zeros((num_points,), device=locations.device))
                continue

            xs, ys = locations[:, 0], locations[:, 1]
            l = xs[:, None] - boxes[:, 0][None, :]
            t = ys[:, None] - boxes[:, 1][None, :]
            r = boxes[:, 2][None, :] - xs[:, None]
            b = boxes[:, 3][None, :] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets = reg_targets_per_im.max(dim=2)[0]
            is_cared_in_the_level = torch.zeros_like(is_in_boxes)
            start_idx = 0
            for level_idx, n_p in enumerate(num_points_per_level):
                end_idx = start_idx + n_p
                lvl_min, lvl_max = RANGE_ON_LEVELS[level_idx]
                is_cared_in_the_level[start_idx:end_idx, :] = \
                    (max_reg_targets[start_idx:end_idx, :] >= lvl_min) & \
                    (max_reg_targets[start_idx:end_idx, :] <= lvl_max)
                start_idx = end_idx
            
            match_matrix = is_in_boxes & is_cared_in_the_level

            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            areas = areas[None].repeat(num_points, 1)
            areas[~match_matrix] = INF
            min_area, min_area_inds = areas.min(dim=1)

            labels_per_im = classes[min_area_inds]
            labels_per_im[min_area == INF] = -1
            
            reg_targets_per_im = reg_targets_per_im[range(num_points), min_area_inds]
            
            left_right = reg_targets_per_im[:, [0, 2]]
            top_bottom = reg_targets_per_im[:, [1, 3]]
            centerness_per_im = torch.sqrt(
                (left_right.min(dim=1)[0] / left_right.max(dim=1)[0]) * \
                (top_bottom.min(dim=1)[0] / top_bottom.max(dim=1)[0])
            )
            centerness_per_im[min_area == INF] = 0

            labels_list.append(labels_per_im)
            reg_targets_list.append(reg_targets_per_im)
            centerness_list.append(centerness_per_im)

        return torch.stack(labels_list), torch.stack(reg_targets_list), torch.stack(centerness_list)

    def compute_loss(self, cls_logits, bbox_regs, centerness, 
                     target_cls, target_reg, target_center):
        
        flatten_cls_logits = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, p.shape[1]) for p in cls_logits], dim=0)
        flatten_bbox_regs = torch.cat([p.permute(0, 2, 3, 1).reshape(-1, 4) for p in bbox_regs], dim=0)
        flatten_centerness = torch.cat([p.permute(0, 2, 3, 1).reshape(-1) for p in centerness], dim=0)

        flatten_target_cls = target_cls.view(-1)
        flatten_target_reg = target_reg.view(-1, 4)
        flatten_target_center = target_center.view(-1)

        num_classes = flatten_cls_logits.shape[1] # 12

        target_one_hot = torch.zeros_like(flatten_cls_logits)

        pos_inds = torch.nonzero(flatten_target_cls >= 0).squeeze(1)
        
        if pos_inds.numel() > 0:
            pos_labels = flatten_target_cls[pos_inds].long()
            target_one_hot[pos_inds, pos_labels] = 1.0

        loss_cls_elementwise = ops.sigmoid_focal_loss(
            flatten_cls_logits, 
            target_one_hot, 
            alpha=0.25, 
            gamma=2.0, 
            reduction="none"
        )

        bg_inds = (flatten_target_cls == -1)
        
        bg_probs = torch.sigmoid(flatten_cls_logits[bg_inds])

        with torch.no_grad():
            probs = torch.sigmoid(flatten_cls_logits)
            max_probs, _ = probs.max(dim=1)
            should_ignore = (flatten_target_cls == -1) & (max_probs > self.sparse_ignore_threshold)
            loss_weight = torch.ones_like(max_probs)
            loss_weight[should_ignore] = 0.0

        loss_cls = (loss_cls_elementwise.sum(dim=1) * loss_weight).sum()

        num_pos = max(pos_inds.numel(), 1.0)
        loss_cls = loss_cls / num_pos

        if pos_inds.numel() > 0:
            pos_bbox_preds = flatten_bbox_regs[pos_inds]
            pos_bbox_targets = flatten_target_reg[pos_inds]
            pos_centerness_preds = flatten_centerness[pos_inds]
            pos_centerness_targets = flatten_target_center[pos_inds]

            loss_bbox = self.iou_loss_func(pos_bbox_preds, pos_bbox_targets, pos_centerness_targets)

            loss_centerness = self.centerness_loss_func(pos_centerness_preds, pos_centerness_targets).mean()
        else:
            loss_bbox = torch.tensor(0.0, device=flatten_bbox_regs.device)
            loss_centerness = torch.tensor(0.0, device=flatten_centerness.device)

        return {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_centerness": loss_centerness
        }