import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.ops import nms, box_iou
from sklearn.metrics import confusion_matrix

# FCOS 的步长定义
STRIDES = [4, 8, 16]

class Evaluator:
    def __init__(self, device, num_classes=12, iou_threshold=0.4, score_threshold=0.2):
        self.device = device
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.reset()

    def reset(self):
        self.matched_preds = []
        self.matched_targets = []
        self.total_gt_count = 0
        self.detected_gt_count = 0

    def _coords_fmap2orig(self, h, w, stride):
        shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32, device=self.device)
        shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32, device=self.device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        return torch.stack((shift_x + stride / 2, shift_y + stride / 2), dim=1)

    def process_batch(self, outputs, targets, img_size=(300, 300)):
        # ... (process_batch 代码保持不变，逻辑不需要改) ...
        cls_logits = outputs['cls_logits']
        bbox_reg = outputs['bbox_reg']
        centerness = outputs['centerness']
        B = cls_logits[0].shape[0]
        
        for b in range(B):
            pred_boxes, pred_scores, pred_labels = [], [], []
            for i, stride in enumerate(STRIDES):
                cls = cls_logits[i][b].permute(1, 2, 0).sigmoid()
                ctr = centerness[i][b].permute(1, 2, 0).sigmoid()
                reg = bbox_reg[i][b].permute(1, 2, 0)
                H, W, C = cls.shape
                scores = torch.sqrt(cls * ctr).reshape(-1, C)
                max_scores, labels = scores.max(dim=1)
                keep_mask = max_scores > self.score_threshold
                if not keep_mask.any(): continue
                coords = self._coords_fmap2orig(H, W, stride)
                coords = coords[keep_mask]
                reg = reg.reshape(-1, 4)[keep_mask]
                scores = max_scores[keep_mask]
                labels = labels[keep_mask]
                x1 = coords[:, 0] - reg[:, 0]
                y1 = coords[:, 1] - reg[:, 1]
                x2 = coords[:, 0] + reg[:, 2]
                y2 = coords[:, 1] + reg[:, 3]
                x1 = x1.clamp(min=0, max=img_size[1])
                y1 = y1.clamp(min=0, max=img_size[0])
                x2 = x2.clamp(min=0, max=img_size[1])
                y2 = y2.clamp(min=0, max=img_size[0])
                boxes = torch.stack([x1, y1, x2, y2], dim=1)
                pred_boxes.append(boxes)
                pred_scores.append(scores)
                pred_labels.append(labels)
            
            if len(pred_boxes) == 0:
                self.total_gt_count += len(targets[b]['boxes'])
                continue
                
            pred_boxes = torch.cat(pred_boxes)
            pred_scores = torch.cat(pred_scores)
            pred_labels = torch.cat(pred_labels)
            keep_idx = nms(pred_boxes, pred_scores, iou_threshold=0.6)
            final_boxes = pred_boxes[keep_idx]
            final_labels = pred_labels[keep_idx]
            
            gt_boxes = targets[b]['boxes']
            gt_labels = targets[b]['labels']
            num_gt = len(gt_boxes)
            self.total_gt_count += num_gt
            if num_gt == 0 or len(final_boxes) == 0: continue
                
            ious = box_iou(final_boxes, gt_boxes)
            max_iou_per_gt, pred_idx_per_gt = ious.max(dim=0)
            for g in range(num_gt):
                iou = max_iou_per_gt[g]
                if iou > self.iou_threshold:
                    self.detected_gt_count += 1
                    pred_cls = final_labels[pred_idx_per_gt[g]].item()
                    gt_cls = gt_labels[g].item()
                    self.matched_preds.append(pred_cls)
                    self.matched_targets.append(gt_cls)

    def get_metrics(self):
        """计算并返回最终的评估指标 (已修改为支持 Macro Accuracy)"""
        # 1. 召回率 (Detection Recall)
        recall = self.detected_gt_count / max(self.total_gt_count, 1)
        
        cm = None
        class_acc = {}
        micro_acc = 0.0
        macro_acc = 0.0
        
        if len(self.matched_targets) > 0:
            # 计算混淆矩阵
            cm = confusion_matrix(
                self.matched_targets, 
                self.matched_preds, 
                labels=list(range(self.num_classes))
            )
            
            class_counts = cm.sum(axis=1) # 真实标签的数量
            correct_counts = cm.diagonal()
            
            valid_classes_count = 0
            sum_accuracy = 0.0

            for i in range(self.num_classes):
                if class_counts[i] > 0:
                    # 该类别的准确率
                    acc = correct_counts[i] / class_counts[i]
                    sum_accuracy += acc
                    valid_classes_count += 1
                else:
                    acc = 0.0 # 或者 np.nan，但在日志中 0.0 更安全
                
                class_acc[f"acc_class_{i}"] = acc
                
            # [原指标] Micro Accuracy (受样本量影响大)
            micro_acc = correct_counts.sum() / class_counts.sum()

            # [新指标] Macro Accuracy (即 Balanced Accuracy，所有类别权重相等)
            if valid_classes_count > 0:
                macro_acc = sum_accuracy / valid_classes_count
            else:
                macro_acc = 0.0

        else:
            cm = np.zeros((self.num_classes, self.num_classes), dtype=int)
            
        return {
            "Recall_Total": recall,
            "Acc_Micro": micro_acc,
            "Acc_Macro": macro_acc,
            "Class_Acc": class_acc,
            "Confusion_Matrix": cm
        }

    def plot_cm(self, cm, save_path):
        if cm is None: return
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=range(self.num_classes),
                    yticklabels=range(self.num_classes))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix (Normalized by True Class)")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()