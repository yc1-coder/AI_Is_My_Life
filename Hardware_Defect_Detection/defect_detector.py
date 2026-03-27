"""
PyTorch 缺陷检测模型集成模块
提供可直接使用的检测模型架构
"""
import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, List, Tuple


class DefectDetector:
    """缺陷检测器基类"""

    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        初始化检测器
        :param num_classes: 类别数量（包括背景）
        :param pretrained: 是否使用预训练权重
        """
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.model = self._create_model()

    def _create_model(self) -> nn.Module:
        """创建模型（由子类实现）"""
        raise NotImplementedError

    def train(self, dataloader: torch.utils.data.DataLoader,
              device: torch.device, epochs: int, lr: float = 0.001):
        """
        训练模型
        :param dataloader: 数据加载器
        :param device: 训练设备
        :param epochs: 训练轮数
        :param lr: 学习率
        """
        self.model.to(device)
        self.model.train()

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

        for epoch in range(epochs):
            total_loss = 0.0

            for batch_idx, batch in enumerate(dataloader):
                images = [img.to(device) for img in batch['images']]
                targets = [{k: v.to(device) for k, v in t.items()}
                           for t in batch['targets']]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += losses.item()

            scheduler.step()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader,
                 device: torch.device) -> Dict[str, float]:
        """
        评估模型
        :param dataloader: 数据加载器
        :param device: 评估设备
        :return: 评估指标
        """
        self.model.to(device)
        self.model.eval()

        predictions = []
        ground_truths = []

        for batch in dataloader:
            images = [img.to(device) for img in batch['images']]
            outputs = self.model(images)

            predictions.extend(outputs)
            ground_truths.extend(batch['targets'])

        metrics = self._calculate_map(predictions, ground_truths)

        return metrics

    def _calculate_map(self, predictions: List[Dict],
                       ground_truths: List[Dict]) -> Dict[str, float]:
        """计算 mAP 指标"""
        iou_thresholds = [0.5, 0.75]
        map_results = {'map': 0.0, 'map_50': 0.0, 'map_75': 0.0}

        total_ap = 0.0
        count = 0

        for pred, target in zip(predictions, ground_truths):
            pred_boxes = pred['boxes'].cpu()
            pred_scores = pred['scores'].cpu()
            pred_labels = pred['labels'].cpu()

            target_boxes = target['boxes'].cpu()
            target_labels = target['labels'].cpu()

            if len(pred_boxes) == 0 or len(target_boxes) == 0:
                continue

            for thresh in iou_thresholds:
                tp = 0
                fp = 0

                for i in range(len(pred_boxes)):
                    max_iou = 0.0
                    for j in range(len(target_boxes)):
                        iou = self._calculate_iou(pred_boxes[i], target_boxes[j])
                        max_iou = max(max_iou, iou)

                    if max_iou >= thresh:
                        tp += 1
                    else:
                        fp += 1

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                total_ap += precision
                count += 1

        if count > 0:
            avg_ap = total_ap / count
            map_results['map'] = avg_ap
            map_results['map_50'] = avg_ap
            map_results['map_75'] = avg_ap

        return map_results

    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """计算两个边界框的 IoU"""
        x1_min, y1_min, w1, h1 = box1
        x2_min, y2_min, w2, h2 = box2

        x1_max = x1_min + w1
        y1_max = y1_min + h1
        x2_max = x2_min + w2
        y2_max = y2_min + h2

        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        box1_area = w1 * h1
        box2_area = w2 * h2

        union_area = box1_area + box2_area - inter_area

        iou = inter_area / union_area if union_area > 0 else 0

        return iou

    def predict(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        单张图像预测
        :param image: 输入图像
        :return: 预测结果
        """
        self.model.eval()
        prediction = self.model([image])
        return prediction[0]

    def save_checkpoint(self, path: str):
        """保存检查点"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)
        print(f"模型已保存到：{path}")

    def load_checkpoint(self, path: str):
        """加载检查点"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已从 {path} 加载")


class FasterRCNNDetector(DefectDetector):
    """Faster R-CNN 缺陷检测器"""

    def _create_model(self) -> nn.Module:
        model = fasterrcnn_resnet50_fpn(pretrained=self.pretrained)

        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

        return model


class RetinaNetDetector(DefectDetector):
    """RetinaNet 缺陷检测器"""

    def _create_model(self) -> nn.Module:
        model = retinanet_resnet50_fpn(pretrained=self.pretrained)

        in_channels = model.head.classification_head.conv[0].in_channels
        num_anchors = model.head.classification_head.num_anchors

        model.head.classification_head.num_classes = self.num_classes

        cls_logits = torch.nn.Conv2d(
            in_channels,
            num_anchors * self.num_classes,
            kernel_size=3,
            stride=1,
            padding=1
        )
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -4.595)

        model.head.classification_head.cls_logits = cls_logits

        return model
