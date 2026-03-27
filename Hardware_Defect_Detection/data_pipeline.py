
"""
数据预处理 Pipeline 模块
封装归一化、数据增强、批次加载等功能
"""
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import pandas as pd


class DefectDataset(Dataset):
    """缺陷检测数据集"""

    def __init__(self, csv_path: str, image_dir: str,
                 transform: Optional[Callable] = None,
                 return_path: bool = False):
        """
        初始化数据集
        :param csv_path: CSV 标注文件路径
        :param image_dir: 图像目录
        :param transform: 数据变换
        :param return_path: 是否返回图像路径
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.return_path = return_path

        # 按图像分组
        self.image_groups = self.df.groupby(
            'image_name' if 'image_name' in self.df.columns else 'image_path'
        )
        self.image_names = list(self.image_groups.groups.keys())

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_name = self.image_names[idx]
        image_path = self.image_dir / image_name

        # 读取图像
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取标注
        annotations = self.image_groups.get_group(image_name)

        # 提取边界框和类别
        boxes = []
        labels = []

        for _, row in annotations.iterrows():
            # 边界框
            if all(k in row for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                x_min = row['xmin']
                y_min = row['ymin']
                x_max = row['xmax']
                y_max = row['ymax']
                box = [x_min, y_min, x_max - x_min, y_max - y_min]
            elif all(k in row for k in ['x', 'y', 'width', 'height']):
                box = [row['x'], row['y'], row['width'], row['height']]
            else:
                continue

            boxes.append(box)

            # 类别
            class_name = row.get('class_name', row.get('category', 'defect'))
            labels.append(hash(class_name) % 1000)  # 简单的类别编码

        # 应用变换
        if self.transform and len(boxes) > 0:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed['image']
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['class_labels'], dtype=torch.long)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        result = {
            'image': image,
            'boxes': boxes,
            'labels': labels
        }

        if self.return_path:
            result['image_path'] = str(image_path)

        return result


class DataPipeline:
    """数据预处理 Pipeline"""

    def __init__(self, img_size: int = 512, batch_size: int = 16,
                 num_workers: int = 4):
        """
        初始化 Pipeline
        :param img_size: 图像尺寸
        :param batch_size: 批次大小
        :param num_workers: 数据加载线程数
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers

        # 定义训练集增强
        self.train_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

        # 定义验证集变换
        self.val_transform = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def create_train_loader(self, csv_path: str, image_dir: str,
                            shuffle: bool = True) -> DataLoader:
        """
        创建训练数据加载器
        :param csv_path: CSV 文件路径
        :param image_dir: 图像目录
        :param shuffle: 是否打乱
        :return: DataLoader
        """
        dataset = DefectDataset(
            csv_path=csv_path,
            image_dir=image_dir,
            transform=self.train_transform
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

        print(f"训练数据集大小：{len(dataset)}")
        print(f"批次大小：{self.batch_size}")
        print(f"批次数量：{len(loader)}")

        return loader

    def create_val_loader(self, csv_path: str, image_dir: str) -> DataLoader:
        """
        创建验证数据加载器
        :param csv_path: CSV 文件路径
        :param image_dir: 图像目录
        :return: DataLoader
        """
        dataset = DefectDataset(
            csv_path=csv_path,
            image_dir=image_dir,
            transform=self.val_transform
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=True
        )

        print(f"验证数据集大小：{len(dataset)}")

        return loader

    def _collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """自定义批处理函数"""
        images = torch.stack([item['image'] for item in batch])
        boxes = [item['boxes'] for item in batch]
        labels = [item['labels'] for item in batch]

        targets = {
            'boxes': boxes,
            'labels': labels
        }

        return {
            'images': images,
            'targets': targets
        }

    def get_statistics(self, csv_path: str, image_dir: str) -> Dict:
        """
        获取数据集统计信息
        :param csv_path: CSV 文件路径
        :param image_dir: 图像目录
        :return: 统计信息字典
        """
        df = pd.read_csv(csv_path)

        stats = {
            'total_images': df['image_name'].nunique() if 'image_name' in df.columns else df['image_path'].nunique(),
            'total_annotations': len(df),
            'avg_annotations_per_image': len(df) / stats['total_images'],
            'class_distribution': df['class_name'].value_counts().to_dict() if 'class_name' in df.columns else {},
            'bbox_area_stats': {}
        }

        # 计算边界框面积统计
        if all(k in df.columns for k in ['width', 'height']):
            areas = df['width'] * df['height']
            stats['bbox_area_stats'] = {
                'min': float(areas.min()),
                'max': float(areas.max()),
                'mean': float(areas.mean()),
                'median': float(areas.median())
            }

        return stats
