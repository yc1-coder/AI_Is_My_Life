"""
硬件缺陷检测数据工程系统 - 主流程控制器
整合所有模块，提供端到端的数据处理流程
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
import argparse

from feature_extractor import DefectFeatureExtractor
from outlier_filter import OutlierFilter
from annotation_converter import AnnotationConverter
from data_pipeline import DataPipeline
from defect_detector import FasterRCNNDetector, RetinaNetDetector


class DefectDataEngine:
    """缺陷数据工程引擎"""

    def __init__(self, config: Dict):
        """
        初始化引擎
        :param config: 配置字典
        """
        self.config = config
        self.feature_extractor = DefectFeatureExtractor()
        self.outlier_filter = None
        self.annotation_converter = None
        self.data_pipeline = None

        # 创建工作目录
        self.work_dir = Path(config.get('work_dir', './output'))
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def run_full_pipeline(self, input_csv: str, image_dir: str,
                          output_format: str = 'coco') -> Dict:
        """
        运行完整的数据处理流程
        :param input_csv: 输入 CSV 文件路径
        :param image_dir: 图像目录
        :param output_format: 输出格式（voc/coco）
        :return: 处理结果统计
        """
        print("=" * 60)
        print("硬件缺陷视觉检测数据工程系统")
        print("=" * 60)

        # 步骤 1: 特征提取
        print("\n[步骤 1/5] 提取缺陷特征...")
        features_df = self.feature_extractor.batch_extract(
            pd.read_csv(input_csv)
        )
        features_path = self.work_dir / 'defect_features.csv'
        features_df.to_csv(features_path, index=False)
        print(f"✓ 特征已保存到：{features_path}")

        # 步骤 2: 异常值过滤
        print("\n[步骤 2/5] 过滤异常值...")
        self.outlier_filter = OutlierFilter(contamination=0.05)
        self.outlier_filter.fit(features_df)
        filtered_df = self.outlier_filter.filter_outliers(features_df)
        filtered_path = self.work_dir / 'filtered_features.csv'
        filtered_df.to_csv(filtered_path, index=False)
        print(f"✓ 过滤后的数据已保存到：{filtered_path}")

        # 步骤 3: 格式转换
        print(f"\n[步骤 3/5] 转换为{output_format.upper()}格式...")
        classes = {'defect': 1, 'scratch': 2, 'dent': 3, 'crack': 4}
        self.annotation_converter = AnnotationConverter(classes)

        if output_format.lower() == 'voc':
            output_dir = self.work_dir / 'voc_annotations'
            xml_files = self.annotation_converter.csv_to_voc(
                input_csv, output_dir, image_dir
            )
            print(f"✓ 生成 {len(xml_files)} 个 VOC 标注文件")
        elif output_format.lower() == 'coco':
            output_path = self.work_dir / 'coco_annotations.json'
            self.annotation_converter.csv_to_coco(
                input_csv, output_path, image_dir
            )
            print(f"✓ COCO 标注文件已保存到：{output_path}")

        # 步骤 4: 数据统计
        print("\n[步骤 4/5] 数据集统计分析...")
        self.data_pipeline = DataPipeline(
            img_size=self.config.get('img_size', 512),
            batch_size=self.config.get('batch_size', 16)
        )
        stats = self.data_pipeline.get_statistics(input_csv, image_dir)

        print(f"总图像数：{stats['total_images']}")
        print(f"总标注数：{stats['total_annotations']}")
        print(f"平均每张图像标注数：{stats['avg_annotations_per_image']:.2f}")

        # 步骤 5: 创建数据加载器
        print("\n[步骤 5/5] 创建 PyTorch 数据加载器...")
        train_loader = self.data_pipeline.create_train_loader(input_csv, image_dir)
        val_loader = self.data_pipeline.create_val_loader(input_csv, image_dir)

        print(f"✓ 训练集批次数量：{len(train_loader)}")
        print(f"✓ 验证集批次数量：{len(val_loader)}")

        # 返回统计信息
        results = {
            'features_extracted': len(features_df),
            'after_filtering': len(filtered_df),
            'filter_rate': len(filtered_df) / len(features_df),
            'statistics': stats,
            'train_batches': len(train_loader),
            'val_batches': len(val_loader)
        }

        print("\n" + "=" * 60)
        print("数据处理流程完成！")
        print("=" * 60)

        return results

    def train_model(self, train_csv: str, image_dir: str,
                    model_type: str = 'fasterrcnn',
                    epochs: int = 10, lr: float = 0.001):
        """
        训练缺陷检测模型
        :param train_csv: 训练集 CSV 文件
        :param image_dir: 图像目录
        :param model_type: 模型类型
        :param epochs: 训练轮数
        :param lr: 学习率
        """
        print("\n开始训练缺陷检测模型...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备：{device}")

        num_classes = 5
        if model_type.lower() == 'fasterrcnn':
            detector = FasterRCNNDetector(num_classes=num_classes, pretrained=True)
        elif model_type.lower() == 'retinanet':
            detector = RetinaNetDetector(num_classes=num_classes, pretrained=True)
        else:
            raise ValueError(f"不支持的模型类型：{model_type}")

        pipeline = DataPipeline(img_size=512, batch_size=8)
        train_loader = pipeline.create_train_loader(train_csv, image_dir)

        detector.train(train_loader, device, epochs=epochs, lr=lr)

        checkpoint_path = self.work_dir / f'{model_type}_checkpoint.pth'
        detector.save_checkpoint(str(checkpoint_path))

        print(f"\n✓ 模型训练完成，已保存到：{checkpoint_path}")

        return detector


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='硬件缺陷检测数据工程系统')
    parser.add_argument('--input_csv', type=str, required=True, help='输入 CSV 文件路径')
    parser.add_argument('--image_dir', type=str, required=True, help='图像目录')
    parser.add_argument('--output_format', type=str, default='coco', choices=['voc', 'coco'], help='输出格式')
    parser.add_argument('--work_dir', type=str, default='./output', help='工作目录')
    parser.add_argument('--img_size', type=int, default=512, help='图像尺寸')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--train', action='store_true', help='是否训练模型')
    parser.add_argument('--model_type', type=str, default='fasterrcnn', help='模型类型')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')

    args = parser.parse_args()

    config = {
        'work_dir': args.work_dir,
        'img_size': args.img_size,
        'batch_size': args.batch_size
    }

    engine = DefectDataEngine(config)

    results = engine.run_full_pipeline(
        input_csv=args.input_csv,
        image_dir=args.image_dir,
        output_format=args.output_format
    )

    if args.train:
        engine.train_model(
            train_csv=args.input_csv,
            image_dir=args.image_dir,
            model_type=args.model_type,
            epochs=args.epochs
        )


if __name__ == '__main__':
    import torch

    main()
