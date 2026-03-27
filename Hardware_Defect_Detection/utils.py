"""
工具函数模块
提供常用的辅助函数
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple
import json


def visualize_annotations(image_path: str, annotations: List[Dict],
                          output_path: str = None, show: bool = False):
    """
    可视化标注结果
    :param image_path: 图像路径
    :param annotations: 标注列表
    :param output_path: 输出路径
    :param show: 是否显示
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像：{image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    colors = plt.cm.hsv(np.linspace(0, 1, 10))

    for ann in annotations:
        bbox = ann.get('bbox', ann.get('box', []))
        label = ann.get('label', ann.get('class_name', 'defect'))

        if len(bbox) == 4:
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
        elif len(bbox) == 4 and all(k in ann for k in ['xmin', 'ymin', 'xmax', 'ymax']):
            x1, y1, x2, y2 = ann['xmin'], ann['ymin'], ann['xmax'], ann['ymax']
        else:
            continue

        color = colors[hash(label) % 10][:3]
        color = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(image, str(label), (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(output_path, image)
        print(f"可视化结果已保存到：{output_path}")

    if show:
        plt.figure(figsize=(15, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def calculate_dataset_statistics(csv_path: str, image_dir: str) -> Dict:
    """
    计算数据集统计信息
    :param csv_path: CSV 文件路径
    :param image_dir: 图像目录
    :return: 统计信息字典
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    stats = {
        'total_samples': len(df),
        'total_images': df['image_name'].nunique() if 'image_name' in df.columns else df['image_path'].nunique(),
        'classes': {}
    }

    if 'class_name' in df.columns or 'category' in df.columns:
        class_col = 'class_name' if 'class_name' in df.columns else 'category'
        class_counts = df[class_col].value_counts().to_dict()
        stats['classes'] = class_counts

    if all(k in df.columns for k in ['width', 'height']):
        areas = df['width'] * df['height']
        stats['area_statistics'] = {
            'min': float(areas.min()),
            'max': float(areas.max()),
            'mean': float(areas.mean()),
            'std': float(areas.std()),
            'median': float(areas.median())
        }

    return stats


def export_to_onnx(model: torch.nn.Module, dummy_input: torch.Tensor,
                   output_path: str, opset_version: int = 11):
    """
    导出模型为 ONNX 格式
    :param model: PyTorch 模型
    :param dummy_input: 示例输入
    :param output_path: 输出路径
    :param opset_version: ONNX opset 版本
    """
    model.eval()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"模型已导出为 ONNX 格式：{output_path}")


def create_sample_data(output_dir: str, num_samples: int = 100):
    """
    创建示例数据用于测试
    :param output_dir: 输出目录
    :param num_samples: 样本数量
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = output_dir / 'images'
    image_dir.mkdir(parents=True, exist_ok=True)

    data = []

    for i in range(num_samples):
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image_name = f'defect_{i:04d}.jpg'
        image_path = image_dir / image_name

        cv2.imwrite(str(image_path), image)

        num_defects = np.random.randint(1, 5)

        for j in range(num_defects):
            x = np.random.randint(0, 400)
            y = np.random.randint(0, 400)
            w = np.random.randint(50, 100)
            h = np.random.randint(50, 100)

            class_name = np.random.choice(['scratch', 'dent', 'crack'])

            data.append({
                'image_name': image_name,
                'xmin': x,
                'ymin': y,
                'xmax': x + w,
                'ymax': y + h,
                'class_name': class_name
            })

    df = pd.DataFrame(data)
    csv_path = output_dir / 'annotations.csv'
    df.to_csv(csv_path, index=False)

    print(f"示例数据已生成:")
    print(f"  - 图像数量：{num_samples}")
    print(f"  - 标注数量：{len(data)}")
    print(f"  - 保存目录：{output_dir}")

    return csv_path, str(image_dir)
