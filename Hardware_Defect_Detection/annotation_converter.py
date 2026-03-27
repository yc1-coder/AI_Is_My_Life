"""
标注格式转换模块
支持 CSV 到 VOC/COCO 格式转换
"""
import pandas as pd
import numpy as np
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """边界框数据类"""
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    width: float
    height: float
    class_name: str
    class_id: int


class AnnotationConverter:
    """标注格式转换器"""

    def __init__(self, classes: Dict[str, int]):
        """
        初始化转换器
        :param classes: 类别名称到 ID 的映射字典
        """
        self.classes = classes
        self.class_names = {v: k for k, v in classes.items()}

    def csv_to_voc(self, csv_path: str, output_dir: str,
                   image_dir: str = None) -> List[str]:
        """
        CSV 转 VOC 格式
        :param csv_path: CSV 文件路径
        :param output_dir: 输出目录
        :param image_dir: 图像目录
        :return: 生成的 XML 文件列表
        """
        df = pd.read_csv(csv_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        xml_files = []

        # 按图像分组
        grouped = df.groupby('image_name' if 'image_name' in df.columns else 'image_path')

        for image_name, group in grouped:
            xml_file = self._create_voc_annotation(
                image_name=image_name,
                annotations=group,
                output_dir=output_dir,
                image_dir=image_dir
            )
            xml_files.append(xml_file)

        print(f"已生成 {len(xml_files)} 个 VOC 标注文件")
        return xml_files

    def _create_voc_annotation(self, image_name: str, annotations: pd.DataFrame,
                               output_dir: Path, image_dir: str = None) -> str:
        """创建单个 VOC 标注文件"""
        # 获取图像信息
        first_row = annotations.iloc[0]

        # 如果有图像路径，尝试读取图像尺寸
        width, height = 800, 600  # 默认值
        if image_dir and Path(image_dir).exists():
            import cv2
            img_path = Path(image_dir) / image_name
            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    height, width, _ = img.shape

        # 创建 XML 结构
        root = ET.Element("annotation")

        # 添加文件夹元素
        folder = ET.SubElement(root, "folder")
        folder.text = "defect_images"

        # 添加文件名元素
        filename = ET.SubElement(root, "filename")
        filename.text = image_name

        # 添加图像尺寸
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(width)
        ET.SubElement(size, "height").text = str(height)
        ET.SubElement(size, "depth").text = "3"

        # 添加标注对象
        for idx, row in annotations.iterrows():
            obj = ET.SubElement(root, "object")

            # 类别名称
            class_name = row.get('class_name', row.get('category', 'defect'))
            ET.SubElement(obj, "name").text = class_name

            # 边界框
            bndbox = ET.SubElement(obj, "bndbox")

            if all(k in row for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                ET.SubElement(bndbox, "xmin").text = str(row['xmin'])
                ET.SubElement(bndbox, "ymin").text = str(row['ymin'])
                ET.SubElement(bndbox, "xmax").text = str(row['xmax'])
                ET.SubElement(bndbox, "ymax").text = str(row['ymax'])
            elif all(k in row for k in ['x', 'y', 'width', 'height']):
                ET.SubElement(bndbox, "xmin").text = str(row['x'])
                ET.SubElement(bndbox, "ymin").text = str(row['y'])
                ET.SubElement(bndbox, "xmax").text = str(row['x'] + row['width'])
                ET.SubElement(bndbox, "ymax").text = str(row['y'] + row['height'])

        # 美化 XML
        xml_str = ET.tostring(root, encoding='utf-8')
        parsed = minidom.parseString(xml_str)
        pretty_xml = parsed.toprettyxml(indent="  ")

        # 保存文件
        xml_filename = Path(image_name).stem + ".xml"
        xml_path = output_dir / xml_filename

        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)

        return str(xml_path)

    def csv_to_coco(self, csv_path: str, output_path: str,
                    image_dir: str = None) -> str:
        """
        CSV 转 COCO 格式
        :param csv_path: CSV 文件路径
        :param output_path: 输出 JSON 文件路径
        :param image_dir: 图像目录
        :return: 输出文件路径
        """
        df = pd.read_csv(csv_path)

        # 初始化 COCO 格式
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        # 添加类别信息
        for class_name, class_id in self.classes.items():
            coco_format["categories"].append({
                "id": class_id,
                "name": class_name,
                "supercategory": "defect"
            })

        # 按图像分组
        if 'image_name' in df.columns:
            grouped = df.groupby('image_name')
        elif 'image_path' in df.columns:
            grouped = df.groupby('image_path')
        else:
            raise ValueError("CSV 必须包含'image_name'或'image_path'列")

        image_id = 0
        annotation_id = 0

        for image_name, group in grouped:
            # 获取图像信息
            first_row = group.iloc[0]

            # 尝试读取图像尺寸
            width, height = 800, 600
            if image_dir and Path(image_dir).exists():
                import cv2
                img_path = Path(image_dir) / image_name
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        height, width, _ = img.shape

            # 添加图像信息
            coco_format["images"].append({
                "id": image_id,
                "file_name": image_name,
                "width": width,
                "height": height,
                "license": 1,
                "flickr_url": "",
                "coco_url": "",
                "date_captured": ""
            })

            # 添加标注
            for idx, row in group.iterrows():
                # 获取边界框
                if all(k in row for k in ['xmin', 'ymin', 'width', 'height']):
                    bbox = [row['xmin'], row['ymin'], row['width'], row['height']]
                elif all(k in row for k in ['x', 'y', 'width', 'height']):
                    bbox = [row['x'], row['y'], row['width'], row['height']]
                elif all(k in row for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                    x_min = row['xmin']
                    y_min = row['ymin']
                    x_max = row['xmax']
                    y_max = row['ymax']
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                else:
                    continue

                # 获取类别 ID
                class_name = row.get('class_name', row.get('category', 'defect'))
                class_id = self.classes.get(class_name, 0)

                # 添加标注
                coco_format["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "segmentation": []
                })

                annotation_id += 1

            image_id += 1

        # 保存 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_format, f, indent=2, ensure_ascii=False)

        print(f"已生成 COCO 格式标注文件：{output_path}")
        print(f"图像数量：{len(coco_format['images'])}")
        print(f"标注数量：{len(coco_format['annotations'])}")

        return output_path
