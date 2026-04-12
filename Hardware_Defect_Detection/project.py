"""
NEU-DET 缺陷检测系统 - YOLOv5 集成版
实现从原始数据到 YOLOv5 模型训练的端到端流程
"""
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
from tqdm import tqdm
import yaml
import torch


# ================= 1. 数据处理类 (保持不变) =================
class NEUDETDataset:
    def __init__(self, neu_det_root: str):
        self.root = Path(neu_det_root)
        self.train_dir = self.root / 'train'
        self.val_dir = self.root / 'validation'
        self.class_mapping = {
            'crazing': 0, 'inclusion': 1, 'patches': 2,
            'pitted_surface': 3, 'rolled-in_scale': 4, 'scratches': 5
        }

    def parse_voc_xml(self, xml_path: Path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find('size')
        width, height = int(size.find('width').text), int(size.find('height').text)
        filename = root.find('filename').text
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            xmin, ymin = float(bndbox.find('xmin').text), float(bndbox.find('ymin').text)
            xmax, ymax = float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)
            objects.append({'class_name': name, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax})
        return {'filename': filename, 'width': width, 'height': height, 'objects': objects}

    def convert_to_csv(self, split: str = 'train'):
        data_dir = self.train_dir if split == 'train' else self.val_dir
        annotation_dir = data_dir / 'annotations'
        all_records = []
        for xml_file in tqdm(annotation_dir.glob('*.xml'), desc=f"解析{split}集"):
            try:
                data = self.parse_voc_xml(xml_file)
                for obj in data['objects']:
                    all_records.append({
                        'image_name': data['filename'], 'width': data['width'], 'height': data['height'],
                        'class_name': obj['class_name'], 'xmin': obj['xmin'], 'ymin': obj['ymin'],
                        'xmax': obj['xmax'], 'ymax': obj['ymax']
                    })
            except:
                continue
        return pd.DataFrame(all_records)


# ================= 2. YOLO 格式转换器 (新增核心) =================

def convert_to_yolo_format(df, image_root, output_root, split='train'):
    """将 CSV 转换为 YOLOv5 需要的 labels/*.txt 和 images 结构"""
    labels_dir = output_root / 'labels' / split
    images_dir = output_root / 'images' / split
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    class_map = {'crazing': 0, 'inclusion': 1, 'patches': 2, 'pitted_surface': 3, 'rolled-in_scale': 4, 'scratches': 5}

    grouped = df.groupby('image_name')
    for img_name, group in tqdm(grouped, desc=f"转换{split}数据"):
        # 1. 寻找并复制图片到目标目录
        src_img_path = None
        # NEU-DET 的图片分布在不同的子文件夹中，需要遍历查找
        for class_folder in image_root.iterdir():
            if class_folder.is_dir():
                candidate = class_folder / img_name
                if candidate.exists():
                    src_img_path = candidate
                    break

        if src_img_path and src_img_path.exists():
            dst_img_path = images_dir / img_name
            if not dst_img_path.exists():
                shutil.copy(src_img_path, dst_img_path)
        else:
            print(f"警告：未找到图片 {img_name}")
            continue

        # 2. 生成 txt 标注
        txt_path = labels_dir / f"{Path(img_name).stem}.txt"
        with open(txt_path, 'w') as f:
            for _, row in group.iterrows():
                cls_id = class_map.get(row['class_name'], 0)
                w_img, h_img = row['width'], row['height']

                # 转换为 YOLO 格式: class x_center y_center width height (归一化)
                x_center = (row['xmin'] + row['xmax']) / 2 / w_img
                y_center = (row['ymin'] + row['ymax']) / 2 / h_img
                w_box = (row['xmax'] - row['xmin']) / w_img
                h_box = (row['ymax'] - row['ymin']) / h_img

                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}\n")



def create_dataset_yaml(output_dir, nc=6, names=None):
    """创建 dataset.yaml 配置文件"""
    if names is None:
        names = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

    data = {
        'path': str(output_dir.absolute()),
        'train': 'images/train',
        'val': 'images/validation',
        'nc': nc,
        'names': names
    }

    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    return yaml_path


# ================= 3. 主流程控制 =================
def run_yolo_pipeline():
    print("=" * 60 + "\n  NEU-DET YOLOv5 端到端训练流程\n" + "=" * 60)

    root = Path(__file__).parent
    neu_det_root = root / 'NEU-DET'
    output_dir = root / 'output_yolo'
    output_dir.mkdir(exist_ok=True)

    # Step 1: 解析数据
    print("\n[1/4] 解析原始 XML 数据...")
    processor = NEUDETDataset(str(neu_det_root))
    train_df = processor.convert_to_csv('train')
    val_df = processor.convert_to_csv('validation')

    # Step 2: 转换为 YOLO 格式
    print("\n[2/4] 转换为 YOLOv5 格式 (Labels & Images)...")
    # 注意：为了节省空间，这里只转换标注。实际运行时需确保图片路径在 yaml 中正确指向
    convert_to_yolo_format(train_df, neu_det_root / 'train' / 'images', output_dir, 'train')
    convert_to_yolo_format(val_df, neu_det_root / 'validation' / 'images', output_dir, 'validation')

    # Step 3: 生成配置文件
    print("\n[3/4] 生成 dataset.yaml...")
    yaml_path = create_dataset_yaml(output_dir)
    print(f"配置文件已生成: {yaml_path}")

    # Step 4: 启动 YOLOv5 训练
    print("\n[4/4] 启动 YOLOv5 训练...")
    try:
        from yolov5 import train as yolo_train

        # 使用 YOLOv5 的 train.run 接口，而不是 model.train()
        yolo_train.run(
            data=str(yaml_path),  # 数据集 yaml 文件路径
            epochs=20,  # 训练轮数
            imgsz=640,  # 图像尺寸
            batch=16,  # 批次大小
            weights='yolov5s.pt',  # 预训练权重
            name='neu_det_defect_exp',  # 实验名称
            project=str(output_dir / 'runs'),  # 输出目录
            exist_ok=True  # 覆盖已有结果
        )
        print("\n✅ 训练完成！模型保存在 output_yolo/runs 目录下")

    except ImportError:
        print("❌ 未检测到 YOLOv5 库。请运行: pip install yolov5")
    except Exception as e:
        print(f"❌ 训练过程中发生错误: {e}")



if __name__ == '__main__':
    run_yolo_pipeline()
