"""
配置文件模块
定义系统配置参数
"""
from pathlib import Path


class Config:
    """系统配置类"""

    def __init__(self):
        # 路径配置
        self.WORK_DIR = Path('./output')
        self.IMAGE_DIR = Path('./data/images')
        self.ANNOTATION_DIR = Path('./data/annotations')
        self.MODEL_DIR = Path('./models')

        # 数据配置
        self.IMG_SIZE = 512
        self.BATCH_SIZE = 16
        self.NUM_WORKERS = 4

        # 特征提取配置
        self.FEATURE_COLUMNS = [
            'area', 'perimeter', 'extent', 'mean_intensity',
            'std_intensity', 'circularity', 'aspect_ratio', 'solidity'
        ]

        # 异常值检测配置
        self.OUTLIER_CONTAMINATION = 0.05
        self.OUTLIER_N_SIGMA = 3.0

        # 数据增强配置
        self.AUGMENTATION_PROB = 0.5
        self.HORIZONTAL_FLIP_PROB = 0.5
        self.VERTICAL_FLIP_PROB = 0.5
        self.ROTATE_90_PROB = 0.5
        self.COLOR_JITTER_PROB = 0.5
        self.GAUSSIAN_BLUR_PROB = 0.3
        self.GAUSSIAN_NOISE_PROB = 0.3

        # 归一化配置
        self.NORMALIZE_MEAN = [0.485, 0.456, 0.406]
        self.NORMALIZE_STD = [0.229, 0.224, 0.225]

        # 模型训练配置
        self.NUM_CLASSES = 5
        self.NUM_EPOCHS = 10
        self.LEARNING_RATE = 0.001
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 0.0005
        self.LR_STEP_SIZE = 3
        self.LR_GAMMA = 0.1

        # 类别映射
        self.CLASS_MAPPING = {
            'background': 0,
            'defect': 1,
            'scratch': 2,
            'dent': 3,
            'crack': 4
        }

        # 评估配置
        self.IOU_THRESHOLDS = [0.5, 0.75]
        self.SCORE_THRESHOLD = 0.5

        # 日志配置
        self.LOG_LEVEL = 'INFO'
        self.LOG_DIR = Path('./logs')

        # 创建必要的目录
        self._create_directories()

    def _create_directories(self):
        """创建必要的目录"""
        for dir_path in [self.WORK_DIR, self.IMAGE_DIR, self.ANNOTATION_DIR,
                         self.MODEL_DIR, self.LOG_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self):
        """转换为字典"""
        return self.__dict__.copy()

    def save(self, path: str):
        """保存配置到 JSON 文件"""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        print(f"配置已保存到：{path}")

    @classmethod
    def load(cls, path: str):
        """从 JSON 文件加载配置"""
        import json

        config = cls()
        path = Path(path)

        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)

            for key, value in saved_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

            print(f"配置已从 {path} 加载")

        return config


if __name__ == '__main__':
    config = Config()
    config.save('./config.json')
    print("配置已保存")
