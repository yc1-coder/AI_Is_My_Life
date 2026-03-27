
"""
硬件缺陷特征提取模块
负责提取缺陷的尺寸、灰度分布、形态学特征
"""
import pandas as pd
import numpy as np
import cv2
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DefectFeature:
    """缺陷特征数据类"""
    defect_id: str
    image_path: str
    # 尺寸特征
    area: float
    perimeter: float
    bounding_box_width: float
    bounding_box_height: float
    extent: float
    # 灰度分布特征
    mean_intensity: float
    std_intensity: float
    skewness: float
    kurtosis: float
    # 形态学特征
    circularity: float
    aspect_ratio: float
    solidity: float
    hu_moments: np.ndarray


class DefectFeatureExtractor:
    """缺陷特征提取器"""

    def __init__(self):
        self.feature_columns = [
            'area', 'perimeter', 'bounding_box_width', 'bounding_box_height',
            'extent', 'mean_intensity', 'std_intensity', 'skewness', 'kurtosis',
            'circularity', 'aspect_ratio', 'solidity'
        ]

    def extract_shape_features(self, contour: np.ndarray) -> Dict[str, float]:
        """提取形状特征"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)

        # 计算外接矩形面积
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0

        # 圆形度
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0

        # 纵横比
        aspect_ratio = w / h if h > 0 else 0

        # 凸性检测
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        return {
            'area': area,
            'perimeter': perimeter,
            'bounding_box_width': w,
            'bounding_box_height': h,
            'extent': extent,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity
        }

    def extract_intensity_features(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """提取灰度分布特征"""
        # 提取缺陷区域的像素值
        pixel_values = image[mask > 0]

        if len(pixel_values) == 0:
            return {
                'mean_intensity': 0.0,
                'std_intensity': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }

        mean_intensity = np.mean(pixel_values)
        std_intensity = np.std(pixel_values)

        # 偏度和峰度
        if std_intensity > 0:
            skewness = np.mean(((pixel_values - mean_intensity) / std_intensity) ** 3)
            kurtosis = np.mean(((pixel_values - mean_intensity) / std_intensity) ** 4) - 3
        else:
            skewness = 0.0
            kurtosis = 0.0

        return {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def extract_hu_moments(self, contour: np.ndarray) -> np.ndarray:
        """提取 Hu 矩特征"""
        moments = cv2.HuMoments(cv2.HuMoments(cv2.moments(contour))).flatten()
        # 对数变换
        hu_log = -np.sign(moments) * np.log10(np.abs(moments) + 1e-10)
        return hu_log

    def extract_from_image(self, image_path: str, defect_id: str,
                           mask: np.ndarray = None) -> DefectFeature:
        """从图像中提取完整特征"""
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError(f"无法读取图像：{image_path}")

        # 如果没有提供 mask，使用阈值分割
        if mask is None:
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            mask = thresh

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError(f"图像中未找到缺陷轮廓：{image_path}")

        # 假设最大轮廓是缺陷区域
        largest_contour = max(contours, key=cv2.contourArea)

        # 提取形状特征
        shape_features = self.extract_shape_features(largest_contour)

        # 提取灰度特征
        intensity_features = self.extract_intensity_features(image, mask)

        # 提取 Hu 矩
        hu_moments = self.extract_hu_moments(largest_contour)

        return DefectFeature(
            defect_id=defect_id,
            image_path=image_path,
            **shape_features,
            **intensity_features,
            hu_moments=hu_moments
        )

    def batch_extract(self, csv_data: pd.DataFrame,
                      image_column: str = 'image_path',
                      id_column: str = 'defect_id') -> pd.DataFrame:
        """批量提取特征"""
        features_list = []

        for idx, row in csv_data.iterrows():
            try:
                feature = self.extract_from_image(
                    image_path=row[image_column],
                    defect_id=row[id_column]
                )
                features_list.append(feature)
                print(f"已处理 {idx + 1}/{len(csv_data)}: {row[id_column]}")
            except Exception as e:
                print(f"处理失败 {row[id_column]}: {str(e)}")

        # 转换为 DataFrame
        features_df = pd.DataFrame([{
            'defect_id': f.defect_id,
            'image_path': f.image_path,
            'area': f.area,
            'perimeter': f.perimeter,
            'bounding_box_width': f.bounding_box_width,
            'bounding_box_height': f.bounding_box_height,
            'extent': f.extent,
            'mean_intensity': f.mean_intensity,
            'std_intensity': f.std_intensity,
            'skewness': f.skewness,
            'kurtosis': f.kurtosis,
            'circularity': f.circularity,
            'aspect_ratio': f.aspect_ratio,
            'solidity': f.solidity
        } for f in features_list])

        return features_df
