"""
异常值过滤模块
负责过滤图像噪声和标注错误的数据
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class OutlierFilter:
    """异常值过滤器"""

    def __init__(self, contamination: float = 0.05):
        """
        初始化异常值过滤器
        :param contamination: 异常值比例估计
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.fitted = False

    def fit(self, features_df: pd.DataFrame,
            feature_columns: List[str] = None) -> 'OutlierFilter':
        """
        拟合异常值检测模型
        :param features_df: 特征 DataFrame
        :param feature_columns: 用于检测的特征列
        """
        if feature_columns is None:
            feature_columns = [
                'area', 'perimeter', 'extent', 'mean_intensity',
                'std_intensity', 'circularity', 'aspect_ratio', 'solidity'
            ]

        # 标准化特征
        X = features_df[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)

        # 训练孤立森林
        self.isolation_forest.fit(X_scaled)
        self.fitted = True

        return self

    def detect_outliers(self, features_df: pd.DataFrame,
                        feature_columns: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        检测异常值
        :param features_df: 特征 DataFrame
        :param feature_columns: 用于检测的特征列
        :return: (原始数据，异常值标签系列)
        """
        if not self.fitted:
            raise ValueError("请先调用 fit() 方法拟合模型")

        if feature_columns is None:
            feature_columns = [
                'area', 'perimeter', 'extent', 'mean_intensity',
                'std_intensity', 'circularity', 'aspect_ratio', 'solidity'
            ]

        # 标准化
        X = features_df[feature_columns].values
        X_scaled = self.scaler.transform(X)

        # 预测异常值 (-1 表示异常值，1 表示正常值)
        outlier_labels = self.isolation_forest.predict(X_scaled)
        outlier_scores = self.isolation_forest.score_samples(X_scaled)

        # 转换为布尔系列
        is_outlier = outlier_labels == -1

        return features_df, is_outlier

    def filter_outliers(self, features_df: pd.DataFrame,
                        feature_columns: List[str] = None,
                        threshold: float = None) -> pd.DataFrame:
        """
        过滤异常值
        :param features_df: 特征 DataFrame
        :param feature_columns: 用于检测的特征列
        :param threshold: 自定义阈值（负分数）
        :return: 过滤后的 DataFrame
        """
        _, is_outlier = self.detect_outliers(features_df, feature_columns)

        if threshold is not None:
            # 使用自定义阈值
            scores = self.isolation_forest.score_samples(
                self.scaler.transform(features_df[feature_columns])
            )
            is_outlier = scores < threshold

        # 保留正常数据
        filtered_df = features_df[~is_outlier].copy()

        print(f"原始数据量：{len(features_df)}")
        print(f"检测到异常值：{is_outlier.sum()}")
        print(f"过滤后数据量：{len(filtered_df)}")
        print(f"数据保留率：{len(filtered_df) / len(features_df) * 100:.2f}%")

        return filtered_df

    def statistical_filter(self, features_df: pd.DataFrame,
                           feature_columns: List[str] = None,
                           n_sigma: float = 3.0) -> pd.DataFrame:
        """
        基于统计学的异常值过滤（3σ原则）
        :param features_df: 特征 DataFrame
        :param feature_columns: 用于检测的特征列
        :param n_sigma: 标准差倍数
        :return: 过滤后的 DataFrame
        """
        if feature_columns is None:
            feature_columns = ['area', 'perimeter', 'mean_intensity', 'std_intensity']

        mask = pd.Series(True, index=features_df.index)

        for col in feature_columns:
            mean = features_df[col].mean()
            std = features_df[col].std()

            # 标记在 [mean - n_sigma*std, mean + n_sigma*std] 范围外的数据
            col_mask = (features_df[col] >= mean - n_sigma * std) & \
                       (features_df[col] <= mean + n_sigma * std)

            mask &= col_mask

        filtered_df = features_df[mask].copy()

        print(f"统计学过滤结果:")
        print(f"原始数据量：{len(features_df)}")
        print(f"过滤后数据量：{len(filtered_df)}")
        print(f"移除数据量：{len(features_df) - len(filtered_df)}")

        return filtered_df
