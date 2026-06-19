"""
数据加载与预处理模块
处理 Taobao User Behavior 数据集
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_data(filepath: str, nrows: int = None) -> pd.DataFrame:
    """
    加载原始数据

    Parameters
    ----------
    filepath : str
        数据文件路径
    nrows : int, optional
        读取的行数限制（数据集约1亿行，开发时建议限制）

    Returns
    -------
    pd.DataFrame
    """
    columns = ['user_id', 'item_id', 'category_id', 'behavior_type', 'timestamp']
    df = pd.read_csv(filepath, names=columns, nrows=nrows)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据预处理：
    1. 时间戳转换
    2. 提取时间特征
    3. 行为类型映射
    4. 去除异常值
    """
    # 时间戳转换
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # 行为类型映射
    behavior_map = {'pv': '浏览', 'buy': '购买', 'cart': '加购', 'fav': '收藏'}
    df['behavior_cn'] = df['behavior_type'].map(behavior_map)

    # 去除异常值：时间戳为0或极端值
    df = df[df['timestamp'] > 0].copy()

    return df


def get_basic_stats(df: pd.DataFrame) -> dict:
    """获取数据集基本统计信息"""
    stats = {
        '数据量': len(df),
        '用户数': df['user_id'].nunique(),
        '商品数': df['item_id'].nunique(),
        '品类数': df['category_id'].nunique(),
        '时间范围': f"{df['datetime'].min()} ~ {df['datetime'].max()}",
        '行为分布': df['behavior_type'].value_counts().to_dict(),
    }
    return stats


def sample_by_users(df: pd.DataFrame, n_users: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    按用户抽样（保留每个被抽中用户的完整行为记录）
    用于数据量过大时快速验证分析逻辑
    """
    np.random.seed(seed)
    users = df['user_id'].unique()
    sampled_users = np.random.choice(users, size=min(n_users, len(users)), replace=False)
    return df[df['user_id'].isin(sampled_users)].copy()
