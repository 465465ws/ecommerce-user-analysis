"""
核心分析模块
包含：转化漏斗、RFM 分层、留存分析、关联规则
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict


# ============ 漏斗分析 ============

def build_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算用户行为转化漏斗

    返回每个行为阶段的人数及转化率
    流程：浏览(pv) → 收藏(fav)/加购(cart) → 购买(buy)
    """
    # 各行为去重用户数
    pv_users = set(df[df['behavior_type'] == 'pv']['user_id'].unique())
    fav_users = set(df[df['behavior_type'] == 'fav']['user_id'].unique())
    cart_users = set(df[df['behavior_type'] == 'cart']['user_id'].unique())
    buy_users = set(df[df['behavior_type'] == 'buy']['user_id'].unique())

    # 有收藏或加购行为的用户（任一）
    interested_users = fav_users | cart_users

    funnel = pd.DataFrame({
        '阶段': ['浏览', '收藏/加购', '购买'],
        '用户数': [
            len(pv_users),
            len(interested_users),
            len(buy_users)
        ]
    })

    funnel['总体转化率'] = (funnel['用户数'] / len(pv_users) * 100).round(2)
    stage_rates = [100.0]
    for i in range(1, len(funnel)):
        rate = (funnel['用户数'].iloc[i] / funnel['用户数'].iloc[i - 1] * 100)
        stage_rates.append(round(rate, 2))
    funnel['阶段转化率'] = stage_rates

    return funnel


def funnel_by_category(df: pd.DataFrame, top_n: int = 10) -> dict:
    """按品类计算购买转化率"""
    cat_stats = df.groupby('category_id').agg(
        pv_count=('behavior_type', lambda x: (x == 'pv').sum()),
        buy_count=('behavior_type', lambda x: (x == 'buy').sum()),
    )

    cat_stats['conversion_rate'] = (cat_stats['buy_count'] / cat_stats['pv_count'] * 100).round(4)
    cat_stats = cat_stats[cat_stats['pv_count'] >= 100]  # 过滤低流量品类

    return cat_stats.sort_values('conversion_rate', ascending=False).head(top_n)


# ============ RFM 用户分层 ============

def compute_rfm(df: pd.DataFrame, reference_date: datetime = None) -> pd.DataFrame:
    """
    计算用户 RFM 指标

    R (Recency)  - 最近一次购买距参考日的天数
    F (Frequency) - 购买次数
    M (Monetary)  - 购买涉及的商品数（替代金额，数据集中无金额字段）
    """
    if reference_date is None:
        reference_date = df['datetime'].max() + timedelta(days=1)

    buy_df = df[df['behavior_type'] == 'buy'].copy()

    rfm = buy_df.groupby('user_id').agg(
        Recency=('datetime', lambda x: (reference_date - x.max()).days),
        Frequency=('item_id', 'nunique'),          # 购买了多少不同商品
        Monetary=('behavior_type', 'count'),        # 购买行为次数
    )

    # 按百分位分档（4档），比 qcut 更稳健，不会因为重复值报错
    # R: 越小越好 → 最近购买的用户分高
    rfm['R_Score'] = 4 - pd.cut(rfm['Recency'].rank(pct=True),
                                 bins=[0, 0.25, 0.5, 0.75, 1.0],
                                 labels=[4, 3, 2, 1], include_lowest=True).astype(int)
    # F: 越大越好
    rfm['F_Score'] = pd.cut(rfm['Frequency'].rank(pct=True),
                             bins=[0, 0.25, 0.5, 0.75, 1.0],
                             labels=[1, 2, 3, 4], include_lowest=True).astype(int)
    # M: 越大越好
    rfm['M_Score'] = pd.cut(rfm['Monetary'].rank(pct=True),
                             bins=[0, 0.25, 0.5, 0.75, 1.0],
                             labels=[1, 2, 3, 4], include_lowest=True).astype(int)

    rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

    # 用户分层
    def segment(score):
        if score >= 10:
            return '高价值用户'
        elif score >= 7:
            return '重要用户'
        elif score >= 5:
            return '一般用户'
        else:
            return '低价值用户'

    rfm['Segment'] = rfm['RFM_Score'].apply(segment)

    return rfm


# ============ 留存分析 ============

def compute_retention(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算用户次日/3日/7日/14日/30日留存

    按用户首次活跃日期划分群组，计算每个群组在各个间隔日的留存率
    """
    # 用户首次活跃日
    first_active = df.groupby('user_id')['date'].min().reset_index()
    first_active.columns = ['user_id', 'first_date']

    # 用户每日活跃记录
    user_days = df[['user_id', 'date']].drop_duplicates()
    user_days = user_days.merge(first_active, on='user_id')
    user_days['day_offset'] = (user_days['date'] - user_days['first_date']).apply(lambda x: x.days)

    # 群组大小：每个 first_date 有多少用户
    cohort_size = user_days.groupby('first_date')['user_id'].nunique()

    # 各留存日的用户数
    retention_days = [1, 3, 7, 14, 30]
    cohorts = cohort_size.to_frame('cohort_size')

    for day_n in retention_days:
        retained = user_days[user_days['day_offset'] == day_n].groupby('first_date')['user_id'].nunique()
        cohorts[f'day{day_n}'] = retained
        cohorts[f'day{day_n}_retention'] = (cohorts[f'day{day_n}'] / cohorts['cohort_size'] * 100).round(2)

    cohorts = cohorts.fillna(0)
    return cohorts


# ============ 用户行为特征提取 ============

def extract_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取用户行为特征，用于后续聚类/建模
    """
    features = df.groupby('user_id').agg(
        pv_count=('behavior_type', lambda x: (x == 'pv').sum()),
        fav_count=('behavior_type', lambda x: (x == 'fav').sum()),
        cart_count=('behavior_type', lambda x: (x == 'cart').sum()),
        buy_count=('behavior_type', lambda x: (x == 'buy').sum()),
        unique_items=('item_id', 'nunique'),
        unique_categories=('category_id', 'nunique'),
        active_days=('date', 'nunique'),
        active_hours=('hour', lambda x: x.nunique()),
        avg_hour=('hour', 'mean'),
        weekend_ratio=('is_weekend', 'mean'),
    ).fillna(0)

    # 衍生特征（分母为 0 时填 0）
    features['buy_ratio'] = (features['buy_count'] / features['pv_count'].replace(0, np.nan)).fillna(0)
    features['fav_cart_ratio'] = (
        (features['fav_count'] + features['cart_count']) / features['pv_count'].replace(0, np.nan)
    ).fillna(0)
    features['items_per_day'] = (
        features['unique_items'] / features['active_days'].replace(0, np.nan)
    ).fillna(0)

    return features


# ============ 商品关联规则 ============

def build_item_pairs(df: pd.DataFrame, min_buy_users: int = 5) -> list:
    """
    构建商品共现对（被同一用户购买的商品组合）

    返回适合 mlxtend 处理的事务格式
    """
    from mlxtend.frequent_patterns import apriori, association_rules

    # 筛选购买行为
    buy_df = df[df['behavior_type'] == 'buy'].copy()

    # 统计每个商品被购买的用户数
    item_user_count = buy_df.groupby('item_id')['user_id'].nunique()
    popular_items = item_user_count[item_user_count >= min_buy_users].index
    buy_df = buy_df[buy_df['item_id'].isin(popular_items)]

    # 构建事务：每个用户购买的商品列表
    transactions = buy_df.groupby('user_id')['item_id'].apply(list).tolist()

    return transactions


def mine_association_rules(df: pd.DataFrame, min_support: float = 0.01,
                           min_threshold: float = 0.5) -> pd.DataFrame:
    """
    挖掘关联规则
    """
    from mlxtend.frequent_patterns import apriori, association_rules
    from mlxtend.preprocessing import TransactionEncoder

    transactions = build_item_pairs(df)

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

    # 频繁项集
    frequent_itemsets = apriori(df_onehot, min_support=min_support, use_colnames=True)

    if len(frequent_itemsets) < 2:
        return pd.DataFrame()

    # 关联规则
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=min_threshold)
    rules = rules.sort_values('lift', ascending=False)

    return rules
