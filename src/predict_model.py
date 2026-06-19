"""
用户购买预测模型 (v2)
基于用户行为特征 + 时序特征，预测用户在未来 N 天内是否会购买
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def create_target(df: pd.DataFrame, predict_window: int = 7) -> tuple:
    """
    为每个用户创建预测目标：
    用截止日前的数据做特征，预测截止日后 predict_window 天内是否购买
    """
    max_date = df['datetime'].max()
    cutoff_date = max_date - timedelta(days=predict_window)

    # 训练期数据
    train_data = df[df['datetime'] <= cutoff_date].copy()
    # 预测窗口
    future_data = df[df['datetime'] > cutoff_date]
    # 谁在预测窗口买了
    users_who_bought = set(future_data[future_data['behavior_type'] == 'buy']['user_id'].unique())

    # 目标 DF
    train_users = train_data['user_id'].unique()
    target_df = pd.DataFrame({'user_id': train_users})
    target_df['will_buy'] = target_df['user_id'].isin(users_who_bought).astype(int)

    return train_data, target_df


def build_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    构建用于预测的增强特征集
    在基础特征之上加入时序趋势和行为变化特征
    """
    # 基础特征
    base = df.groupby('user_id').agg(
        pv_count=('behavior_type', lambda x: (x == 'pv').sum()),
        fav_count=('behavior_type', lambda x: (x == 'fav').sum()),
        cart_count=('behavior_type', lambda x: (x == 'cart').sum()),
        buy_count=('behavior_type', lambda x: (x == 'buy').sum()),
        unique_items=('item_id', 'nunique'),
        unique_categories=('category_id', 'nunique'),
        active_days=('date', 'nunique'),
        active_hours=('hour', 'nunique'),
        avg_hour=('hour', 'mean'),
        weekend_ratio=('is_weekend', 'mean'),
        first_active=('datetime', 'min'),
        last_active=('datetime', 'max'),
    ).reset_index()

    # 衍生特征
    base['buy_ratio'] = (base['buy_count'] / base['pv_count'].replace(0, np.nan)).fillna(0)
    base['fav_ratio'] = (base['fav_count'] / base['pv_count'].replace(0, np.nan)).fillna(0)
    base['cart_ratio'] = (base['cart_count'] / base['pv_count'].replace(0, np.nan)).fillna(0)
    base['items_per_day'] = (base['unique_items'] / base['active_days'].replace(0, np.nan)).fillna(0)

    # 时序特征
    base['lifespan_days'] = (base['last_active'] - base['first_active']).dt.days
    base['lifespan_days'] = base['lifespan_days'].replace(0, 1)  # 避免除 0
    base['daily_pv'] = base['pv_count'] / base['active_days']
    base['daily_buy'] = base['buy_count'] / base['active_days']

    # 品类广度
    base['cat_per_item'] = base['unique_categories'] / base['unique_items'].replace(0, np.nan)

    # 是否有购买记录
    base['has_bought'] = (base['buy_count'] > 0).astype(int)

    # 活跃强度指标
    base['hours_per_day'] = base['active_hours'] / base['active_days'].replace(0, np.nan)

    # 清理
    base = base.drop(columns=['first_active', 'last_active'])
    base = base.replace([np.inf, -np.inf], np.nan).fillna(0)

    return base


def build_prediction_model(df: pd.DataFrame, predict_window: int = 7,
                           test_size: float = 0.2, tune: bool = True,
                           random_state: int = 42) -> dict:
    """
    构建并评估用户购买预测模型

    Parameters
    ----------
    df : pd.DataFrame — 预处理后的全量数据
    predict_window : int — 预测窗口（天）
    test_size : float — 测试集比例
    tune : bool — 是否超参数搜索
    random_state : int — 随机种子

    Returns
    -------
    dict — 模型、AUC、特征重要性等
    """
    print(f'=== 用户购买预测模型 (v2) ===')
    print(f'预测目标：未来 {predict_window} 天内是否会购买')
    print()

    # 1. 划分特征期和预测期
    train_data, target_df = create_target(df, predict_window=predict_window)

    # 2. 增强特征
    print('构建增强特征...')
    features_df = build_advanced_features(train_data)
    print(f'特征维度：{features_df.shape[1] - 1} 维')

    # 3. 合并
    model_data = features_df.merge(target_df, on='user_id', how='inner')
    print(f'样本数：{len(model_data)}')

    pos = model_data['will_buy'].sum()
    neg = len(model_data) - pos
    print(f'正样本：{int(pos)} ({pos/len(model_data)*100:.1f}%)')
    print(f'负样本：{int(neg)} ({neg/len(model_data)*100:.1f}%)')
    imbal_ratio = neg / pos if pos > 0 else 1
    print()

    # 4. 特征矩阵
    drop_cols = ['user_id', 'will_buy']
    feature_cols = [c for c in model_data.columns if c not in drop_cols]
    X = model_data[feature_cols].copy()
    y = model_data['will_buy'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f'训练集：{len(X_train)} | 测试集：{len(X_test)}')
    print()

    # 5. 超参数搜索
    if tune:
        print('超参数搜索中...')
        base_model = xgb.XGBClassifier(
            scale_pos_weight=imbal_ratio,
            random_state=random_state,
            eval_metric='logloss',
        )
        param_grid = {
            'n_estimators': [80, 120, 160],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.03, 0.1, 0.2],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.7, 0.85, 1.0],
        }
        search = RandomizedSearchCV(
            base_model, param_grid, n_iter=15, cv=3,
            scoring='roc_auc', random_state=random_state, n_jobs=-1,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print(f'最佳参数：{search.best_params_}')
        print(f'CV AUC (搜索)：{search.best_score_:.4f}')
    else:
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            scale_pos_weight=imbal_ratio, random_state=random_state,
            eval_metric='logloss',
        )
        model.fit(X_train, y_train)

    print()

    # 6. 评估
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)

    print('=== 模型评估 ===')
    print()
    print(classification_report(y_test, y_pred, target_names=['不购买', '会购买']))
    print(f'AUC: {auc:.4f}')

    cm = confusion_matrix(y_test, y_pred)
    print(f'混淆矩阵 — TN:{cm[0][0]} FP:{cm[0][1]} | FN:{cm[1][0]} TP:{cm[1][1]}')
    print(f'准确率：{(cm[0][0]+cm[1][1])/cm.sum()*100:.1f}%')
    print()

    # 7. 特征重要性
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print('=== Top 15 最重要特征 ===')
    for _, row in importance_df.head(15).iterrows():
        bar = '█' * int(row['importance'] * 100)
        print(f'  {row["feature"]:<20s} {row["importance"]:.4f}  {bar}')

    return {
        'model': model,
        'auc': auc,
        'importance': importance_df,
        'feature_cols': feature_cols,
    }
