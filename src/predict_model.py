"""
用户购买预测模型
基于用户行为特征，预测用户在未来 7 天内是否会购买
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb


def create_target(df: pd.DataFrame, predict_window: int = 7) -> pd.DataFrame:
    """
    为每个用户创建预测目标：
    在未来 predict_window 天内是否有购买行为

    思路：用数据集最后 N 天作为预测窗口，
          预测窗口之前的行为作为特征数据
    """
    max_date = df['datetime'].max()
    cutoff_date = max_date - timedelta(days=predict_window)

    # 训练期数据（cutoff 之前）
    train_data = df[df['datetime'] <= cutoff_date].copy()

    # 预测窗口内的购买用户
    predict_window_data = df[df['datetime'] > cutoff_date]
    users_with_buy = set(predict_window_data[predict_window_data['behavior_type'] == 'buy']['user_id'].unique())

    # 目标：在预测窗口内是否购买
    train_users = train_data['user_id'].unique()
    target_df = pd.DataFrame({'user_id': train_users})
    target_df['will_buy'] = target_df['user_id'].isin(users_with_buy).astype(int)

    return train_data, target_df


def build_prediction_model(df: pd.DataFrame, predict_window: int = 7,
                           test_size: float = 0.2, random_state: int = 42) -> dict:
    """
    构建用户购买预测模型

    Parameters
    ----------
    df : pd.DataFrame
        预处理后的全量数据
    predict_window : int
        预测窗口天数
    test_size : float
        测试集比例
    random_state : int
        随机种子

    Returns
    -------
    dict : 包含模型、评估指标、特征重要性
    """
    from .analysis import extract_user_features

    print(f'=== 用户购买预测模型 ===')
    print(f'预测窗口：未来 {predict_window} 天')
    print()

    # 1. 创建训练数据和目标
    train_data, target_df = create_target(df, predict_window=predict_window)

    # 2. 提取特征
    print('提取用户行为特征...')
    features_df = extract_user_features(train_data)

    # 3. 合并特征和目标
    model_data = features_df.merge(target_df, on='user_id', how='inner')
    print(f'样本数：{len(model_data)}')

    # 检查正负样本分布
    pos_count = model_data['will_buy'].sum()
    neg_count = len(model_data) - pos_count
    print(f'正样本（会购买）：{pos_count} ({pos_count/len(model_data)*100:.1f}%)')
    print(f'负样本（不购买）：{neg_count} ({neg_count/len(model_data)*100:.1f}%)')
    print()

    # 4. 准备训练数据
    feature_cols = [c for c in features_df.columns if c != 'user_id']
    X = model_data[feature_cols].copy()
    y = model_data['will_buy'].copy()

    # 处理可能的无穷值
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f'训练集：{len(X_train)}，测试集：{len(X_test)}')
    print()

    # 5. 训练 XGBoost
    # scale_pos_weight 处理样本不均衡
    scale_weight = neg_count / pos_count

    print('训练 XGBoost 模型...')
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_weight,
        random_state=random_state,
        eval_metric='logloss',
    )

    model.fit(X_train, y_train)

    # 6. 评估
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print('=== 模型评估 ===')
    print()
    print(classification_report(y_test, y_pred, target_names=['不购买', '会购买']))

    auc = roc_auc_score(y_test, y_prob)
    print(f'AUC: {auc:.4f}')
    print()

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print('混淆矩阵：')
    print(f'  真负例(TN): {cm[0][0]:5d}    假正例(FP): {cm[0][1]:5d}')
    print(f'  假负例(FN): {cm[1][0]:5d}    真正例(TP): {cm[1][1]:5d}')
    print()

    # 7. 特征重要性
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print('=== Top 10 最重要特征 ===')
    for i, row in importance_df.head(10).iterrows():
        print(f'  {row["feature"]:<20s} {row["importance"]:.4f}')

    # 8. 交叉验证
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
    print()
    print(f'5折交叉验证 AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

    return {
        'model': model,
        'auc': auc,
        'importance': importance_df,
        'feature_cols': feature_cols,
        'cv_auc_mean': cv_scores.mean(),
        'cv_auc_std': cv_scores.std(),
    }
