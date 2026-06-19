"""
可视化模块
统一管理项目中所有图表样式和绘制逻辑
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import pandas as pd

# ============ 全局样式 ============

# 中文字体配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
PALETTE = sns.color_palette(COLORS)

sns.set_style("whitegrid")
sns.set_palette(PALETTE)


def save_fig(fig, name: str, path: str = 'reports/'):
    """保存图表"""
    fig.savefig(f'{path}{name}.png', dpi=150, bbox_inches='tight')


# ============ 行为分布 ============

def plot_behavior_distribution(df: pd.DataFrame) -> plt.Figure:
    """用户行为类型分布"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 饼图
    behavior_counts = df['behavior_type'].value_counts()
    labels = [f'{k}\n({v:,})' for k, v in behavior_counts.items()]
    axes[0].pie(behavior_counts.values, labels=labels, autopct='%1.1f%%',
                colors=COLORS[:4], explode=(0, 0.1, 0, 0))
    axes[0].set_title('用户行为类型分布', fontsize=14, fontweight='bold')

    # 柱状图
    ax = axes[1]
    bars = ax.bar(behavior_counts.index, behavior_counts.values, color=COLORS[:4], edgecolor='white')
    ax.set_title('各行为类型数量', fontsize=14, fontweight='bold')
    ax.set_ylabel('次数')
    for bar, val in zip(bars, behavior_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + val * 0.01,
                f'{val:,}', ha='center', fontsize=11)

    plt.tight_layout()
    return fig


# ============ 时间维度 ============

def plot_hourly_behavior(df: pd.DataFrame) -> plt.Figure:
    """用户行为小时分布"""
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, behavior in enumerate(['pv', 'buy', 'cart', 'fav']):
        hourly = df[df['behavior_type'] == behavior].groupby('hour').size()
        hourly = hourly.reindex(range(24), fill_value=0)
        ax.plot(hourly.index, hourly.values, marker='o', color=COLORS[i],
                linewidth=2, markersize=5, label=behavior)

    ax.set_title('各时段用户行为分布', fontsize=14, fontweight='bold')
    ax.set_xlabel('小时')
    ax.set_ylabel('行为次数')
    ax.legend(loc='upper left')
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, alpha=0.3)

    # 标注高峰
    pv_hourly = df[df['behavior_type'] == 'pv'].groupby('hour').size()
    peak_hour = pv_hourly.idxmax()
    ax.axvline(x=peak_hour, color='red', linestyle='--', alpha=0.5,
               label=f'浏览高峰: {peak_hour}时')

    return fig


def plot_daily_behavior(df: pd.DataFrame) -> plt.Figure:
    """每日用户行为趋势"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    daily = df.groupby(['date', 'behavior_type']).size().unstack(fill_value=0)

    # 每日行为趋势
    for col, color in zip(daily.columns, COLORS[:4]):
        axes[0].plot(daily.index, daily[col], color=color, linewidth=1.5, label=col, alpha=0.8)
    axes[0].set_title('每日用户行为趋势', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)

    # 每日活跃用户数
    dau = df.groupby('date')['user_id'].nunique()
    axes[1].fill_between(range(len(dau)), dau.values, alpha=0.3, color=COLORS[0])
    axes[1].plot(range(len(dau)), dau.values, color=COLORS[0], linewidth=2)
    axes[1].set_title('每日活跃用户数 (DAU)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('日期序号')
    axes[1].set_ylabel('活跃用户数')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ============ 漏斗图 ============

def plot_funnel(funnel_df: pd.DataFrame) -> plt.Figure:
    """绘制转化漏斗"""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 6))

    values = funnel_df['用户数'].values
    labels = funnel_df['阶段'].values
    rates = funnel_df['阶段转化率'].values

    max_width = values[0]
    heights = [0.8] * len(values)
    colors_funnel = ['#2E86AB', '#F18F01', '#C73E1D']

    for i, (val, label, h, c) in enumerate(zip(values, labels, heights, colors_funnel)):
        width = val / max_width * 10
        left = (10 - width) / 2
        rect = FancyBboxPatch((left, i * 1.3), width, h,
                              boxstyle="round,pad=0.1", facecolor=c, alpha=0.85,
                              edgecolor='white', linewidth=2)
        ax.add_patch(rect)

        # 文字标注
        if i == 0:
            ax.text(5, i * 1.3 + h / 2, f'{label}\n{val:,} 人', ha='center',
                    va='center', fontsize=13, fontweight='bold', color='white')
        else:
            ax.text(5, i * 1.3 + h / 2,
                    f'{label}\n{val:,} 人  |  阶段转化率 {rates[i]}%',
                    ha='center', va='center', fontsize=13, fontweight='bold', color='white')

        # 转化率箭头
        if i > 0:
            ax.annotate(f'↓ {rates[i]}%', xy=(5, i * 1.3 + 0.3), xytext=(5, i * 1.3 - 0.2),
                        fontsize=11, ha='center', color='#555', fontweight='bold')

    ax.set_xlim(0, 10)
    ax.set_ylim(len(values) * 1.3, -0.3)
    ax.axis('off')
    ax.set_title('用户行为转化漏斗', fontsize=16, fontweight='bold', pad=20)

    return fig


# ============ RFM 可视化 ============

def plot_rfm_segments(rfm_df: pd.DataFrame) -> plt.Figure:
    """RFM 用户分层可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 分层饼图
    seg_counts = rfm_df['Segment'].value_counts()
    colors_seg = {'高价值用户': '#2E86AB', '重要用户': '#F18F01',
                  '一般用户': '#A23B72', '低价值用户': '#C73E1D'}
    pie_colors = [colors_seg.get(s, '#ccc') for s in seg_counts.index]
    axes[0].pie(seg_counts.values, labels=[f'{k}\n{v:,}' for k, v in seg_counts.items()],
                autopct='%1.1f%%', colors=pie_colors, explode=(0.05, 0, 0, 0))
    axes[0].set_title('用户分层分布', fontsize=13, fontweight='bold')

    # R vs F 散点
    axes[1].scatter(rfm_df['Recency'], rfm_df['Frequency'],
                    c=rfm_df['RFM_Score'], cmap='RdYlGn', alpha=0.5, s=10)
    axes[1].set_xlabel('Recency (天)')
    axes[1].set_ylabel('Frequency (购买商品数)')
    axes[1].set_title('R vs F 散点分布', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # 各层平均指标
    seg_avg = rfm_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean()
    seg_avg.plot(kind='bar', ax=axes[2], color=[COLORS[0], COLORS[2], COLORS[1]])
    axes[2].set_title('各层平均指标', fontsize=13, fontweight='bold')
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=45, ha='right')
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    return fig


# ============ 留存热力图 ============

def plot_retention_heatmap(cohorts_df: pd.DataFrame) -> plt.Figure:
    """留存率热力图"""
    retention_cols = ['day1_retention', 'day3_retention', 'day7_retention',
                      'day14_retention', 'day30_retention']

    data = cohorts_df[retention_cols].head(10)  # 最近 10 个群组
    data.index = [str(d) for d in cohorts_df.index[:10]]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                linewidths=0.5, cbar_kws={'label': '留存率 (%)'})
    ax.set_title('用户留存率热力图 (按首活日期群组)', fontsize=14, fontweight='bold')
    ax.set_xlabel('活跃间隔')
    ax.set_ylabel('首次活跃日期')

    return fig


# ============ 品类分析 ============

def plot_category_top10(category_stats: pd.DataFrame, column: str = 'conversion_rate') -> plt.Figure:
    """Top 10 品类分析"""
    fig, ax = plt.subplots(figsize=(12, 6))

    top10 = category_stats.head(10)
    colors_bar = plt.cm.RdYlGn(np.linspace(0.2, 0.8, 10))

    bars = ax.barh(range(len(top10)), top10[column].values, color=colors_bar, edgecolor='white')
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels([f'品类 {cid}' for cid in top10.index])
    ax.invert_yaxis()
    ax.set_xlabel(column)
    ax.set_title(f'Top 10 品类 - 按 {column} 排序', fontsize=14, fontweight='bold')

    for bar, val in zip(bars, top10[column].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=10)

    return fig


# ============ 用户聚类可视化 ============

def plot_cluster_2d(features_scaled: np.ndarray, labels: np.ndarray,
                    centers: np.ndarray = None) -> plt.Figure:
    """聚类结果 2D 可视化（PCA 降维后）"""
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))

    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1],
                         c=labels, cmap='tab10', alpha=0.5, s=10)

    if centers is not None:
        centers_2d = pca.transform(centers)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1],
                   c='red', marker='X', s=200, edgecolors='white', linewidth=1.5,
                   label='聚类中心')

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} 方差)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} 方差)')
    ax.set_title('用户聚类结果 (PCA 降维)', fontsize=14, fontweight='bold')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='聚类标签')

    return fig
