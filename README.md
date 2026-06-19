# 电商用户行为分析

## 项目概述

基于真实电商平台用户行为日志，完成从**数据清洗 → 探索性分析 → 用户分层 → 转化漏斗 → 建模预测**的全链路数据分析项目。

## 数据来源

阿里天池公开数据集 — Taobao User Behavior（约 1 亿条用户行为记录）

字段说明：
| 字段 | 含义 |
|------|------|
| user_id | 用户 ID |
| item_id | 商品 ID |
| category_id | 品类 ID |
| behavior_type | 行为类型（pv=浏览, buy=购买, cart=加购, fav=收藏）|
| timestamp | 时间戳 |

## 分析框架

```
数据清洗 → EDA（用户/商品/时间维度）→ 漏斗分析 → 用户分层(RFM) → 关联规则 → 结论与建议
```

## 核心产出

- **用户转化漏斗**：从浏览到购买的完整转化路径及流失点
- **用户分层模型**：基于 RFM 的用户价值分群
- **用户行为画像**：活跃时段、品类偏好、购买决策周期
- **商品关联规则**：支持度/置信度/提升度分析
- **可落地的业务建议**

## 技术栈

- Python 3.8+
- Pandas / NumPy（数据处理）
- Matplotlib / Seaborn（可视化）
- Scikit-learn（用户聚类）
- Mlxtend（关联规则）

## 项目结构

```
├── README.md
├── requirements.txt
├── data/                   # 数据集（需自行下载）
├── notebooks/
│   └── ecommerce_analysis.ipynb   # 完整分析流程
├── src/
│   ├── data_loader.py      # 数据加载与预处理
│   ├── analysis.py         # 分析函数
│   └── visualize.py        # 可视化函数
└── reports/
    └── analysis_report.md  # 分析报告
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 下载数据
# https://tianchi.aliyun.com/dataset/649

# 将数据放入 data/ 目录后运行 notebook
jupyter notebook notebooks/ecommerce_analysis.ipynb
```

## 作者

- 数据科学与大数据技术专业
- 熟练 Python / SQL / Pandas / Scikit-learn
- 邮箱 2143492748@qq.com
