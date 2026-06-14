# 📦 电商供应链需求预测与库存优化系统

> **供应链智能决策系统** | 5种预测方法对比 · EOQ库存优化 · Streamlit交互看板

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 项目简介

针对电商SKU库存管理痛点，构建**从需求预测到库存决策的完整数据驱动方案**。

```
生成假数据 → 探索分析 → 5方法预测对比 → EOQ库存优化 → 交互看板
(模块1)     (模块2)     (模块3)         (模块4)        (模块5)
```

---

## 🚀 快速开始

```bash
# 1. 进入项目
cd supply-chain-forecast

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行全流程（按顺序）
python 01_data_generate.py       # 生成模拟数据 (~2秒)
python 02_eda_analysis.py        # EDA分析 (~5秒)
python 03_demand_forecast.py     # 5方法需求预测 (~30秒)
python 04_inventory_optimize.py  # 库存优化 (~5秒)

# 4. 启动看板
streamlit run 05_dashboard.py
# 或直接双击：一键启动看板.bat
```

浏览器访问 http://localhost:8501 即可看到交互式Dashboard。

---

## 🔮 核心功能

### 1. 需求预测 — 5种方法全方位对比

| 方法 | 类型 | MAPE |
|------|------|------|
| 移动平均(MA-7) | Baseline基准线 | ~5.8% |
| Holt-Winters | 指数平滑 | ~3.0% |
| SARIMA | 经典时序 | ~3.3% |
| **Facebook Prophet** | **大厂主流 🥇** | **~2.1%** |
| LSTM (PyTorch) | 深度学习 | ~3.7% |

> 以MAPE为核心指标自动选择最优模型（本项目冠军：**Prophet**）

### 2. 库存优化 — EOQ + 安全库存

| 指标 | 优化前 | 优化后 | 节省 |
|------|--------|--------|------|
| 每次订货量 | 350件 | **4,753件** | - |
| 年订货次数 | 645次 | **48次** | ↓597次 |
| 年总成本 | 705.9万 | **682.9万** | **省22.9万(3.2%)** |

### 3. 交互看板 (Streamlit)

4个页面：数据总览 / 需求预测 / 库存优化 / 图表库

---

## 📁 项目结构

```
supply-chain-forecast/
├── README.md                    ← 本文件 (GitHub首页展示)
├── requirements.txt             ← Python依赖清单
├── 一键启动看板.bat              ← 双击启动！
│
├── 01_data_generate.py          ← 模拟3年电商销售数据
├── 02_eda_analysis.py           ← 探索性分析 (6张图)
├── 03_demand_forecast.py        ← 5方法预测对比 ⭐
├── 04_inventory_optimize.py     ← EOQ+安全库存优化
├── 05_dashboard.py              ← Streamlit看板
│
├── data/                        ← 生成的数据
├── outputs/
│   ├── charts/                  ← 13张分析图表
│   ├── models/                  ← 模型文件
│   └── reports/                 ← 统计报表 (CSV)
└── docs/
    └── PROJECT_GUIDE.md         ← 零基础完整教学文档
```

---

## 📊 核心产出

- 📈 **13张专业图表** (PNG)
- 📋 **6份统计报表** (CSV，可用Excel打开)
- 🔮 **未来30天需求预测** (含95%置信区间)
- 📦 **EOQ最优订货策略** (年省22.9万元)
- 🌐 **Streamlit交互看板** (一键启动)

---

## 🛠️ 技术栈

`Python` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Statsmodels` `Prophet` `PyTorch` `Scikit-learn` `Streamlit`

---


---

## 👤 作者

**qikun-77** | 上海海事大学 · 工业工程与管理 · 数据分析/供应链方向

---

## 📝 License


