# ============================================================
# 模块01：模拟电商销售数据生成
# 生成3年日粒度电商销售数据，包含：趋势、季节性、促销效应、噪声
# ============================================================

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 0. 参数配置（面试时可说明"所有参数可调，适应不同品类"）
# =====================================================
CONFIG = {
    'start_date': '2023-01-01',      # 数据起始日
    'end_date': '2025-12-31',        # 数据截止日（3年）
    'base_demand': 500,              # 基础日销量
    'trend_coef': 0.15,              # 日增长趋势系数
    'weekly_seasonality': [0.9, 1.05, 1.0, 0.95, 0.98, 1.15, 1.25],  # 周日~周六
    'yearly_amplitude': 100,          # 年季节性振幅
    'noise_std': 30,                  # 随机噪声标准差
    'promo_dates': [],                # 促销日期列表（动态生成）
    'promo_effect': 1.4,              # 促销提升倍数
    'stockout_prob': 0.02,            # 缺货概率
    'output_dir': 'data',
}

# =====================================================
# 1. 生成日期序列并构造季节性
# =====================================================
def generate_dates(start, end):
    """生成日期范围"""
    return pd.date_range(start=start, end=end, freq='D')

def compute_features(dates):
    """从日期提取特征（核心特征工程）"""
    df = pd.DataFrame({'ds': dates})
    df['dayofweek'] = df['ds'].dt.dayofweek  # 0=周一, 6=周日
    df['month'] = df['ds'].dt.month
    df['day_of_year'] = df['ds'].dt.dayofyear
    df['year'] = df['ds'].dt.year
    df['day_index'] = range(len(df))  # 0,1,2,... 用于趋势
    return df

# =====================================================
# 2. 模拟需求公式
# =====================================================
def simulate_demand(df, cfg):
    """
    需求模型：
    demand = base + trend + yearly_seasonal + weekly_seasonal + promo + noise
    约束：demand >= 0
    """
    n = len(df)
    np.random.seed(42)
    
    # --- 趋势 ---
    trend = cfg['base_demand'] + cfg['trend_coef'] * df['day_index']
    
    # --- 年季节性（正弦波）---
    yearly_seasonal = cfg['yearly_amplitude'] * np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    
    # --- 周季节性 ---
    weekly_seasonal = np.array([cfg['weekly_seasonality'][d] for d in df['dayofweek']])
    
    # --- 促销效应 ---
    promo_dates = _generate_promo_dates(df, cfg)
    promo_effect = np.ones(n)
    for p_date in promo_dates:
        promo_effect[df['ds'] == p_date] = cfg['promo_effect']
    
    # --- 噪声 ---
    noise = np.random.normal(0, cfg['noise_std'], n)
    
    # --- 合成需求 ---
    demand = trend * weekly_seasonal + yearly_seasonal
    demand = demand * promo_effect + noise
    demand = np.maximum(demand, 10)  # 最低销量10件
    
    return np.round(demand).astype(int), promo_dates

def _generate_promo_dates(df, cfg):
    """生成促销日期：每月1号和双11、618等大促"""
    promo_list = []
    for year in df['year'].unique():
        # 每月1日促销
        for m in range(1, 13):
            d = pd.Timestamp(f'{year}-{m:02d}-01')
            if d in df['ds'].values:
                promo_list.append(d)
        # 618
        for day in [16, 17, 18]:
            d = pd.Timestamp(f'{year}-06-{day}')
            if d in df['ds'].values:
                promo_list.append(d)
        # 双11
        for day in [10, 11, 12]:
            d = pd.Timestamp(f'{year}-11-{day}')
            if d in df['ds'].values:
                promo_list.append(d)
        # 双12
        d = pd.Timestamp(f'{year}-12-12')
        if d in df['ds'].values:
            promo_list.append(d)
    return sorted(list(set(promo_list)))

# =====================================================
# 3. 添加库存/供应链相关字段
# =====================================================
def add_supply_fields(df, cfg):
    """
    供应链相关字段：
    - lead_time: 采购提前期（天）
    - cost_per_unit: 单位采购成本
    - selling_price: 售价
    - stockout: 是否缺货（0/1）
    """
    np.random.seed(123)
    n = len(df)
    
    df['lead_time'] = np.random.choice([3, 5, 7], size=n, p=[0.5, 0.35, 0.15])
    df['cost_per_unit'] = np.random.normal(30, 5, n).round(2)
    df['selling_price'] = df['cost_per_unit'] * np.random.uniform(1.5, 2.5, n).round(2)
    df['is_promo'] = 0
    df.loc[df['is_promo_date'], 'is_promo'] = 1
    df['stockout'] = np.random.binomial(1, cfg['stockout_prob'], n)
    return df

# =====================================================
# 4. 主函数
# =====================================================
def main():
    print("="*60)
    print(" 🏭 电商供应链数据生成器 v1.0")
    print("="*60)
    
    # 生成日期
    dates = generate_dates(CONFIG['start_date'], CONFIG['end_date'])
    df = compute_features(dates)
    print(f"📅 日期范围: {CONFIG['start_date']} ~ {CONFIG['end_date']}")
    print(f"📊 数据条数: {len(df)} 天")
    
    # 模拟需求
    demand, promo_dates = simulate_demand(df, CONFIG)
    df['demand'] = demand
    df['is_promo_date'] = df['ds'].isin(promo_dates)
    
    # 添加供应链字段
    df = add_supply_fields(df, CONFIG)
    
    # 保存
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    output_path = os.path.join(CONFIG['output_dir'], 'ecommerce_demand.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    # 统计摘要
    print(f"\n📦 数据统计摘要:")
    print(f"   日均需求量: {df['demand'].mean():.1f} 件")
    print(f"   最大日需求: {df['demand'].max()} 件")
    print(f"   最小日需求: {df['demand'].min()} 件")
    print(f"   促销天数: {df['is_promo_date'].sum()} 天")
    print(f"   缺货率: {df['stockout'].mean()*100:.2f}%")
    print(f"\n✅ 数据已保存至: {output_path}")
    print(f"   列名: {list(df.columns)}")
    print("="*60)
    
    return df

if __name__ == '__main__':
    df = main()