# ============================================================
# 模块02：探索性数据分析 (EDA)
# 产出：趋势图、季节性分解、促销效果对比、相关性矩阵
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，服务器/自动化友好
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# 0. 全局设置
# =====================================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("Set2")

DATA_PATH = 'data/ecommerce_demand.csv'
OUTPUT_DIR = 'outputs/charts'
SUMMARY_OUTPUT = 'outputs/reports/eda_summary.csv'

# =====================================================
# 1. 数据加载与总览
# =====================================================
def load_and_preview():
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds').sort_index()
    
    print("="*60)
    print(" 📊 EDA: 探索性数据分析")
    print("="*60)
    print(f"\n📅 时间范围: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"📏 数据条数: {len(df)}")
    print(f"\n📋 前5行:")
    print(df.head(), "\n")
    print(f"📈 缺失值统计:\n{df.isnull().sum()}\n")
    return df

# =====================================================
# 2. 绘制图表函数
# =====================================================
def save_fig(name, dpi=150):
    """统一保存图表"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 已保存: {path}")

def plot_demand_overview(df):
    """图1：3年日均销量总览 + 30日滚动均值"""
    fig, ax = plt.subplots(figsize=(14, 5))
    
    ax.plot(df.index, df['demand'], alpha=0.3, color='steelblue', linewidth=0.8, label='日销量')
    ax.plot(df.index, df['demand'].rolling(30).mean(), color='darkred', linewidth=1.8, label='30日滚动均值')
    
    ax.set_title('电商日销量总览（2023-2025）\nDaily Sales Overview with 30-Day MA', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('销量 (件)')
    ax.legend(loc='upper left')
    
    # 标注促销区域
    promo_periods = df[df['is_promo_date']].index
    for d in promo_periods[::5]:  # 每5个标一个
        ax.axvline(d, color='orange', alpha=0.3, linewidth=1)
    
    save_fig('01_demand_overview.png')

def plot_seasonal_decomposition(df):
    """图2：时间序列分解（趋势 + 周季节性 + 残差）"""
    decomposition = seasonal_decompose(df['demand'], model='additive', period=7)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    
    axes[0].plot(decomposition.observed, linewidth=0.8, color='steelblue')
    axes[0].set_title('Observed 原始序列', fontweight='bold')
    
    axes[1].plot(decomposition.trend, linewidth=1.2, color='darkred')
    axes[1].set_title('Trend 趋势成分', fontweight='bold')
    
    axes[2].plot(decomposition.seasonal, linewidth=0.6, color='green')
    axes[2].set_title('Seasonal 季节性成分 (周期=7天)', fontweight='bold')
    
    axes[3].plot(decomposition.resid, linewidth=0.5, color='gray')
    axes[3].set_title('Residual 残差成分', fontweight='bold')
    
    fig.suptitle('时间序列分解 STL Decomposition', fontsize=15, fontweight='bold')
    save_fig('02_seasonal_decomposition.png')

def plot_weekly_pattern(df):
    """图3：周内日销量分布（箱线图）"""
    df_plot = df.copy()
    df_plot['dow_name'] = df_plot.index.day_name()
    order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_plot, x='dow_name', y='demand', order=order, 
                palette='Blues', ax=ax)
    ax.set_title('周内日销量分布 Distribution by Day of Week', fontsize=13, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('日销量 (件)')
    save_fig('03_weekly_pattern.png')

def plot_monthly_heatmap(df):
    """图4：月-年热力图"""
    df_plot = df.copy()
    df_plot['year'] = df_plot.index.year
    df_plot['month'] = df_plot.index.month
    pivot = df_plot.pivot_table(values='demand', index='month', columns='year', aggfunc='mean').round(0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': '平均日销量'})
    ax.set_title('月均销量热力图 Monthly Average Demand Heatmap', fontsize=13, fontweight='bold')
    ax.set_xlabel('年份')
    ax.set_ylabel('月份')
    save_fig('04_monthly_heatmap.png')

def plot_promo_effect(df):
    """图5：促销vs非促销日销量对比"""
    promo = df[df['is_promo_date']]['demand']
    non_promo = df[~df['is_promo_date']]['demand']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 直方图对比
    axes[0].hist(non_promo, bins=40, alpha=0.6, label=f'非促销 (μ={non_promo.mean():.0f})', color='steelblue')
    axes[0].hist(promo, bins=20, alpha=0.7, label=f'促销日 (μ={promo.mean():.0f})', color='darkorange')
    axes[0].set_title('促销 vs 非促销 销量分布')
    axes[0].legend()
    axes[0].set_xlabel('日销量 (件)')
    axes[0].set_ylabel('频次')
    
    # 箱线图
    data = [non_promo.values, promo.values]
    bp = axes[1].boxplot(data, labels=['非促销日', '促销日'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('darkorange')
    axes[1].set_title('促销 vs 非促销 箱线图对比')
    axes[1].set_ylabel('日销量 (件)')
    
    fig.suptitle('促销效果分析 Promo Effect Analysis', fontsize=14, fontweight='bold')
    save_fig('05_promo_effect.png')

def plot_demand_distribution(df):
    """图6：需求分布直方图 + Q-Q图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    from scipy import stats
    
    axes[0].hist(df['demand'], bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    mu, std = df['demand'].mean(), df['demand'].std()
    x = np.linspace(df['demand'].min(), df['demand'].max(), 100)
    axes[0].plot(x, stats.norm.pdf(x, mu, std), 'r--', linewidth=2, label=f'N({mu:.0f}, {std:.0f}²)')
    axes[0].set_title('需求分布 & 正态拟合')
    axes[0].legend()
    axes[0].set_xlabel('日销量 (件)')
    
    stats.probplot(df['demand'], dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (正态性检验)')
    axes[1].get_lines()[0].set_markerfacecolor('steelblue')
    axes[1].get_lines()[1].set_color('darkred')
    
    fig.suptitle('需求分布特征 Demand Distribution Analysis', fontsize=14, fontweight='bold')
    save_fig('06_demand_distribution.png')

# =====================================================
# 3. 数值统计摘要
# =====================================================
def generate_summary_stats(df):
    """生成数值统计摘要，输出CSV"""
    stats_dict = {
        'metric': ['总天数','日均需求','需求标准差','最小日需求','最大日需求',
                   '促销天数','非促销日均需求','促销日均需求','促销提升幅度%',
                   '缺货天数','缺货率%','变异系数'],
        'value': [
            len(df),
            round(df['demand'].mean(), 1),
            round(df['demand'].std(), 1),
            df['demand'].min(),
            df['demand'].max(),
            df['is_promo_date'].sum(),
            round(df[~df['is_promo_date']]['demand'].mean(), 1),
            round(df[df['is_promo_date']]['demand'].mean(), 1),
            round((df[df['is_promo_date']]['demand'].mean() / df[~df['is_promo_date']]['demand'].mean() - 1) * 100, 1),
            df['stockout'].sum(),
            round(df['stockout'].mean() * 100, 2),
            round(df['demand'].std() / df['demand'].mean(), 3),
        ]
    }
    summary = pd.DataFrame(stats_dict)
    os.makedirs(os.path.dirname(SUMMARY_OUTPUT), exist_ok=True)
    summary.to_csv(SUMMARY_OUTPUT, index=False, encoding='utf-8-sig')
    print(f"\n📊 统计摘要:\n{summary.to_string(index=False)}")
    print(f"\n✅ 摘要已保存至: {SUMMARY_OUTPUT}")
    return summary

# =====================================================
# 4. 主函数
# =====================================================
def main():
    df = load_and_preview()
    
    print("\n📈 正在生成分析图表...")
    plot_demand_overview(df)          # 图1
    plot_seasonal_decomposition(df)    # 图2
    plot_weekly_pattern(df)           # 图3
    plot_monthly_heatmap(df)          # 图4
    plot_promo_effect(df)             # 图5
    plot_demand_distribution(df)      # 图6
    
    summary = generate_summary_stats(df)
    
    print(f"\n{'='*60}")
    print(f" ✅ EDA分析完成！共生成 6 张图表 + 1 份统计摘要")
    print(f" 📁 输出目录: outputs/")
    print(f"{'='*60}")
    return df

if __name__ == '__main__':
    df = main()