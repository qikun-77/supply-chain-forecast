# ============================================================
# 模块05：Streamlit 交互式供应链看板
# 启动命令: streamlit run 05_dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime, timedelta

# =====================================================
# 页面配置
# =====================================================
st.set_page_config(
    page_title="电商供应链智能决策系统",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; text-align: center; }
    .metric-card { background: #f0f2f6; border-radius: 10px; padding: 15px; text-align: center; }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1f77b4; }
    .metric-label { font-size: 0.9rem; color: #555; }
    .danger { color: #d62728; }
    .success { color: #2ca02c; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# 数据加载
# =====================================================
@st.cache_data
def load_all_data():
    """加载所有中间结果"""
    data = {}
    
    # 原始数据
    df = pd.read_csv('data/ecommerce_demand.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds').sort_index()
    data['df'] = df
    
    # EDA摘要
    if os.path.exists('outputs/reports/eda_summary.csv'):
        data['eda_summary'] = pd.read_csv('outputs/reports/eda_summary.csv')
    
    # 模型对比
    if os.path.exists('outputs/reports/model_comparison.csv'):
        data['model_comparison'] = pd.read_csv('outputs/reports/model_comparison.csv')
    
    # 未来预测
    if os.path.exists('outputs/reports/future_30day_forecast.csv'):
        fc = pd.read_csv('outputs/reports/future_30day_forecast.csv')
        fc['date'] = pd.to_datetime(fc['date'])
        data['future_forecast'] = fc
    
    # 库存优化结果
    if os.path.exists('outputs/reports/eoq_result.csv'):
        data['eoq'] = pd.read_csv('outputs/reports/eoq_result.csv')
    if os.path.exists('outputs/reports/safety_stock_result.csv'):
        data['safety'] = pd.read_csv('outputs/reports/safety_stock_result.csv')
    if os.path.exists('outputs/reports/cost_comparison.csv'):
        data['cost'] = pd.read_csv('outputs/reports/cost_comparison.csv')
    
    return data

data = load_all_data()
df = data.get('df')

# =====================================================
# 侧边栏
# =====================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/supply-chain.png", width=80)
    st.title("📦 电商供应链")
    st.title("   智能决策系统")
    st.markdown("---")
    
    st.markdown("### 🧭 导航")
    page = st.radio("", ["📊 数据总览", "🔮 需求预测", "📦 库存优化", "📈 图表库"])
    
    st.markdown("---")
    st.markdown(f"🕐 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    st.markdown("© 2026 Supply Chain AI")

# =====================================================
# 页面1：数据总览
# =====================================================
if page == "📊 数据总览":
    st.markdown('<p class="main-header">📊 数据总览 Dashboard</p>', unsafe_allow_html=True)
    
    # KPI卡片
    if df is not None:
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📅 数据天数", f"{len(df):,}", delta="3年日粒度")
        with col2:
            st.metric("📦 日均销量", f"{df['demand'].mean():.0f} 件")
        with col3:
            st.metric("📈 最高日销", f"{df['demand'].max():.0f} 件")
        with col4:
            promo_days = df['is_promo_date'].sum()
            st.metric("🎯 促销天数", f"{promo_days}", delta=f"{promo_days/len(df)*100:.1f}%占比")
        with col5:
            stockout_rate = df['stockout'].mean() * 100
            st.metric("⚠️ 缺货率", f"{stockout_rate:.2f}%", delta_color="inverse")
    
    st.markdown("---")
    
    # 时间序列
    st.subheader("📈 历史销量趋势 (最近365天)")
    if df is not None:
        recent = df.iloc[-365:]
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(recent.index, recent['demand'], alpha=0.4, color='steelblue', linewidth=0.8, label='日销量')
        ax.plot(recent.index, recent['demand'].rolling(30).mean(), color='darkred', linewidth=2, label='30日均值')
        ax.set_xlabel('日期')
        ax.set_ylabel('销量 (件)')
        ax.legend()
        ax.set_title('最近365天销量趋势')
        st.pyplot(fig)
    
    # 统计摘要表
    if data.get('eda_summary') is not None:
        st.subheader("📋 统计摘要")
        st.dataframe(data['eda_summary'], use_container_width=True, hide_index=True)
    
    # 数据预览
    if df is not None:
        st.subheader("🔍 原始数据预览 (最近30条)")
        st.dataframe(df.tail(30), use_container_width=True)

# =====================================================
# 页面2：需求预测
# =====================================================
elif page == "🔮 需求预测":
    st.markdown('<p class="main-header">🔮 需求预测分析</p>', unsafe_allow_html=True)
    
    # 模型对比
    if data.get('model_comparison') is not None:
        st.subheader("🏆 五模型效果对比")
        
        # 用bar chart展示
        model_df = data['model_comparison']
        best_model = model_df.iloc[0]['模型']
        
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(model_df, use_container_width=True, hide_index=True)
        with col2:
            st.success(f"🥇 最优模型: **{best_model}**")
            st.info(f"📊 MAPE: **{model_df.iloc[0]['MAPE(%)']}%**")
            st.info(f"📊 RMSE: **{model_df.iloc[0]['RMSE']}**")
        
        # MAPE对比柱状图
        st.subheader("📊 MAPE 对比")
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.barh(model_df['模型'], model_df['MAPE(%)'], color=['gold' if m==best_model else 'steelblue' for m in model_df['模型']])
        ax.set_xlabel('MAPE (%)')
        ax.set_title('各模型MAPE对比 (越低越好)')
        for bar, v in zip(bars, model_df['MAPE(%)']):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, f'{v:.1f}%', va='center')
        ax.invert_yaxis()
        st.pyplot(fig)
    
    # 未来预测
    if data.get('future_forecast') is not None:
        st.subheader(f"🔮 未来30天需求预测")
        fc = data['future_forecast']
        
        # 预测表格
        with st.expander("📋 查看详细预测数据", expanded=False):
            st.dataframe(fc, use_container_width=True, hide_index=True)
        
        # 预测折线图
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(fc['date'], fc['forecast'], 'o-', color='darkred', linewidth=2, label='预测值')
        ax.fill_between(fc['date'], fc['lower_bound'], fc['upper_bound'], alpha=0.3, color='red', label='95%置信区间')
        ax.set_title('未来30天需求预测')
        ax.set_xlabel('日期')
        ax.set_ylabel('预测销量 (件)')
        ax.legend()
        ax.tick_params(axis='x', rotation=30)
        st.pyplot(fig)
        
        # 预测摘要
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 预测平均日销", f"{fc['forecast'].mean():.0f} 件")
        with col2:
            st.metric("📈 预测最高日销", f"{fc['forecast'].max():.0f} 件")
        with col3:
            st.metric("📉 预测最低日销", f"{fc['forecast'].min():.0f} 件")
    
    # 展示图表文件
    st.subheader("🖼️ 预测分析图表")
    chart_files = {
        '07_forecast_comparison.png': '五模型预测对比',
        '08_future_forecast.png': '未来30天预测',
        '09_residual_diagnostics.png': '残差诊断分析',
    }
    for fname, title in chart_files.items():
        fpath = f'outputs/charts/{fname}'
        if os.path.exists(fpath):
            with st.expander(f"📊 {title}", expanded=False):
                st.image(fpath, use_container_width=True)

# =====================================================
# 页面3：库存优化
# =====================================================
elif page == "📦 库存优化":
    st.markdown('<p class="main-header">📦 库存优化决策</p>', unsafe_allow_html=True)
    
    # EOQ结果
    if data.get('eoq') is not None:
        eoq = data['eoq']
        st.subheader("📐 EOQ经济订货批量")
        cols = st.columns(4)
        metrics_list = list(eoq.columns)
        for i, col_name in enumerate(metrics_list):
            with cols[i % 4]:
                val = eoq[col_name].values[0]
                if isinstance(val, float):
                    st.metric(col_name, f"{val:,.1f}" if '次数' in col_name or '间隔' in col_name else f"{val:,.0f}")
                else:
                    st.metric(col_name, val)
    
    if data.get('safety') is not None:
        safety = data['safety']
        st.subheader("🛡️ 安全库存 & 再订货点")
        cols = st.columns(5)
        safety_metrics = list(safety.columns)
        for i, col_name in enumerate(safety_metrics):
            with cols[i % 5]:
                st.metric(col_name, safety[col_name].values[0])
    
    # 成本对比
    if data.get('cost') is not None:
        cost = data['cost']
        st.subheader("💰 成本对比：当前策略 vs EOQ优化")
        st.dataframe(cost, use_container_width=True, hide_index=True)
        
        # 节省金额高亮
        total_savings = cost[cost['项目'] == '年总成本 (元)']['节省 (元)'].values[0]
        savings_pct = cost[cost['项目'] == '年总成本 (元)']['节省%'].values[0]
        st.success(f"🎉 优化后可**年节省 {total_savings:,.0f} 元** (降低成本 {savings_pct}%)")
    
    # 库存图表
    st.subheader("🖼️ 库存优化图表")
    inventory_charts = {
        '10_eoq_sensitivity.png': 'EOQ敏感性分析',
        '11_inventory_strategy.png': '库存策略示意图',
        '12_cost_waterfall.png': '成本瀑布对比',
        '13_inventory_simulation.png': '90天库存模拟',
    }
    for fname, title in inventory_charts.items():
        fpath = f'outputs/charts/{fname}'
        if os.path.exists(fpath):
            with st.expander(f"📊 {title}", expanded=('模拟' in title)):
                st.image(fpath, use_container_width=True)

# =====================================================
# 页面4：图表库
# =====================================================
elif page == "📈 图表库":
    st.markdown('<p class="main-header">📈 完整图表库</p>', unsafe_allow_html=True)
    
    all_charts = {
        'EDA分析': [
            ('01_demand_overview.png', '📈 需求总览'),
            ('02_seasonal_decomposition.png', '🔍 季节性分解'),
            ('03_weekly_pattern.png', '📅 周内分布'),
            ('04_monthly_heatmap.png', '🔥 月-年热力图'),
            ('05_promo_effect.png', '🎯 促销效果'),
            ('06_demand_distribution.png', '📊 需求分布'),
        ],
        '需求预测': [
            ('07_forecast_comparison.png', '🏆 预测对比'),
            ('08_future_forecast.png', '🔮 未来预测'),
            ('09_residual_diagnostics.png', '🔬 残差诊断'),
        ],
        '库存优化': [
            ('10_eoq_sensitivity.png', '📐 EOQ敏感性'),
            ('11_inventory_strategy.png', '🛡️ 库存策略'),
            ('12_cost_waterfall.png', '💰 成本瀑布'),
            ('13_inventory_simulation.png', '🎮 库存模拟'),
        ],
    }
    
    for section, charts in all_charts.items():
        st.subheader(f"--- {section} ---")
        available = [c for c in charts if os.path.exists(f'outputs/charts/{c[0]}')]
        if not available:
            st.info(f"请先运行对应模块生成图表")
            continue
        
        cols = st.columns(min(len(available), 2))
        for i, (fname, title) in enumerate(available):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.image(f'outputs/charts/{fname}', use_container_width=True)

# =====================================================
# 底部
# =====================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>"
    "📦 电商供应链需求预测与库存优化系统 | "
    "数据驱动决策 · 运筹优化 · 机器学习"
    "</p>",
    unsafe_allow_html=True
)