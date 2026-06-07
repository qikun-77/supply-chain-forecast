# ============================================================
# 模块04：库存优化 - EOQ + 安全库存 + (s,Q)补货策略
# 产出：最优订货量、再订货点、安全库存、成本对比
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, json
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# =====================================================
# 0. 全局配置
# =====================================================
DATA_PATH = 'data/ecommerce_demand.csv'
FORECAST_PATH = 'outputs/reports/future_30day_forecast.csv'
OUTPUT_DIR = 'outputs/charts'
REPORT_DIR = 'outputs/reports'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# 供应链参数（实际业务中从ERP获取）
SUPPLY_CONFIG = {
    # --- EOQ参数 ---
    'annual_demand': None,           # 年需求量（从数据计算）
    'ordering_cost': 500,            # 每次订货固定成本（元）
    'holding_cost_per_unit': 10,     # 单位年持有成本（元/件/年）
    'cost_per_unit': 30,             # 单位采购成本（元）
    
    # --- 安全库存参数 ---
    'service_level': 0.95,           # 服务水平 (z=1.645)
    'lead_time_days': 7,             # 采购提前期（天）
    'lead_time_std': 1.5,            # 提前期标准差（天）
    
    # --- 当前策略（用于对比）---
    'current_order_qty': 350,        # 当前每次订350件
    'current_reorder_point': 100,    # 当前库存低于100就补货
}

# =====================================================
# 1. 数据加载与计算
# =====================================================
def load_and_calc():
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'])
    
    # 日均需求 & 标准差
    daily_demand_mean = df['demand'].mean()
    daily_demand_std = df['demand'].std()
    
    # 年需求
    annual_demand = daily_demand_mean * 365
    SUPPLY_CONFIG['annual_demand'] = annual_demand
    
    # 加载未来预测
    if os.path.exists(FORECAST_PATH):
        forecast = pd.read_csv(FORECAST_PATH)
        future_daily_mean = forecast['forecast'].mean()
    else:
        future_daily_mean = daily_demand_mean
    
    return df, daily_demand_mean, daily_demand_std, annual_demand, future_daily_mean

# =====================================================
# 2. EOQ模型
# =====================================================
def calc_eoq():
    """
    EOQ = sqrt(2 * D * S / H)
    D: 年需求量, S: 订货成本, H: 单位年持有成本
    """
    D = SUPPLY_CONFIG['annual_demand']
    S = SUPPLY_CONFIG['ordering_cost']
    H = SUPPLY_CONFIG['holding_cost_per_unit']
    
    EOQ = np.sqrt(2 * D * S / H)
    
    # 年订货次数
    N = D / EOQ
    # 年总成本
    total_cost = (D / EOQ) * S + (EOQ / 2) * H + D * SUPPLY_CONFIG['cost_per_unit']
    ordering_cost_total = (D / EOQ) * S
    holding_cost_total = (EOQ / 2) * H
    purchase_cost = D * SUPPLY_CONFIG['cost_per_unit']
    
    return {
        'EOQ (件)': round(EOQ, 0),
        '年订货次数': round(N, 1),
        '订货间隔 (天)': round(365 / N, 1),
        '年订货成本 (元)': round(ordering_cost_total, 0),
        '年持有成本 (元)': round(holding_cost_total, 0),
        '年采购成本 (元)': round(purchase_cost, 0),
        '年总成本 (元)': round(total_cost, 0),
    }

# =====================================================
# 3. 安全库存 & 再订货点
# =====================================================
def calc_safety_stock(daily_demand_mean, daily_demand_std):
    """
    安全库存 SS = z * σ_d * sqrt(L)
    z: 服务水平对应的安全系数
    σ_d: 日需求标准差
    L: 提前期
    """
    from scipy.stats import norm
    
    z = norm.ppf(SUPPLY_CONFIG['service_level'])
    L = SUPPLY_CONFIG['lead_time_days']
    
    # 安全库存
    SS = z * daily_demand_std * np.sqrt(L)
    
    # 再订货点 ROP = d * L + SS
    ROP = daily_demand_mean * L + SS
    
    return {
        '服务水平': f"{SUPPLY_CONFIG['service_level']*100:.0f}%",
        'z值': round(z, 4),
        '提前期 (天)': L,
        '安全库存 (件)': round(SS, 0),
        '再订货点 ROP (件)': round(ROP, 0),
    }

# =====================================================
# 4. 成本对比分析
# =====================================================
def calc_cost_comparison(eoq_result, safety_result, annual_demand, daily_demand_mean):
    """EOQ策略 vs 当前策略成本对比"""
    S = SUPPLY_CONFIG['ordering_cost']
    H = SUPPLY_CONFIG['holding_cost_per_unit']
    C = SUPPLY_CONFIG['cost_per_unit']
    
    # EOQ策略
    eoq_qty = eoq_result['EOQ (件)']
    eoq_ordering = (annual_demand / eoq_qty) * S
    eoq_holding = (eoq_qty / 2 + safety_result['安全库存 (件)']) * H
    eoq_purchase = annual_demand * C
    eoq_total = eoq_ordering + eoq_holding + eoq_purchase
    
    # 当前策略
    cur_qty = SUPPLY_CONFIG['current_order_qty']
    cur_rop = SUPPLY_CONFIG['current_reorder_point']
    cur_ss = cur_rop - daily_demand_mean * SUPPLY_CONFIG['lead_time_days']
    cur_ordering = (annual_demand / cur_qty) * S
    cur_holding = (cur_qty / 2 + cur_ss) * H
    cur_purchase = annual_demand * C
    cur_total = cur_ordering + cur_holding + cur_purchase
    
    savings = cur_total - eoq_total
    
    comparison = pd.DataFrame({
        '项目': ['订货量 (件)', '年订货次数', '年订货成本 (元)', '年持有成本 (元)', 
                '年采购成本 (元)', '年总成本 (元)', '安全库存 (件)'],
        'EOQ策略': [
            round(eoq_qty, 0),
            round(annual_demand / eoq_qty, 1),
            round(eoq_ordering, 0),
            round(eoq_holding, 0),
            round(eoq_purchase, 0),
            round(eoq_total, 0),
            safety_result['安全库存 (件)'],
        ],
        '当前策略': [
            cur_qty,
            round(annual_demand / cur_qty, 1),
            round(cur_ordering, 0),
            round(cur_holding, 0),
            round(cur_purchase, 0),
            round(cur_total, 0),
            round(cur_ss, 0),
        ],
    })
    
    comparison['节省 (元)'] = comparison['当前策略'] - comparison['EOQ策略']
    comparison['节省%'] = round((comparison['节省 (元)'] / comparison['当前策略']) * 100, 1)
    
    return comparison, savings

# =====================================================
# 5. 可视化
# =====================================================
def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 已保存: {path}")

def plot_eoq_sensitivity():
    """EOQ敏感性分析：订货成本 vs 持有成本 对EOQ的影响"""
    D = SUPPLY_CONFIG['annual_demand']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 订货成本敏感性
    S_range = np.linspace(100, 1500, 50)
    eoq_s = np.sqrt(2 * D * S_range / SUPPLY_CONFIG['holding_cost_per_unit'])
    axes[0].plot(S_range, eoq_s, linewidth=2, color='steelblue')
    axes[0].axvline(SUPPLY_CONFIG['ordering_cost'], color='red', linestyle='--', 
                    label=f'当前S={SUPPLY_CONFIG["ordering_cost"]}')
    axes[0].set_xlabel('订货成本 S (元/次)')
    axes[0].set_ylabel('EOQ (件)')
    axes[0].set_title('订货成本对EOQ的影响')
    axes[0].legend()
    
    # 持有成本敏感性
    H_range = np.linspace(2, 30, 50)
    eoq_h = np.sqrt(2 * D * SUPPLY_CONFIG['ordering_cost'] / H_range)
    axes[1].plot(H_range, eoq_h, linewidth=2, color='darkorange')
    axes[1].axvline(SUPPLY_CONFIG['holding_cost_per_unit'], color='red', linestyle='--',
                    label=f'当前H={SUPPLY_CONFIG["holding_cost_per_unit"]}')
    axes[1].set_xlabel('持有成本 H (元/件/年)')
    axes[1].set_ylabel('EOQ (件)')
    axes[1].set_title('持有成本对EOQ的影响')
    axes[1].legend()
    
    fig.suptitle('EOQ敏感性分析 EOQ Sensitivity Analysis', fontsize=14, fontweight='bold')
    save_fig('10_eoq_sensitivity.png')

def plot_inventory_strategy(daily_demand_mean, daily_demand_std, safety_result):
    """库存策略示意图：再订货点 + 安全库存"""
    from scipy.stats import norm
    
    L = SUPPLY_CONFIG['lead_time_days']
    lead_time_demand_mean = daily_demand_mean * L
    lead_time_demand_std = daily_demand_std * np.sqrt(L)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.linspace(lead_time_demand_mean - 4*lead_time_demand_std, 
                    lead_time_demand_mean + 4*lead_time_demand_std, 200)
    y = norm.pdf(x, lead_time_demand_mean, lead_time_demand_std)
    
    ax.plot(x, y, linewidth=2, color='steelblue')
    ax.fill_between(x, 0, y, alpha=0.3, color='steelblue')
    
    # 再订货点
    ROP = safety_result['再订货点 ROP (件)']
    ax.axvline(ROP, color='darkred', linestyle='--', linewidth=2, 
               label=f'再订货点 ROP = {ROP:.0f} 件')
    ax.axvline(lead_time_demand_mean, color='gray', linestyle=':', linewidth=1.5,
               label=f'LT期望需求 = {lead_time_demand_mean:.0f} 件')
    
    # 安全库存区域
    x_ss = np.linspace(lead_time_demand_mean, ROP, 100)
    y_ss = norm.pdf(x_ss, lead_time_demand_mean, lead_time_demand_std)
    ax.fill_between(x_ss, 0, y_ss, alpha=0.4, color='orange', label=f'安全库存 SS = {safety_result["安全库存 (件)"]:.0f}')
    
    ax.set_title('库存策略：再订货点与安全库存\n(s,Q) Inventory Policy - Reorder Point & Safety Stock', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('提前期内需求量 (件)')
    ax.set_ylabel('概率密度')
    ax.legend(loc='upper right')
    
    save_fig('11_inventory_strategy.png')

def plot_cost_waterfall(eoq_result, safety_result, annual_demand, daily_demand_mean):
    """成本瀑布图"""
    S = SUPPLY_CONFIG['ordering_cost']
    H = SUPPLY_CONFIG['holding_cost_per_unit']
    C = SUPPLY_CONFIG['cost_per_unit']
    
    eoq_qty = eoq_result['EOQ (件)']
    eoq_ordering = (annual_demand / eoq_qty) * S
    eoq_holding = (eoq_qty / 2 + safety_result['安全库存 (件)']) * H
    eoq_purchase = annual_demand * C
    
    cur_qty = SUPPLY_CONFIG['current_order_qty']
    cur_rop = SUPPLY_CONFIG['current_reorder_point']
    cur_ss = cur_rop - daily_demand_mean * SUPPLY_CONFIG['lead_time_days']
    cur_ordering = (annual_demand / cur_qty) * S
    cur_holding = (cur_qty / 2 + cur_ss) * H
    cur_purchase = annual_demand * C
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    categories = ['采购成本', '订货成本', '持有成本\n(含安全库存)']
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [cur_purchase, cur_ordering, cur_holding], width, 
                   label='当前策略', color='gray', alpha=0.8)
    bars2 = ax.bar(x + width/2, [eoq_purchase, eoq_ordering, eoq_holding], width,
                   label='EOQ优化策略', color='steelblue', alpha=0.9)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('年成本 (元)')
    ax.set_title('库存成本对比：当前策略 vs EOQ优化策略\nInventory Cost Comparison', 
                 fontsize=13, fontweight='bold')
    ax.legend()
    
    # 标注数值
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f'{bar.get_height():.0f}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f'{bar.get_height():.0f}', ha='center', fontsize=9)
    
    save_fig('12_cost_waterfall.png')

# =====================================================
# 6. 库存模拟
# =====================================================
def simulate_inventory(daily_demand_mean, daily_demand_std, safety_result, eoq_result):
    """模拟90天库存变化"""
    np.random.seed(42)
    days = 90
    demand = np.random.normal(daily_demand_mean, daily_demand_std, days)
    demand = np.maximum(demand, 5)
    
    ROP = safety_result['再订货点 ROP (件)']
    EOQ = eoq_result['EOQ (件)']
    L = SUPPLY_CONFIG['lead_time_days']
    
    inventory = [EOQ]  # 初始库存
    orders_placed = []
    
    for t in range(1, days):
        inv = inventory[-1]
        
        # 收到提前期的订单
        for order_day, order_qty in orders_placed:
            if t == order_day + L:
                inv += order_qty
        orders_placed = [(od, oq) for od, oq in orders_placed if t < od + L]
        
        # 出货
        inv -= demand[t]
        inv = max(inv, 0)
        
        # 补货决策
        if inv <= ROP and not any(od == t for od, _ in orders_placed):
            orders_placed.append((t, EOQ))
        
        inventory.append(inv)
    
    inventory = np.array(inventory)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(inventory, linewidth=1.5, color='steelblue', label='库存水平')
    ax.axhline(ROP, color='darkred', linestyle='--', linewidth=1.5, label=f'再订货点 ({ROP:.0f})')
    ax.axhline(safety_result['安全库存 (件)'], color='orange', linestyle=':', linewidth=1.5,
               label=f'安全库存 ({safety_result["安全库存 (件)"]:.0f})')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)
    
    # 标注下单事件
    for order_day, order_qty in orders_placed:
        ax.axvline(order_day, color='green', alpha=0.4, linestyle='--', linewidth=1)
        ax.annotate(f'订货{order_qty:.0f}', (order_day, inventory[order_day]),
                   textcoords="offset points", xytext=(0, 15), ha='center', fontsize=8, color='green')
    
    ax.set_title('90天库存模拟 (s,Q策略)\nInventory Simulation with (s,Q) Policy', 
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('天数')
    ax.set_ylabel('库存量 (件)')
    ax.legend(loc='upper right')
    
    save_fig('13_inventory_simulation.png')
    
    return inventory

# =====================================================
# 7. 主函数
# =====================================================
def main():
    print("="*60)
    print(" 📦 库存优化模块 - EOQ + 安全库存 + 补货策略")
    print("="*60)
    
    df, daily_mean, daily_std, annual_demand, future_mean = load_and_calc()
    print(f"\n📊 日需求: μ={daily_mean:.1f}, σ={daily_std:.1f}")
    print(f"📊 年需求: {annual_demand:.0f} 件")
    
    # EOQ分析
    print("\n--- 1. EOQ经济订货批量分析 ---")
    eoq_result = calc_eoq()
    for k, v in eoq_result.items():
        print(f"   {k}: {v}")
    
    # 安全库存
    print("\n--- 2. 安全库存 & 再订货点 ---")
    safety_result = calc_safety_stock(daily_mean, daily_std)
    for k, v in safety_result.items():
        print(f"   {k}: {v}")
    
    # 成本对比
    print("\n--- 3. 成本对比分析 ---")
    comparison, total_savings = calc_cost_comparison(eoq_result, safety_result, annual_demand, daily_mean)
    print(comparison.to_string(index=False))
    print(f"\n 💰 年总成本节省: {total_savings:.0f} 元 ({total_savings/comparison.loc[5,'当前策略']*100:.1f}%)")
    
    # 保存结果
    eoq_df = pd.DataFrame([eoq_result])
    safety_df = pd.DataFrame([safety_result])
    eoq_df.to_csv(os.path.join(REPORT_DIR, 'eoq_result.csv'), index=False, encoding='utf-8-sig')
    safety_df.to_csv(os.path.join(REPORT_DIR, 'safety_stock_result.csv'), index=False, encoding='utf-8-sig')
    comparison.to_csv(os.path.join(REPORT_DIR, 'cost_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # 图表
    print("\n📈 生成库存优化图表...")
    plot_eoq_sensitivity()
    plot_inventory_strategy(daily_mean, daily_std, safety_result)
    plot_cost_waterfall(eoq_result, safety_result, annual_demand, daily_mean)
    
    # 模拟
    print("\n🎮 运行库存模拟...")
    simulate_inventory(daily_mean, daily_std, safety_result, eoq_result)
    
    print(f"\n{'='*60}")
    print(f" ✅ 库存优化完成！")
    print(f" 📋 优化建议: EOQ={eoq_result['EOQ (件)']:.0f}件, ")
    print(f"    ROP={safety_result['再订货点 ROP (件)']:.0f}件, ")
    print(f"    年节省={total_savings:.0f}元")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()