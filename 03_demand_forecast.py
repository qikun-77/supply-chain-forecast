# ============================================================
# 模块03：需求预测 - 5种方法全方位对比
# 方法：移动平均 | 指数平滑(Holt-Winters) | ARIMA | Prophet | LSTM(PyTorch)
# 产出：模型对比表 + 最优模型 + 未来30天预测 + 残差诊断
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, pickle, json
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# =====================================================
# 0. 配置
# =====================================================
DATA_PATH = 'data/ecommerce_demand.csv'
OUTPUT_DIR = 'outputs/charts'
MODEL_DIR = 'outputs/models'
REPORT_DIR = 'outputs/reports'
FORECAST_DAYS = 30          # 预测未来30天
TRAIN_RATIO = 0.85          # 训练集比例

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# =====================================================
# 1. 数据准备
# =====================================================
def load_data():
    df = pd.read_csv(DATA_PATH)
    df['ds'] = pd.to_datetime(df['ds'])
    df = df.set_index('ds').sort_index()
    series = df['demand'].astype(float)
    return series, df

def train_test_split(series):
    split_idx = int(len(series) * TRAIN_RATIO)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test

# =====================================================
# 2. 评估函数
# =====================================================
def calc_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {'MAE': round(mae, 2), 'RMSE': round(rmse, 2), 'MAPE': round(mape, 2)}

# =====================================================
# 3. 方法1：简单移动平均 (Baseline)
# =====================================================
def forecast_moving_average(train, test, window=7):
    preds = []
    history = list(train.values)
    for t in range(len(test)):
        pred = np.mean(history[-window:])
        preds.append(pred)
        history.append(test.iloc[t])
    metrics = calc_metrics(test.values, np.array(preds))
    return np.array(preds), metrics, {'model_name': 'Moving Average (MA-7)', 'window': window}

# =====================================================
# 4. 方法2：Holt-Winters 指数平滑
# =====================================================
def forecast_holt_winters(train, test):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        model = ExponentialSmoothing(
            train, trend='add', seasonal='add', seasonal_periods=7,
            initialization_method='estimated'
        ).fit()
    except:
        model = ExponentialSmoothing(
            train, trend='add', seasonal='add', seasonal_periods=7
        ).fit()
    
    preds = model.forecast(len(test))
    metrics = calc_metrics(test.values, preds.values)
    return preds.values, metrics, {'model_name': 'Holt-Winters (HW)', 'model': model}

# =====================================================
# 5. 方法3：SARIMA
# =====================================================
def forecast_sarima(train, test):
    from statsmodels.tsa.arima.model import ARIMA
    try:
        model = ARIMA(train, order=(2,1,2), seasonal_order=(1,1,1,7)).fit(method_kwargs={'maxiter': 200})
    except:
        model = ARIMA(train, order=(1,1,1)).fit()
    
    preds = model.forecast(len(test))
    metrics = calc_metrics(test.values, preds.values)
    return preds.values, metrics, {'model_name': 'SARIMA(2,1,2)(1,1,1)7', 'model': model}

# =====================================================
# 6. 方法4：Facebook Prophet
# =====================================================
def forecast_prophet(train, test):
    from prophet import Prophet
    df_train = pd.DataFrame({'ds': train.index, 'y': train.values})
    df_train['floor'] = 0
    
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, 
                    daily_seasonality=False, changepoint_prior_scale=0.05,
                    interval_width=0.95)
    model.fit(df_train)
    
    future = model.make_future_dataframe(periods=len(test), include_history=False)
    forecast = model.predict(future)
    preds = forecast['yhat'].values
    
    metrics = calc_metrics(test.values, preds)
    return preds, metrics, {'model_name': 'Facebook Prophet', 'model': model}

# =====================================================
# 7. 方法5：LSTM (PyTorch)
# =====================================================
def create_sequences(data, seq_len=14):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def forecast_lstm(train, test, seq_len=14, epochs=50):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import MinMaxScaler
    
    # 归一化
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
    test_scaled = scaler.transform(test.values.reshape(-1, 1)).flatten()
    
    # 创建序列
    X_train, y_train = create_sequences(train_scaled, seq_len)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    
    # LSTM模型
    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_size=32, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])
    
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
    
    # 预测
    model.eval()
    history = list(train_scaled[-seq_len:])
    preds_scaled = []
    with torch.no_grad():
        for _ in range(len(test)):
            x = torch.tensor([history[-seq_len:]], dtype=torch.float32).unsqueeze(-1)
            p = model(x).item()
            preds_scaled.append(p)
            history.append(test_scaled[len(preds_scaled)-1])  # 用真实值
    
    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    metrics = calc_metrics(test.values, preds)
    return preds, metrics, {'model_name': 'LSTM (PyTorch)', 'seq_len': seq_len, 'epochs': epochs}

# =====================================================
# 8. 模型对比与选择
# =====================================================
def compare_models(results_dict):
    """汇总所有模型结果"""
    rows = []
    for name, (preds, metrics, info) in results_dict.items():
        rows.append({
            '模型': name,
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE'],
            'MAPE(%)': metrics['MAPE'],
        })
    
    comparison = pd.DataFrame(rows).sort_values('MAPE(%)')
    best_name = comparison.iloc[0]['模型']
    
    print("\n" + "="*70)
    print(" 🏆 模型效果对比 (测试集)")
    print("="*70)
    print(comparison.to_string(index=False))
    print(f"\n 🥇 最优模型: {best_name} (MAPE最小)")
    
    comparison.to_csv(os.path.join(REPORT_DIR, 'model_comparison.csv'), index=False, encoding='utf-8-sig')
    
    # 保存最优模型名
    with open(os.path.join(MODEL_DIR, 'best_model_name.json'), 'w') as f:
        json.dump({'best_model': best_name}, f, ensure_ascii=False)
    
    return comparison, best_name

# =====================================================
# 9. 可视化
# =====================================================
def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 已保存: {path}")

def plot_all_predictions(train, test, results_dict, best_name):
    """所有模型预测曲线对比"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    model_names = list(results_dict.keys())
    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']
    
    for i, name in enumerate(model_names):
        ax = axes[i]
        preds = results_dict[name][0]
        
        ax.plot(train.index[-90:], train.values[-90:], color='gray', linewidth=1, alpha=0.7, label='训练集')
        ax.plot(test.index, test.values, color='steelblue', linewidth=1.2, alpha=0.7, label='实际值')
        ax.plot(test.index, preds, color=colors[i], linewidth=1.8, alpha=0.9, label=f'{name} 预测')
        
        m = results_dict[name][1]
        ax.set_title(f'{name}\nMAE={m["MAE"]:.1f} | RMSE={m["RMSE"]:.1f} | MAPE={m["MAPE"]:.1f}%', 
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.tick_params(axis='x', rotation=30)
    
    # 最后一个子图：最优模型
    ax = axes[5]
    best_preds = results_dict[best_name][0]
    best_m = results_dict[best_name][1]
    ax.plot(train.index[-90:], train.values[-90:], color='gray', linewidth=1, alpha=0.5, label='历史')
    ax.plot(test.index, test.values, color='steelblue', linewidth=1.5, label='实际')
    ax.plot(test.index, best_preds, color='darkred', linewidth=2, label=f'{best_name}')
    ax.fill_between(test.index, test.values, best_preds, alpha=0.2, color='red')
    ax.set_title(f'🏆 最优模型: {best_name}\nMAPE={best_m["MAPE"]:.1f}%', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    
    fig.suptitle('5种需求预测方法对比 Demand Forecast Methods Comparison', fontsize=16, fontweight='bold', y=1.01)
    save_fig('07_forecast_comparison.png')

def plot_future_forecast(train, best_name):
    """未来30天预测"""
    from prophet import Prophet
    
    # 用Prophet或最佳模型做未来预测
    df_prophet = pd.DataFrame({'ds': train.index, 'y': train.values})
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=FORECAST_DAYS)
    forecast = model.predict(future)
    
    # 提取最后30天
    future_dates = forecast['ds'].iloc[-FORECAST_DAYS:]
    future_preds = forecast['yhat'].iloc[-FORECAST_DAYS:]
    future_lower = forecast['yhat_lower'].iloc[-FORECAST_DAYS:]
    future_upper = forecast['yhat_upper'].iloc[-FORECAST_DAYS:]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 历史数据（最近180天）
    ax.plot(train.index[-180:], train.values[-180:], color='steelblue', linewidth=1.2, label='历史销量')
    ax.plot(future_dates, future_preds, color='darkred', linewidth=2, label=f'未来{FORECAST_DAYS}天预测')
    ax.fill_between(future_dates, future_lower, future_upper, alpha=0.25, color='red', label='95%置信区间')
    
    ax.axvline(train.index[-1], color='gray', linestyle='--', alpha=0.7, label='今日')
    ax.set_title(f'未来{FORECAST_DAYS}天需求预测\nDemand Forecast - Next {FORECAST_DAYS} Days', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('预测销量 (件)')
    ax.legend(loc='upper left')
    
    save_fig('08_future_forecast.png')
    
    # 保存预测结果
    forecast_df = pd.DataFrame({
        'date': future_dates.values,
        'forecast': future_preds.values,
        'lower_bound': future_lower.values,
        'upper_bound': future_upper.values,
    })
    forecast_df.to_csv(os.path.join(REPORT_DIR, 'future_30day_forecast.csv'), index=False, encoding='utf-8-sig')
    print(f"\n📋 未来30天预测摘要:")
    print(f"   平均预测: {future_preds.mean():.0f} 件/天")
    print(f"   最低预测: {future_preds.min():.0f} 件/天")
    print(f"   最高预测: {future_preds.max():.0f} 件/天")

def plot_residual_diagnostics(train, test, best_name, results_dict):
    """残差诊断"""
    best_preds = results_dict[best_name][0]
    residuals = test.values - best_preds
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # 残差时序
    axes[0,0].plot(test.index, residuals, 'o-', markersize=3, color='steelblue')
    axes[0,0].axhline(0, color='red', linestyle='--')
    axes[0,0].set_title('残差时序 Residuals Over Time')
    axes[0,0].set_ylabel('残差')
    
    # 残差直方图
    axes[0,1].hist(residuals, bins=30, color='steelblue', edgecolor='white', density=True)
    from scipy import stats
    mu_r, std_r = residuals.mean(), residuals.std()
    x_r = np.linspace(residuals.min(), residuals.max(), 100)
    axes[0,1].plot(x_r, stats.norm.pdf(x_r, mu_r, std_r), 'r--', linewidth=2)
    axes[0,1].set_title(f'残差分布 μ={mu_r:.1f}, σ={std_r:.1f}')
    
    # 预测vs实际
    axes[1,0].scatter(test.values, best_preds, alpha=0.5, color='steelblue')
    axes[1,0].plot([test.values.min(), test.values.max()], 
                   [test.values.min(), test.values.max()], 'r--', linewidth=1.5)
    axes[1,0].set_title('预测值 vs 实际值')
    axes[1,0].set_xlabel('实际')
    axes[1,0].set_ylabel('预测')
    
    # ACF残差
    from statsmodels.graphics.tsaplots import plot_acf
    plot_acf(residuals, lags=30, ax=axes[1,1])
    axes[1,1].set_title('残差自相关 (白噪声检验)')
    
    fig.suptitle(f'残差诊断 - {best_name}', fontsize=14, fontweight='bold')
    save_fig('09_residual_diagnostics.png')

# =====================================================
# 10. 主函数
# =====================================================
def main():
    print("="*70)
    print(" 🔮 需求预测模块 - 5方法对比")
    print("="*70)
    
    series, df = load_data()
    train, test = train_test_split(series)
    print(f"\n📊 训练集: {len(train)} 天 | 测试集: {len(test)} 天")
    
    results_dict = {}
    
    # 方法1：移动平均
    print("\n[1/5] 移动平均 MA-7...")
    preds, metrics, info = forecast_moving_average(train, test)
    results_dict['移动平均(MA-7)'] = (preds, metrics, info)
    print(f"      MAPE={metrics['MAPE']:.1f}%")
    
    # 方法2：Holt-Winters
    print("[2/5] Holt-Winters 指数平滑...")
    preds, metrics, info = forecast_holt_winters(train, test)
    results_dict['Holt-Winters'] = (preds, metrics, info)
    print(f"      MAPE={metrics['MAPE']:.1f}%")
    
    # 方法3：SARIMA
    print("[3/5] SARIMA...")
    preds, metrics, info = forecast_sarima(train, test)
    results_dict['SARIMA'] = (preds, metrics, info)
    print(f"      MAPE={metrics['MAPE']:.1f}%")
    
    # 方法4：Prophet
    print("[4/5] Facebook Prophet...")
    preds, metrics, info = forecast_prophet(train, test)
    results_dict['Prophet'] = (preds, metrics, info)
    print(f"      MAPE={metrics['MAPE']:.1f}%")
    
    # 方法5：LSTM
    print("[5/5] LSTM (PyTorch)...")
    preds, metrics, info = forecast_lstm(train, test)
    results_dict['LSTM'] = (preds, metrics, info)
    print(f"      MAPE={metrics['MAPE']:.1f}%")
    
    # 对比
    comparison, best_name = compare_models(results_dict)
    
    # 绘图
    print("\n📈 生成预测图表...")
    plot_all_predictions(train, test, results_dict, best_name)
    plot_future_forecast(train, best_name)
    plot_residual_diagnostics(train, test, best_name, results_dict)
    
    print(f"\n{'='*70}")
    print(f" ✅ 预测模块完成！")
    print(f" 🥇 最优模型: {best_name}")
    print(f" 📁 输出: outputs/charts/ + outputs/models/ + outputs/reports/")
    print(f"{'='*70}")
    
    return comparison, best_name

if __name__ == '__main__':
    comparison, best_name = main()