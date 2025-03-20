# 导入库
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import os  # 新增，用于检查模型文件
import joblib

# 读取数据
train_df = pd.read_csv('Beijing_Weather_Train.csv')
test_df = pd.read_csv('Beijing_Weather_Test.csv')

# **新增：去除异常值（IQR 方法）**
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)  # 第 25 百分位数
        Q3 = df[col].quantile(0.75)  # 第 75 百分位数
        IQR = Q3 - Q1  # 四分位距
        lower_bound = Q1 - 1.5 * IQR  # 下界
        upper_bound = Q3 + 1.5 * IQR  # 上界
        df[col] = df[col].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)  # 异常值设为 NaN
    df.fillna(method='bfill', inplace=True)  # 处理 NaN 值，向后填充
    return df

# **应用异常值去除**
columns_to_check = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
train_df = remove_outliers(train_df, columns_to_check)
test_df = remove_outliers(test_df, columns_to_check)

# 计算新的特征：湿度与气压比
def humidity_pressure_ratio(df):
    df['humidity_pressure_ratio'] = df['humidity'] / df['meanpressure']
    return df

train_df = humidity_pressure_ratio(train_df)
test_df = humidity_pressure_ratio(test_df)

# 计算过去 N 天的均值
lookback_mean = 3  # 过去 3 天均值
train_df['meantemp_mean_3d'] = train_df['meantemp'].rolling(window=lookback_mean).mean()
train_df['humidity_mean_3d'] = train_df['humidity'].rolling(window=lookback_mean).mean()
train_df['wind_speed_mean_3d'] = train_df['wind_speed'].rolling(window=lookback_mean).mean()
train_df['meanpressure_mean_3d'] = train_df['meanpressure'].rolling(window=lookback_mean).mean()

test_df['meantemp_mean_3d'] = test_df['meantemp'].rolling(window=lookback_mean).mean()
test_df['humidity_mean_3d'] = test_df['humidity'].rolling(window=lookback_mean).mean()
test_df['wind_speed_mean_3d'] = test_df['wind_speed'].rolling(window=lookback_mean).mean()
test_df['meanpressure_mean_3d'] = test_df['meanpressure'].rolling(window=lookback_mean).mean()

# 填充 NaN 值（前几天数据不足）
train_df.fillna(method='bfill', inplace=True)
test_df.fillna(method='bfill', inplace=True)

# 解析日期
def get_date_columns(date):
    year, month, day = date.split('-')
    return month, day

train_df[['month', 'day']] = pd.DataFrame(train_df['date'].apply(get_date_columns).tolist(), index=train_df.index)
test_df[['month', 'day']] = pd.DataFrame(test_df['date'].apply(get_date_columns).tolist(), index=test_df.index)

# 选择特征和目标变量
targets = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
features = ['month', 'day', 'humidity', 'wind_speed', 'meanpressure', 'humidity_pressure_ratio',
            'meantemp_mean_3d', 'humidity_mean_3d', 'wind_speed_mean_3d', 'meanpressure_mean_3d'] + targets

tr_timeseries = train_df[features].values.astype('float32')
te_timeseries = test_df[features].values.astype('float32')

# 归一化
scaler = MinMaxScaler()
tr_timeseries = scaler.fit_transform(tr_timeseries)
te_timeseries = scaler.transform(te_timeseries)

scaler_y = MinMaxScaler()
scaler_y.fit(train_df[targets])
# ✅ **保存 scaler_y 供 `predict_next_week.py` 使用**
joblib.dump(scaler_y, "scaler_y_B.pkl")

print("✅ scaler_y 已保存到 scaler_y_B.pkl")
'''
print("✅ 目标变量的原始最小值和最大值:")
print("🔍 scaler_y 归一化最小值:", scaler_y.data_min_)
print("🔍 scaler_y 归一化最大值:", scaler_y.data_max_)
print("最小值:", train_df[targets].min().values)
print("最大值:", train_df[targets].max().values)
'''
# 生成时间序列数据集
def create_dataset(dataset, lookback, target_size):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset[i:i + lookback, :-target_size]  # 过去 lookback 天的所有特征（不包含目标变量）
        target = dataset[i + lookback, -target_size:]  # 预测多个目标变量
        X.append(feature)
        y.append(target)

    # 🚀 先转 NumPy 数组，再转 Tensor，加快速度
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return torch.tensor(X), torch.tensor(y)

lookback = 7
target_size = len(targets)  # 目标变量个数
X_train, y_train = create_dataset(tr_timeseries, lookback, target_size)
X_test, y_test = create_dataset(te_timeseries, lookback, target_size)

# 数据加载
batch_size = 8
loader = data.DataLoader(data.TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

torch.save(X_test, "X_test_tensor_B.pth")  # 保存 X_test
print("✅ X_test 已保存到 X_test_tensor_B.pth")

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size=10, num_layers=2, hidden_size=128, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(128 * 2, 64)
        self.linear2 = nn.Linear(64, 8)
        self.output_linear = nn.Linear(8, target_size)  # 预测多个变量

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # 取最后时间步的输出
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.output_linear(x)  # 现在输出多个变量
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备: {device}")

# **初始化模型，并移动到正确的设备**
model = LSTMModel(target_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
loss_fn = nn.MSELoss()

# **如果有已保存的模型，加载并移动到正确的设备**
model_path = "weather_lstm_model_B.pth"
if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("✅ 已加载已有模型，继续训练！")
else:
    print("⚠️ 未找到模型，从零开始训练...")

# **确保训练数据也在正确的设备上**
X_train, y_train = X_train.to(device), y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# 训练模型
n_epochs = 500
best_score = None
best_weights = None

if __name__ == "__main__":
    print("✅ 直接运行 `ModelTrain_Beijing.py`，执行模型训练...")

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_train_pred = model(X_train)
            y_test_pred = model(X_test)

            # 分别计算 4 个目标变量的 RMSE
            train_rmse_per_variable = torch.sqrt(torch.mean((y_train_pred - y_train) ** 2, dim=0))
            test_rmse_per_variable = torch.sqrt(torch.mean((y_test_pred - y_test) ** 2, dim=0))

            # 打印每个变量的 RMSE
            print(f'Epoch {epoch}:')
            for i, var in enumerate(targets):
                print(f'  {var}: Train RMSE = {train_rmse_per_variable[i]:.5f}, Test RMSE = {test_rmse_per_variable[i]:.5f}')

            if best_score is None or test_rmse_per_variable.mean() < best_score:
                best_score = test_rmse_per_variable.mean()
                best_weights = model.state_dict()

    # **新增：训练完成后保存模型**
    if best_weights is not None:
        model.load_state_dict(best_weights)
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, model_path)
        print("✅ 训练完成，模型已保存！")


    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train).cpu().numpy()
        y_pred_test = model(X_test).cpu().numpy()

    # **可视化 4 个预测结果**
    plt.figure(figsize=(20, 10))

    for i, var in enumerate(targets):
        plt.subplot(2, 2, i + 1)
        plt.plot(tr_timeseries[:, -4 + i], label=f'Train {var}', color='blue')
        plt.plot(np.concatenate([np.full((lookback,), np.nan), y_pred_train[:, i]]), label=f'Train Prediction {var}', color='red')
        plt.plot(range(len(tr_timeseries[:, -4 + i]), len(tr_timeseries[:, -4 + i]) + len(te_timeseries[:, -4 + i])),
                 np.concatenate([np.full((lookback,), np.nan), y_pred_test[:, i]]), label=f'Test Prediction {var}', color='green')
        plt.plot(range(len(tr_timeseries[:, -4 + i]), len(tr_timeseries[:, -4 + i]) + len(te_timeseries[:, -4 + i])),
                 te_timeseries[:, -4 + i], label=f'Test {var}', color='yellow')
        plt.legend()
        plt.title(var)

    plt.tight_layout()
    plt.show()

