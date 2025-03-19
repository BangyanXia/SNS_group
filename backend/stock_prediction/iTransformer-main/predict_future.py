import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from model.iTransformer import Model  # 确保 iTransformer 代码路径正确

# **🚨 强制使用 CPU 运行**
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# **🔹 配置文件路径**
checkpoint_path = "./checkpoints/stock_iTransformer_custom_M_ft365_sl48_ll7_pl512_dm8_nh2_el1_dl2048_df1_fctimeF_ebTrue_dttest_projection_0/checkpoint.pth"  # 你的训练好的模型文件
data_path = "./cleaned_stock_fed_rate.csv"  # 你的数据集文件
pred_len = 7  # **预测未来 7 天**

# **🔹 强制使用 CPU**
device = torch.device("cpu")
print(f"✅ 设备: {device}")

# **🔹 加载数据**
df = pd.read_csv(data_path)
df['date'] = pd.to_datetime(df['date'])

# **🔹 获取最新的 `96` 天输入数据**
seq_len = 365  # 必须和训练时的 `seq_len` 保持一致
latest_data = df.iloc[-seq_len:][['Open', 'Close', 'High', 'Low', 'Volume', 'DFF']].values

# **🔹 归一化数据**
scaler = StandardScaler()
# **确保 fit 和 transform 输入相同**
scaler.fit(df[['Open', 'Close', 'High', 'Low', 'Volume', 'DFF']].values)  # 转换为 NumPy 数组
latest_data_scaled = scaler.transform(latest_data)  # 确保一致


# **🔹 转换为 Tensor**
latest_data_scaled = torch.tensor(latest_data_scaled, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 6)

# **🔹 未来 7 天时间戳**
last_date = df['date'].iloc[-1]
future_dates = [last_date + timedelta(days=i + 1) for i in range(pred_len)]

# **🔹 生成 `x_mark_enc`**
def generate_time_features(dates):
    """ 生成时间特征：月、日、星期 """
    return np.array([[d.month, d.day, d.weekday()] for d in dates], dtype=np.float32)

x_mark_enc = generate_time_features(df['date'].iloc[-seq_len:])  # 取最近 `seq_len` 天的时间特征
x_mark_dec = generate_time_features(future_dates)  # 未来 `pred_len` 天的时间特征

# **🔹 转换为 Tensor**
x_mark_enc = torch.tensor(x_mark_enc, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 3)
x_mark_dec = torch.tensor(x_mark_dec, dtype=torch.float32).unsqueeze(0).to(device)  # (1, pred_len, 3)

# **🔹 手动构造 `model_configs`，确保包含所有超参数**
from argparse import Namespace
model_configs = Namespace(
    enc_in=6, dec_in=6, c_out=6, seq_len=365, pred_len=pred_len, label_len=48,
    d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048, dropout=0.1, factor=1,
    activation='gelu', output_attention=False, use_norm=True, embed='timeF',
    freq='h', class_strategy='projection'
)

# **🔹 初始化模型**
model = Model(model_configs).to(device)

# **🔹 加载训练好的模型权重**
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# **🔹 进行预测**
with torch.no_grad():
    x_dec = torch.zeros((1, pred_len, 6)).to(device)  # 空的解码器输入
    predicted_seq = model(latest_data_scaled, x_mark_enc, x_dec, x_mark_dec)

# **🔹 逆归一化**
predictions = predicted_seq.cpu().numpy().squeeze()
predictions_original = scaler.inverse_transform(predictions)  # 还原到真实数据

# **🔹 保存预测结果**
df_preds = pd.DataFrame(predictions_original, columns=['Open', 'Close', 'High', 'Low', 'Volume', 'DFF'])
df_preds.insert(0, 'date', future_dates)  # 添加日期列

# **🔹 导出 CSV 文件**
pred_csv_path = "predicted_stock_7days.csv"
df_preds.to_csv(pred_csv_path, index=False)
print(f"✅ 预测完成，结果已保存至 {pred_csv_path}")

# **🔹 显示预测结果**
print(df_preds)
