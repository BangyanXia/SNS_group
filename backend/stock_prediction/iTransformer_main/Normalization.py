import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
file_path = "./cleaned_stock_fed_rate.csv"  # 修改为你的文件路径
df = pd.read_csv(file_path)

# 确保日期列不参与归一化
if 'date' in df.columns:
    df.set_index('date', inplace=True)  # 将日期列设为索引

# 归一化所有数值列
scaler = MinMaxScaler(feature_range=(0, 1))
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# 保存归一化后的数据
df_normalized.to_csv("normalized_stock_fed_rate.csv")
print("✅ save to: normalized_stock_fed_rate.csv")

# 显示归一化后的数据
print(df_normalized.head())
