import torch
import sys,os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Beijing.ModelTrain_Beijing import LSTMModel
import joblib


# 1️⃣ 加载训练好的模型
model_path = "Beijing/weather_lstm_model_B.pth"
model = LSTMModel(target_size=4)
model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.eval()
print("✅ 已加载训练好的模型")

# 2️⃣ 加载 X_test
X_test = torch.load("Beijing/X_test_tensor_B.pth")
print("✅ 成功加载 X_test，形状:", X_test.shape)

# 3️⃣ 预测未来 7 天
X_input = X_test[-1].unsqueeze(0)  # 取最后 7 天
future_preds = []

for _ in range(7):
    with torch.no_grad():
        y_pred = model(X_input).cpu().numpy()  # 预测 1 天
        future_preds.append(y_pred)

    # 4️⃣ **修正：补全 6 个缺失特征**
    y_pred_filled = np.zeros((1, 1, X_input.shape[2]))  # 形状 (1, 1, 10)
    y_pred_filled[:, :, -4:] = y_pred  # 只替换最后 4 个目标变量

    # 5️⃣ **修正：转换为 PyTorch Tensor**
    y_pred_tensor = torch.tensor(y_pred_filled, dtype=torch.float32)

    # 6️⃣ **修正：拼接维度**
    X_input = torch.cat((X_input[:, 1:, :], y_pred_tensor), dim=1)



# ✅ 先把 `future_preds` 转换成 `numpy` 数组
future_preds = np.array(future_preds).reshape(7, 4)
print(type(future_preds))


print("🔍 反归一化前（模型输出的归一化值）:")
print(future_preds)  # 确保 shape = (7, 4)

# ✅ 反归一化
scaler_y = joblib.load("Beijing/scaler_y_B.pkl")

future_preds = scaler_y.inverse_transform(future_preds)
print("✅ 反归一化完成")

# 8️⃣ 打印最终预测结果
print("未来 7 天的天气预测（温度、湿度、风速、气压）:")
print(future_preds)

# 9️⃣ 可视化
days = np.arange(1, 8)  # 未来 7 天的天数
labels = ["Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)", "Pressure (hPa)"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# 创建 2x2 子图
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

for i, ax in enumerate(axs.flat):
    ax.plot(days, future_preds[:, i], marker='o', color=colors[i], label=labels[i])
    ax.set_title(labels[i])
    ax.set_xlabel("Days")
    ax.set_ylabel(labels[i])
    ax.grid()
    ax.legend()

plt.tight_layout()
plt.show()
