import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# 读取 `npy` 文件
true_values = np.load('D:/WorkSpace/PycharmProjects/iTransformer-main/results/GBP_to_CHY_iTransformer_custom_M_ft365_sl48_ll7_pl512_dm8_nh2_el1_dl2048_df1_fctimeF_ebTrue_dttest_projection_0/true.npy')  # 真实值
pred_values = np.load('D:/WorkSpace/PycharmProjects/iTransformer-main/results/GBP_to_CHY_iTransformer_custom_M_ft365_sl48_ll7_pl512_dm8_nh2_el1_dl2048_df1_fctimeF_ebTrue_dttest_projection_0/pred.npy')  # 预测值

# 确保数据维度匹配
print(f"True shape: {true_values.shape}, Pred shape: {pred_values.shape}")


# 选择不同时间步的 `Close` 价格
t_values = [0]  # 选择 `t=0`（最近预测）、`t=48`（中期）、`t=95`（长期）
titles = ["Short-term Prediction (t=0)", "Mid-term Prediction (t=48)", "Long-term Prediction (t=95)"]

# 绘制三张独立的图
for i, t in enumerate(t_values):
    plt.figure(figsize=(10, 4))

    true_selected = true_values[:, t, 3]  # 真实值
    pred_selected = pred_values[:, t, 3]  # 预测值

    plt.plot(true_selected, label="GroundTruth", color="blue", linestyle="solid")
    plt.plot(pred_selected, label="Prediction", color="orange", linestyle="dashed")

    plt.xlabel("Time Step")
    plt.ylabel("Stock Close Price")
    plt.title(titles[i])
    plt.legend()

    plt.show()


# 计算误差
metrics_results = {}
for t in [0]:  # 选择不同时间步
    true_selected = true_values[:, t, 1]  # 真实值
    pred_selected = pred_values[:, t, 1]  # 预测值

    mse = mean_squared_error(true_selected, pred_selected)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_selected, pred_selected)
    r2 = r2_score(true_selected, pred_selected)

    metrics_results[f't={t}'] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2
    }

# 转换为 DataFrame 并打印
metrics_df = pd.DataFrame(metrics_results).T
print(metrics_df)

# 如果想存成 CSV 方便查看
metrics_df.to_csv("prediction_metrics.csv", index=True)
