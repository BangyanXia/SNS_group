# 介绍
## 使用数据集
### 训练集：[DailyDelhiClimateTrain.csv](DailyDelhiClimateTrain.csv)
### 测试集：[DailyDelhiClimateTest.csv](DailyDelhiClimateTest.csv)
## 代码
### 模型训练：[test.py](test.py)
### 预测7天后气象指标：[predict_next_week.py](predict_next_week.py)
### 测试CUDA状态：[testcuda.py](testcuda.py)
## 模型
### 由 [test.py](test.py) 训练完成的LSTM模型：[weather_lstm_model.pth](weather_lstm_model.pth)
## 文件
### 测试数据的 Tensor 文件，存储了 X_test，用于模型评估和预测：[X_test_tensor.pth](X_test_tensor.pth)
### 数据归一化文件：[scaler.pkl](scaler.pkl)
### 目标变量的归一化参数：[scaler_y.pkl](scaler_y.pkl)

