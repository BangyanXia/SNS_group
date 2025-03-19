import numpy as np

# 加载 .npy 文件
data = np.load('D:/WorkSpace/PycharmProjects/iTransformer-main/results/GBP_to_CHY_iTransformer_custom_M_ft128_sl48_ll7_pl512_dm8_nh2_el1_dl2048_df1_fctimeF_ebTrue_dttest_projection_0/real_prediction.npy')

print("加载的数据:")
print(data)
