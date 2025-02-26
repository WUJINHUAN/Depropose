import numpy as np

def normalization_stats(complete_data):
    # 计算沿样本轴的均值和标准差，结果形状为 (17, 3)
    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)
    return data_mean, data_std

def normalize_data(data, data_mean, data_std):
    # 标准化数据，保持形状不变 (32, 17, 3)
    data_out = (data - data_mean) / data_std
    return data_out

def unNormalizeData(normalized_data, data_mean, data_std):
    # 反标准化数据，恢复原始形状 (32, 17, 3)
    orig_data = normalized_data * data_std + data_mean
    return orig_data