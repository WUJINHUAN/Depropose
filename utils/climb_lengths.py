import torch


# 定义计算两点之间的欧氏距离函数
def calculate_distance(point1, point2):
    return torch.sqrt(torch.sum((point1 - point2) ** 2))


# 计算肢体长度
def calculate_limb_lengths(joint_tensor):
    limb_lengths = []

    joint_indices = {
        'hip': 0, 'r_hip': 1, 'r_knee': 2, 'r_foot': 3,
        'l_hip': 4, 'l_knee': 5, 'l_foot': 6, 'spine': 7,
        'thorax': 8, 'neck': 9, 'head': 10, 'l_shoulder': 11,
        'l_elbow': 12, 'l_wrist': 13, 'r_shoulder': 14,
        'r_elbow': 15, 'r_wrist': 16
    }

    # 存储每个样本的肢体长度
    all_lengths = []

    for i in range(joint_tensor.shape[0]):  # 遍历每个样本
        sample = joint_tensor[i]
        lengths = []

        # 上肢长度
        lengths.append(calculate_distance(sample[joint_indices['l_shoulder']], sample[joint_indices['l_elbow']]))
        lengths.append(calculate_distance(sample[joint_indices['r_shoulder']], sample[joint_indices['r_elbow']]))

        lengths.append(calculate_distance(sample[joint_indices['l_elbow']], sample[joint_indices['l_wrist']]))
        lengths.append(calculate_distance(sample[joint_indices['r_elbow']], sample[joint_indices['r_wrist']]))

        # 下肢长度
        lengths.append(calculate_distance(sample[joint_indices['l_hip']], sample[joint_indices['l_knee']]))
        lengths.append(calculate_distance(sample[joint_indices['r_hip']], sample[joint_indices['r_knee']]))

        lengths.append(calculate_distance(sample[joint_indices['l_knee']], sample[joint_indices['l_foot']]))
        lengths.append(calculate_distance(sample[joint_indices['r_knee']], sample[joint_indices['r_foot']]))

        # 躯干长度
        lengths.append(calculate_distance(sample[joint_indices['thorax']], sample[joint_indices['head']]))
        lengths.append(calculate_distance(sample[joint_indices['spine']], sample[joint_indices['hip']]))

        # 总腿长度
        lengths.append(lengths[-2] + lengths[-1])  # left_leg
        lengths.append(lengths[-4] + lengths[-3])  # right_leg

        all_lengths.append(lengths)

    return torch.tensor(all_lengths)


# 计算肢体长度误差
def calculate_length_error(predicted_tensor, label_tensor):
    predicted_lengths = calculate_limb_lengths(predicted_tensor)

    label_lengths = calculate_limb_lengths(label_tensor)

    # 计算每个肢体部分的误差（均方误差）
    length_errors = {}
    total_error = 0  # 用于累加总误差

    # 针对每个肢体部分计算误差
    limb_parts = ['left_upper_arm', 'right_upper_arm', 'left_forearm', 'right_forearm',
                  'left_thigh', 'right_thigh', 'left_shank', 'right_shank',
                  'torso', 'spine', 'left_leg', 'right_leg']

    for i, limb in enumerate(limb_parts):
        predicted = predicted_lengths[:, i].reshape(-1)  # 预测肢体长度
        label = label_lengths[:, i].reshape(-1)  # 标签肢体长度

        # 计算误差（均方误差）
        error = torch.mean((predicted - label) ** 2)
        length_errors[limb] = error.item()  # 获取误差的标量值

        # 累加总误差
        total_error += error.item()

    return length_errors, total_error
