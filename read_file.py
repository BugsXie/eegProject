import os
import random
from sklearn.model_selection import train_test_split

# 设置文件夹路径和类别名称
folders = [r'D:\PycharmProjects\eegProject\data\Test_EEG\HC', r'D:\PycharmProjects\eegProject\data\Test_EEG\MDD', r'D:\PycharmProjects\eegProject\data\Test_EEG\BD']
class_names = ['HC', 'MDD', 'BD']

# 定义划分比例
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# 创建存储数据集的字典
data_splits = {'train': [], 'val': [], 'test': []}

# 遍历每个类别的文件夹
for folder, class_name in zip(folders, class_names):
    # 获取文件夹中的.mat文件列表
    file_list = os.listdir(folder)
    file_list = [os.path.join(folder, file) for file in file_list if file.endswith('Clean.mat')]

    # 随机划分数据
    train_files, remaining_files = train_test_split(file_list, train_size=train_ratio, random_state=42)
    val_files, test_files = train_test_split(remaining_files, test_size=test_ratio / (test_ratio + val_ratio),
                                             random_state=42)

    # 将数据加入到对应的数据集中
    data_splits['train'].extend([(file, class_name) for file in train_files])
    data_splits['val'].extend([(file, class_name) for file in val_files])
    data_splits['test'].extend([(file, class_name) for file in test_files])

# 打印每个划分的数据统计信息
for split, files in data_splits.items():
    print(f"{split} set: {len(files)} samples")
