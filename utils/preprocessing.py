import os

import numpy as np
import pandas as pd


def merge_csv_files(input_dir, group_size=10):
    csv_files = [file for file in os.listdir(input_dir) if file.endswith(".csv")]

    grouped_files = [
        csv_files[i : i + group_size] for i in range(0, len(csv_files), group_size)
    ]

    data = []
    label = []

    for group_idx, group in enumerate(grouped_files):
        group_data = []
        df = pd.DataFrame()

        for csv_file in group:
            file_path = os.path.join(input_dir, csv_file)
            df = pd.read_csv(file_path, header=None)
            pure_data = df.iloc[1:, 1:-1]
            group_data.append(pure_data.to_numpy().flatten())

        group_data = np.array(group_data)
        group_label = df.iloc[1, -1]
        label.append(group_label)
        data.append(group_data)

    return np.array(data, dtype=np.float32), np.array(label, dtype=np.int64)
