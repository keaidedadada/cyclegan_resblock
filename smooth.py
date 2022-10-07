import pandas as pd
import numpy as np
import os


def smooth(csv_path, csv_name, weight=0.85):
    path = os.path.join(csv_path, csv_name)
    data = pd.read_csv(filepath_or_buffer=path, header=0, names=['Step', 'Value'],
                       dtype={'Step': np.int64, 'Value': np.float32})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step': data['Step'].values, 'Value': smoothed})
    namelist = path.split('\\')
    name = namelist[-1].split('.')
    name = name[0]
    name = 'smooth_' + name + '.csv'
    savepath = os.path.join(csv_path, name)
    save.to_csv(savepath)
    print('done!')


if __name__ == '__main__':
    smooth('D:\dai', 'd.csv')
