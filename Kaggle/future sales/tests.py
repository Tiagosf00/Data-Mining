import pandas as pd
import os


def open_csv(file):
    # path = '/kaggle/input/competitive-data-science-predict-future-sales'
    path = ''
    return pd.read_csv(os.path.join(path, file))




sub = open_csv('xgboostcat/submissionxgboostcat.csv')

print(sub['item_cnt_month'])

print(sub['item_cnt_month']/2160)

print(sub['item_cnt_month'])

# print(sub.describe())

# sub['item_cnt_month'] = sub.apply(lambda row: max(0, min(row['item_cnt_month']/2160, 20)), axis=1)

# print(sub.describe())

# sub.to_csv('xgboostcat/sub_normalxgboostcat.csv', index=False)
