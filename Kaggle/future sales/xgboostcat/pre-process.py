import pandas as pd
import os


def open_csv(file):
    path = '/kaggle/input/competitive-data-science-predict-future-sales'
    return pd.read_csv(os.path.join(path, file))


items = open_csv('items.csv')
shops = open_csv('shops.csv')
item_categories = open_csv('item_categories.csv')
train = open_csv('sales_train.csv')
submission = open_csv('sample_submission.csv')
test = open_csv('test.csv')

category = {row[2]: row[3] for row in items.itertuples(index=True, name='Pandas')}

train['category'] = train.apply(lambda row: category[row['item_id']], axis=1)
train['month'] = train.apply(lambda row: row['date_block_num'] % 12, axis=1)
train['year'] = train.apply(lambda row: row['date_block_num'] // 12, axis=1)


train = train.pivot_table(index=['shop_id', 'category', 'year', 'month'],
                          values='item_cnt_day',
                          aggfunc='sum',
                          fill_value=0,
                          dropna=False).reset_index()

print('Dataframe manipulation completed.')

# print(train)

train.to_csv('new_train_file.csv', index=False)