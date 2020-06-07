import pandas as pd
import os
import matplotlib.pyplot as plt


def open_csv(file):
    path = '/kaggle/input/competitive-data-science-predict-future-sales'
    return pd.read_csv(os.path.join(path, file))


# Abre os arquivos
items = open_csv('items.csv')
shops = open_csv('shops.csv')
item_categories = open_csv('item_categories.csv')
train = open_csv('sales_train.csv')
submission = open_csv('sample_submission.csv')
test = open_csv('test.csv')
print('Files opened.')

# Remove dados duplicados
attr = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
train.drop_duplicates(attr, inplace=True)

# Retira dados sem relevancia para a análise
train.drop(columns=['date'], inplace=True)
train.drop(columns=['item_price'], inplace=True)

# Retira outliers
train = train[(train['item_cnt_day'] > 0) & (train['item_cnt_day'] < 1000)]

# Gera a pivot table
train = train.pivot_table(index=['shop_id', 'item_id'],
                          columns='date_block_num',
                          values='item_cnt_day',
                          aggfunc='sum',
                          fill_value=0).reset_index()


train.to_csv('new_train_file.csv', index=False)

print('Dataframe manipulation completed.')
