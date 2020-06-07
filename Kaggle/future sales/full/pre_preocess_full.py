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
print('Files opened.')


# Retirando dados sem relevancia para a análise
train.drop(columns=['date'], inplace=True)
train.drop(columns=['item_price'], inplace=True)

n_shop = shops.shape[0]  # Número de lojas
n_item = items.shape[0]  # Número de produtos
n_date = 34

date = [k for k in range(n_date) for i in range(n_shop*n_item)]
item_cnt = [0 for i in range(n_shop*n_item)]*n_date
item = [i for i in range(n_item)]*n_shop*n_date
shop = [[i]*n_item for i in range(n_shop)]
shop = [x for sublist in shop for x in sublist]*n_date

zeros = pd.DataFrame(data={'date_block_num': date,
                           'shop_id': shop,
                           'item_id': item,
                           'item_cnt_day': item_cnt})
print('Zero rows created.')

train = pd.concat([train, zeros], ignore_index=True)
print('Zero rows included.')

train['month'] = train.apply(lambda row: row['date_block_num'] % 12, axis=1)
train['year'] = train.apply(lambda row: row['date_block_num'] // 12, axis=1)

train = train.pivot_table(index=['shop_id', 'item_id', 'year', 'month'],
                          values='item_cnt_day',
                          aggfunc='sum',
                          fill_value=0).reset_index()
print('Pivot table created.')
print(train)


train.to_csv('new_train_file.csv', index=False)

print('Dataframe manipulation completed.')
