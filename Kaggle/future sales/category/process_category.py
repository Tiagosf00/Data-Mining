import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor


def open_csv(file):
    path = '/kaggle/input/competitive-data-science-predict-future-sales'
    return pd.read_csv(os.path.join(path, file))


items = open_csv('items.csv')
shops = open_csv('shops.csv')
item_categories = open_csv('item_categories.csv')
train = pd.read_csv('/kaggle/working/new_train_file.csv')
submission = open_csv('sample_submission.csv')
test = open_csv('test.csv')
print('Opening files completed.')

# PRE-PROCESSAMENTO

# Test
test['category'] = test.apply(lambda row: [row['item_id']], axis=1)
test['shop_id'] = test['shop_id'].astype('str')
test['category'] = test['category'].astype('str')
test['year'] = '2'
test['month'] = '10'
X_test = test[['shop_id', 'category', 'year', 'month']]

# print(X_test)
print('X_test created')

# Train
train['shop_id'] = train['shop_id'].astype('str')
train['category'] = train['category'].astype('str')
train['month'] = train['month'].astype('str')
train['year'] = train['year'].astype('str')

train = train[(train['year'] < '2') | (train['month'] < '10')].reset_index()
X_train = train[['shop_id', 'category', 'year', 'month']]
y_train = train['item_cnt_day']

# print(X_train)
# print(y_train)


print('X_train and y_train created.')

# PROCESSAMENTO

regressor = RandomForestRegressor(n_estimators=10)
regressor.fit(X_train, y_train)
y_test = regressor.predict(X_test)
submission['item_cnt_month'] = y_test/2160
print(y_test/2160)
submission.to_csv('submission.csv', index=False)

print('Submission file done.')