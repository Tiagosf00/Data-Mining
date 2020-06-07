import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor


def open_csv(file):
    path = '/kaggle/input/competitive-data-science-predict-future-sales'
    return pd.read_csv(os.path.join(path, file))


items = open_csv('items.csv')
shops = open_csv('shops.csv')
item_categories = open_csv('item_categories.csv')
train = pd.read_csv('/kaggle/input/train-final/new_train_final.csv')
submission = open_csv('sample_submission.csv')
test = open_csv('test.csv')
print('Opening files completed.')

# Construção dos casos de treino e teste

# Test
test['shop_id'] = test['shop_id'].astype('str')
test['item_id'] = test['item_id'].astype('str')
test['year'] = '2'
test['month'] = '10'
test.drop(columns=['ID'], inplace=True)


print('X_test created')

# Train
train['shop_id'] = train['shop_id'].astype('str')
train['item_id'] = train['item_id'].astype('str')
train['month'] = train['month'].astype('str')
train['year'] = train['year'].astype('str')

y_train = train['item_cnt_day']
train.drop(columns=['item_cnt_day'], inplace=True)


print('X_train and y_train created.')

# PROCESSAMENTO

regressor = RandomForestRegressor(n_estimators=5)
regressor.fit(train, y_train)
y_test = regressor.predict(test)
submission['item_cnt_month'] = y_test
submission.to_csv('submission.csv', index=False)

print('Submission file done.')
