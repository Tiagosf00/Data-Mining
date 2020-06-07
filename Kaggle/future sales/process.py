import pandas as pd
import os

from sklearn.ensemble import RandomForestRegressor


def open_csv(file):
    # path = '/kaggle/input/competitive-data-science-predict-future-sales'
    path = ''
    return pd.read_csv(os.path.join(path, file))


train = open_csv('new_train_file.csv')
submission = open_csv('sample_submission.csv')
test = open_csv('test.csv')
print('Opening files completed.')

# PRE-PROCESSAMENTO

# Train
train['shop_id'] = train['shop_id'].astype('str')
train['item_id'] = train['item_id'].astype('str')

X_train = train.drop(columns=['33'])
y_train = train['33']

print('X_train and y_train created.')

# Test
test['shop_id'] = test['shop_id'].astype('str')
test['item_id'] = test['item_id'].astype('str')
X_test = test.merge(train, how="left", on=["shop_id", "item_id"]).fillna(0)
X_test.drop(columns=['0'], inplace=True)
X_test.drop(columns=['ID'], inplace=True)

print('X_test created')


# PROCESSAMENTO

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train, y_train)
y_test = regressor.predict(X_test)
submission['item_cnt_month'] = y_test
submission.to_csv('submission.csv', index=False)

print('Submission file done.')
