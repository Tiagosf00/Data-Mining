import pandas as pd
import os
import xgboost as xgb


def open_csv(file):
    path = '/kaggle/input/competitive-data-science-predict-future-sales'
    return pd.read_csv(os.path.join(path, file))


train = pd.read_csv('/kaggle/working/new_train_file.csv')
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

param = {'max_depth': 10,
         'subsample': 1,
         'min_child_weight': 0.5,
         'eta': 0.3,
         'num_round': 1000,
         'seed': 1,
         'silent': 0,
         'eval_metric': 'rmse'}

xgbtrain = xgb.DMatrix(X_train.values, y_train.values)
bst = xgb.train(param, xgbtrain)

y_test = bst.predict(xgb.DMatrix(X_test.values))

submission['item_cnt_month'] = y_test
submission.to_csv('submissionxgboost.csv', index=False)

print('Submission file done.')
