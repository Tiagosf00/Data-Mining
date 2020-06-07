# PRE-PROCESSAMENTO

test['date_block_num'] = 34

train = train.sample(frac=0.1).reset_index(drop=True)

train['shop_id'] = train.shop_id.astype('str')
train['item_id'] = train.item_id.astype('str')
test['shop_id'] = test.shop_id.astype('str')
test['item_id'] = test.item_id.astype('str')

train = train.groupby(['item_id',
                       'date_block_num',
                       'shop_id']).agg({'item_cnt_day': ['sum']}).reset_index()

train.columns = train.columns.get_level_values(0)


attr = LabelBinarizer().fit(train['date_block_num'])

attr_train = pd.DataFrame(attr.transform(train['date_block_num']))
attr_test = pd.DataFrame(attr.transform(test['date_block_num']))


attr_test = attr_test.rename(columns=lambda x: f'month_{x}')
attr_train = attr_train.rename(columns=lambda x: f'month_{x}')


X_train = train[['shop_id', 'item_id']]
y_train = train['item_cnt_day']

X_test = test[['shop_id', 'item_id']]

X_train = X_train.join(attr_train)
X_test = X_test.join(attr_test)

print('Pre-process completed.')
