import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

path = 'dataframe.csv'

data = pd.read_csv(path)

print(data.head())
print(data.columns)

# Prediction Target
y = data.LotArea

feature_names = [
    'LotArea',
    'YearBuilt',
    'FullBath',
    'BedroomAbvGr',
    'TotRmsAbvGrd']

# Features
X = data[feature_names]

# Árvore de Regressão
model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)

predictions = model.predict(X)  # Dados para treino

# Erro absoluto
mean_absolute_error(y, predictions)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
model = DecisionTreeRegressor(random_state=1)
model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)

# Random Forests
forest_model = RandomForestRegressor(random_state=1)

# Output
output = pd.DataFrame({'Id': data.Id,
                       'SalePrice': val_predictions})
output.to_csv('submission.csv', index=False)
