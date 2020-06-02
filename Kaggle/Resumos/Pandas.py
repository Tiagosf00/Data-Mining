import pandas as pd

# DataFrame (table)

tabela = pd.DataFrame({'Yes': [50, 21], 'No': [131, 2]},  # 2 columns (Yes e No)
                      index=['Product A', 'Product B'])   # 2 rows (A e B)

# Series (list)

lista = pd.Series([1, 2, 3, 4, 5])
lista = pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'],
                  name='Product A')

# Reading data files

file = pd.read_csv("file.csv")  # Has a lot of parameters
# index_col=0 -> Starts the columns with 0

print(file.shape)  # (rows, columns)
print(file.head())  # Shows the first five rows

# Save

tabela.to_csv('tabela.csv')


# Acessing column

print(tabela.Yes)
print(tabela['YES'])
# prints a Serie out of a DataFrame

print(tabela['YES'][0])
# prints the first element of the column

# Acessing row

# loc (exclusive 0:10 = 0 to 9) only work with numbers

print(tabela.iloc[0])
# prints the first row

print(tabela.iloc[:, 0])
# prints the first column

tabela.iloc[:3, 0]  # first column, from row 0 to 2
tabela.iloc[[1, 2, 3], 0]

print(tabela.iloc[-5:])
# prints the five last rows

# iloc (inclusive 0:10 = 0 to 10)

tabela.loc[[0, 1], ['YES', 'NO']]

# index

tabela.set_index("title")
# sets the title as the new index column

# Conditionals

tabela.country == 'Italy'
# boolean Serie

tabela.loc[tabela.country == 'Italy']
# every row that has 'Italy' in country

tabela.loc[(tabela.country == 'Italy') & (tabela.points >= 90)]  # |

# isin
tabela.loc[tabela.country.isin(['Italy', 'France'])]

# isnull notnull
tabela.loc[tabela.price.notnull()]


# Assigning data

tabela['critic'] = 'everyone'
tabela['index_backwards'] = range(len(tabela), 0, -1)


# Working on the data

# describe
tabela.points.describe()  # gives statistical data about it
tabela.taster_name.describe()  # different data for strings

# mean
tabela.points.mean()
# median
tabela.points.median()

# unique
tabela.taster_name.unique()  # unique values

# value_counts
tabela.taster_name.value_counts()  # number of times each one appears

# Maps

points_mean = tabela.points.mean()
tabela.points.map(lambda p: p - points_mean)  # Series


def remean_points(row):  # apply
    row.points = row.points - points_mean
    return row


tabela.apply(remean_points, axis='columns')  # DataFrame

# Faster than add or map

points_mean = tabela.points.mean()
tabela.points - points_mean
# just subtract

tabela.country + ' - ' + tabela.region_1
# just add


# Group Data

tabela.groupby('points').points.count()  # Group by point, than count
tabela.groupby('points').price.min()  # Grupou by point, get the minimum price

# apply
tabela.groupby('winery').apply(lambda df: df.title.iloc[0])
# group by winery, and get the title of the first appear of this winery

tabela.groupby(['country', 'province']).apply(lambda df: df.loc[df.points.idxmax()])
# group by two caracteristics

# agg
a = tabela.groupby(['country']).price.agg([len, min, max])
# group by country and apply 3 methods (3 columns)

# groupby returns a Multi-index
# one index can represent one or more rows
a = a.reset_index()  # convert it into single index

# Sort

tabela.sort_values(by='len', ascending=False)
# sorts in ascending order if parameter not specified
# by=['min', 'max'] -> first in min than in max (break ties)

tabela.sort_index()  # sorts by the indexes

# Data type
tabela.price.dtype  # returns the type
tabela.points.astype('float64')  # changes the type


# Missing data
tabela[pd.isnull(tabela.country)]  # shows only rows with missing data

# fillna
tabela.region_2.fillna("Unknown")  # replaces each NaN with Unknown

# replace
tabela.taster_twitter_handle.replace("@kerinokeefe", "@kerino")


# Rename

tabela.rename(columns={'points': 'score'})
tabela.rename(index={0: 'firstEntry', 1: 'secondEntry'})
tabela.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns')


# Combining

# concat (vertical)
canadian_youtube = pd.read_csv("CAvideos.csv")
british_youtube = pd.read_csv("GBvideos.csv")

pd.concat([canadian_youtube, british_youtube])

# join (horizontal)
left = canadian_youtube.set_index(['title', 'trending_date'])
right = british_youtube.set_index(['title', 'trending_date'])

left.join(right, lsuffix='_CAN', rsuffix='_UK')