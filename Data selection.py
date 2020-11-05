import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_missing(df):

    print(" \nshow the boolean Dataframe : \n\n", df.isnull())

    print(" \nCount total NaN at each column in a DataFrame : \n\n",
          df.isnull().sum())


def basic_stats(column_name):
    """
    takes a column name and returns basic statisctics of it
    :param:
    :return:
    """

    grouped = data.groupby(column_name).agg(
        min_price=('SalePrice', np.min),
        mean_price=('SalePrice', np.mean),
        max_price=('SalePrice', np.max),
        median_price=('SalePrice', np.median),
        std_price=('SalePrice', np.std)
    )

    uniques = data[column_name].value_counts()

    left_joined = pd.concat([grouped, uniques], axis=1)
    left_joined.rename(columns={column_name: 'count'}, inplace=True)
    left_joined.sort_values('mean_price', inplace=True, ascending=False)

    return left_joined


def scatter(x, y):
    """
    creates a scatter plot of a variable
    :param x:
    :param y:
    :return:
    """

    plt.scatter(x, y)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
    plt.show()


data = pd.read_csv('Data/train.csv', sep=',', header=0)


data['MasVnrType'].fillna(0, inplace=True)

print(len(data.columns))

print(data.columns)

print(basic_stats('MSSubClass'))

find_missing(data['MSSubClass'])

scatter(data['MSSubClass'], data['SalePrice'])
