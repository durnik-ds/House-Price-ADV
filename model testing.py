import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb

data = pd.read_csv('Data/train.csv', sep=',', header=0)

GRADES = ['Ex', 'Gd', 'TA', 'Fa', 'Po']

data['LotFrontage'].fillna(0, inplace=True)

data['LotShape'] = np.where(data['LotShape'] == 'Reg', 1, 0)

data['CentralAir'] = np.where(data['CentralAir'] == 'Y', 1, 0)

data['GarageCars'] = np.where(data['GarageCars'] > 2, 1, 0)

data['TotRmsAbvGrd'] = np.where(data['TotRmsAbvGrd'] > 6, 1, 0)

data['Electrical'] = np.where(data['Electrical'] == 'SBrkr', 1, 0)

neighborhoods = ['Crawfor', 'ClearCr', 'Somerst', 'Veenker', 'Timber', 'StoneBr', 'NridgHt', 'NoRidge']
data['Neighborhood'] = np.where(data['Neighborhood'].isin(neighborhoods), 1, 0)

data['MasVnrArea'].fillna(0, inplace=True)

data['BldgType'] = np.where(data['BldgType'] == '1Fam', 1, 0)

data['HouseStyle'] = np.where((data['HouseStyle'] == '2.5Fin') | (data['HouseStyle'] == '2Story'), 1, 0)

data['HalfBath'] = np.where(data['HalfBath'] == 1, 1, 0)

data['FullBath'] = np.where(data['FullBath'] > 1, 1, 0)

data['LandSlope'] = np.where((data['LandSlope'] == 'Sev') | (data['LandSlope'] == 'Mod'), 1, 0)

data['1stFlrSF'] = np.log(data['1stFlrSF'])

data['2ndFlrSF'] = np.log1p(data['2ndFlrSF'])

data['ExterQual'] = np.where(data['ExterQual'].isin(GRADES[0:2]), 1, 0)

data['KitchenQual'] = np.where(data['KitchenQual'].isin(GRADES[0:2]), 1, 0)

data['BsmtQual'] = np.where(data['BsmtQual'].isin(GRADES[0:1]), 1, 0)

data['BsmtFullBath'] = np.where(data['BsmtFullBath'] > 0, 1, 0)

data['BedroomAbvGr'] = np.where(data['BedroomAbvGr'] > 1, 1, 0)

data['Foundation'] = np.where(data['Foundation'] == 'PConc', 1, 0)

data['YearBuilt'] = np.log(data['YearBuilt'])

data['BsmtFinType1'] = np.where(data['BsmtFinType1'] == 'GLQ', 1, 0)

data['HeatingQC'] = np.where(data['HeatingQC'].isin(GRADES[0:1]), 1, 0)

data['GarageQual'] = np.where(data['GarageQual'].isin(GRADES[0:3]), 1, 0)

config = ['CulDSac', 'FR3']
data['LotConfig'] = np.where(data['LotConfig'].isin(config), 1, 0)

data['BsmtFinSF1'] = np.log1p(data['BsmtFinSF1'])

data['LandContour'] = np.where(data['LandContour'] == 'HLS', 1, 0)

data['SaleCondition'] = np.where((data['SaleCondition'] == 'Partial') | (data['SaleCondition'] == 'Normal'), 1, 0)

data['Functional'] = np.where(data['Functional'] == 'Typ', 1, 0)

conditions = ['PosA', 'PosN', 'RRNn', 'RRNe']
data['Condition1'] = np.where(data['Condition1'].isin(conditions), 1, 0)

sale_types = ['New', 'Con', 'CWD', 'ConLI']
data['SaleType'] = np.where(data['SaleType'].isin(sale_types), 1, 0)

zonnings = ['FV', 'RL']
data['MSZoning'] = np.where(data['MSZoning'].isin(zonnings), 1, 0)

classes = [60, 120, 75, 20]
data['MSSubClass'] = np.where(data['MSSubClass'].isin(classes), 1, 0)

X = data[['LotArea', 'OverallQual', 'Fireplaces', 'OpenPorchSF', 'LotShape', 'CentralAir', 'GarageCars',
          'GrLivArea', 'Electrical', 'LotFrontage', 'Neighborhood', 'MasVnrArea', 'OverallCond', 'BldgType',
          'HouseStyle', 'HalfBath', 'FullBath', 'LandSlope', '1stFlrSF', '2ndFlrSF', 'ExterQual', 'KitchenQual',
          'BsmtQual', 'BsmtFullBath', 'BedroomAbvGr', 'Foundation', 'YearBuilt', 'BsmtFinType1', 'HeatingQC',
          'GarageQual', 'LotConfig', 'BsmtFinSF1', 'LandContour', 'SaleCondition', 'Functional', 'Condition1',
          'SaleType', 'MSZoning', 'MSSubClass']]

y = data['SalePrice'].values

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=21
)

mdl = LinearRegression(fit_intercept=True)
mdl = mdl.fit(X_train, y_train)

test_prediction = mdl.predict(X_test)

train_prediction = mdl.predict(X_train)
print('Error on a train set: ', np.sqrt(mean_squared_error(np.log(y_train), np.log(train_prediction))))
print('Error on a test set: ', np.sqrt(mean_squared_error(np.log(y_test), np.log(test_prediction))))

train_errors = np.abs(y_train - train_prediction)
percentile99 = np.percentile(train_errors, 99)
is_big_error = np.where(train_errors > percentile99, 1, 0)


# delete outliers
inline_mask = train_errors <= percentile99
X_train, y_train = X_train[inline_mask], y_train[inline_mask]

# X_train.to_csv('X_train.csv')
# y_train.to_csv('y_train.csv')

test_errors = np.abs(y_test - test_prediction)
percentile99 = np.percentile(test_errors, 99)
is_big_error = np.where(test_errors > percentile99, 1, 0)


inline_mask = test_errors <= percentile99
X_test, y_test = X_test[inline_mask], y_test[inline_mask]

mdl = LinearRegression(fit_intercept=True)
mdl = mdl.fit(X_train, y_train)

test_prediction = mdl.predict(X_test)

train_prediction = mdl.predict(X_train)
print('Error on a train set: ', np.sqrt(mean_squared_error(np.log(y_train), np.log(train_prediction))))
print('Error on a test set: ', np.sqrt(mean_squared_error(np.log(y_test), np.log(test_prediction))))

X_train['SalePrice'] = y_train
X_test['SalePrice'] = y_test


# X_train.to_csv('final_train.csv', index=False)
# X_test.to_csv('final_test.csv', index=False)

dtrain = xgb.DMatrix(X_train, y_train)
dtest = xgb.DMatrix(X_test, y_test)

params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.24,
        'subsample': 0.75,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.85,
        'lambda': 0.95,
        'gamma': 1.5,
        'max_depth': 3,
        'min_child_weight': 2,
        'eval_metric': 'rmse',
        'seed': 21
         }

mdl = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    early_stopping_rounds=20,
    evals=[(dtrain, 'Train'), (dtest, 'Test')]
)

cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=500,
    early_stopping_rounds=20,
    nfold=4,
    verbose_eval=True
)
