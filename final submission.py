import pandas as pd
import numpy as np
import xgboost as xgb

sdf = pd.read_csv('Data/test.csv', sep=',', header=0)

train = pd.read_csv('final_train.csv', sep=',', header=0)
test = pd.read_csv('final_test.csv', sep=',', header=0)
sample = pd.read_csv('Data/sample_submission.csv', sep=',', header=0, index_col='Id')

tdata = pd.concat([train, test], axis=0)

X = tdata[[col for col in tdata if col not in {'SalePrice'}]].copy()
y = tdata['SalePrice'].values

GRADES = ['Ex', 'Gd', 'TA', 'Fa', 'Po']

sdf['LotFrontage'].fillna(0, inplace=True)

sdf['LotShape'] = np.where(sdf['LotShape'] == 'Reg', 1, 0)

sdf['CentralAir'] = np.where(sdf['CentralAir'] == 'Y', 1, 0)

sdf['GarageCars'] = np.where(sdf['GarageCars'] > 2, 1, 0)

sdf['TotRmsAbvGrd'] = np.where(sdf['TotRmsAbvGrd'] > 6, 1, 0)

sdf['Electrical'] = np.where(sdf['Electrical'] == 'SBrkr', 1, 0)

neighborhoods = ['Crawfor', 'ClearCr', 'Somerst', 'Veenker', 'Timber', 'StoneBr', 'NridgHt', 'NoRidge']
sdf['Neighborhood'] = np.where(sdf['Neighborhood'].isin(neighborhoods), 1, 0)

sdf['MasVnrArea'].fillna(0, inplace=True)

sdf['BldgType'] = np.where(sdf['BldgType'] == '1Fam', 1, 0)

sdf['HouseStyle'] = np.where((sdf['HouseStyle'] == '2.5Fin') | (sdf['HouseStyle'] == '2Story'), 1, 0)

sdf['HalfBath'] = np.where(sdf['HalfBath'] == 1, 1, 0)

sdf['FullBath'] = np.where(sdf['FullBath'] > 1, 1, 0)

sdf['LandSlope'] = np.where((sdf['LandSlope'] == 'Sev') | (sdf['LandSlope'] == 'Mod'), 1, 0)

sdf['1stFlrSF'] = np.log(sdf['1stFlrSF'])

sdf['2ndFlrSF'] = np.log1p(sdf['2ndFlrSF'])

sdf['ExterQual'] = np.where(sdf['ExterQual'].isin(GRADES[0:2]), 1, 0)

sdf['KitchenQual'] = np.where(sdf['KitchenQual'].isin(GRADES[0:2]), 1, 0)

sdf['BsmtQual'] = np.where(sdf['BsmtQual'].isin(GRADES[0:1]), 1, 0)

sdf['BsmtFullBath'] = np.where(sdf['BsmtFullBath'] > 0, 1, 0)

sdf['BedroomAbvGr'] = np.where(sdf['BedroomAbvGr'] > 1, 1, 0)

sdf['Foundation'] = np.where(sdf['Foundation'] == 'PConc', 1, 0)

sdf['YearBuilt'] = np.log(sdf['YearBuilt'])

sdf['BsmtFinType1'] = np.where(sdf['BsmtFinType1'] == 'GLQ', 1, 0)

sdf['HeatingQC'] = np.where(sdf['HeatingQC'].isin(GRADES[0:1]), 1, 0)

sdf['GarageQual'] = np.where(sdf['GarageQual'].isin(GRADES[0:3]), 1, 0)

config = ['CulDSac', 'FR3']
sdf['LotConfig'] = np.where(sdf['LotConfig'].isin(config), 1, 0)

sdf['BsmtFinSF1'] = np.log1p(sdf['BsmtFinSF1'])

sdf['LandContour'] = np.where(sdf['LandContour'] == 'HLS', 1, 0)

sdf['SaleCondition'] = np.where((sdf['SaleCondition'] == 'Partial') | (sdf['SaleCondition'] == 'Normal'), 1, 0)

sdf['Functional'] = np.where(sdf['Functional'] == 'Typ', 1, 0)

conditions = ['PosA', 'PosN', 'RRNn', 'RRNe']
sdf['Condition1'] = np.where(sdf['Condition1'].isin(conditions), 1, 0)

sale_types = ['New', 'Con', 'CWD', 'ConLI']
sdf['SaleType'] = np.where(sdf['SaleType'].isin(sale_types), 1, 0)

zonnings = ['FV', 'RL']
sdf['MSZoning'] = np.where(sdf['MSZoning'].isin(zonnings), 1, 0)

classes = [60, 120, 75, 20]
sdf['MSSubClass'] = np.where(sdf['MSSubClass'].isin(classes), 1, 0)

submission = sdf[['LotArea', 'OverallQual', 'Fireplaces', 'OpenPorchSF', 'LotShape', 'CentralAir', 'GarageCars',
          'GrLivArea', 'Electrical', 'LotFrontage', 'Neighborhood', 'MasVnrArea', 'OverallCond', 'BldgType',
          'HouseStyle', 'HalfBath', 'FullBath', 'LandSlope', '1stFlrSF', '2ndFlrSF', 'ExterQual', 'KitchenQual',
          'BsmtQual', 'BsmtFullBath', 'BedroomAbvGr', 'Foundation', 'YearBuilt', 'BsmtFinType1', 'HeatingQC',
          'GarageQual', 'LotConfig', 'BsmtFinSF1', 'LandContour', 'SaleCondition', 'Functional', 'Condition1',
          'SaleType', 'MSZoning', 'MSSubClass']]

X_train, y_train = X, y

X_test = submission
y_test = sample['SalePrice']

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
    params=params,  # словарь с набором параметров модели
    dtrain=dtrain,  # матрица с обучающими данными
    num_boost_round=56  # обучающие и тестовые данные для оценки качесва
)

y_test = mdl.predict(dtest)

sample['SalePrice'] = mdl.predict(dtest)

print(sample)

# sample.to_csv('Submission_durnik.csv')
