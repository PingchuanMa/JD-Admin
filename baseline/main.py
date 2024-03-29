# -*- coding: utf-8 -*-
# author：Cookly
from util import DataLoader, Features
from SBBTree_ONLINE import SBBTree
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import lightgbm as lgb

# test code
Data = DataLoader(
  FILE_jdata_sku_basic_info='../data/jdata_sku_basic_info.csv',
  FILE_jdata_user_action='../data/jdata_user_action.csv',
  FILE_jdata_user_basic_info='../data/jdata_user_basic_info.csv',
  FILE_jdata_user_comment_score='../data/jdata_user_comment_score.csv',
  FILE_jdata_user_order='../data/jdata_user_order.csv'
)

# train data
TrainFeatures = Features(
  DataLoader=Data,
  PredMonthBegin=datetime(2017, 4, 1),
  PredMonthEnd=datetime(2017, 4, 30),
  FeatureMonthList=[(datetime(2017, 3, 1), datetime(2017, 3, 31), 1), \
                    (datetime(2017, 1, 1), datetime(2017, 3, 31), 3), \
                    (datetime(2016, 10, 1), datetime(2017, 3, 31), 6)],
  MakeLabel=True
)

# pred data
PredFeatures = Features(
  DataLoader=Data,
  PredMonthBegin=datetime(2017, 5, 1),
  PredMonthEnd=datetime(2017, 5, 31),
  FeatureMonthList=[(datetime(2017, 4, 1), datetime(2017, 4, 30), 1), \
                    (datetime(2017, 2, 1), datetime(2017, 4, 30), 3), \
                    (datetime(2016, 11, 1), datetime(2017, 4, 30), 6)],
  MakeLabel=False
)

train_features = TrainFeatures.TrainColumns

cols = TrainFeatures.IDColumns + TrainFeatures.LabelColumns + train_features

###########
params = {
  'task': 'train',
  'boosting_type': 'gbdt',
  'objective': 'regression',
  'metric': {'l2', 'auc'},
  'num_leaves': 31,
  'learning_rate': 0.05,
  'feature_fraction': 0.9,
  'bagging_fraction': 0.8,
  'bagging_freq': 5,
  'verbose': 0
}
###############################################################
# model = SBBTree(params=params, stacking_num=5, bagging_num=3, bagging_test_size=0.33, num_boost_round=10000,
#                 early_stopping_rounds=200)

# train 下个月购买次数预测 回归模型
train_features = TrainFeatures.TrainColumns
train_label_BuyNum = 'Label_30_101_BuyNum'

X = TrainFeatures.data_BuyOrNot_FirstTime[train_features].values
y = TrainFeatures.data_BuyOrNot_FirstTime[train_label_BuyNum].values

m1 = TrainFeatures.data_BuyOrNot_FirstTime['OM1_o_date_cate_30_101_gap_var'].values
m3 = TrainFeatures.data_BuyOrNot_FirstTime['OM3_o_date_cate_30_101_gap_var'].values
m6 = TrainFeatures.data_BuyOrNot_FirstTime['OM6_o_date_cate_30_101_gap_var'].values
m1[np.isnan(m1)] = np.inf
m3[np.isnan(m3)] = np.inf
m6[np.isnan(m6)] = np.inf

confidence = np.exp(-m1 / 4.9) / 2 + np.exp(-m3 / 12.5) / 3 + np.exp(-m6 / 21.2) / 6

X_pred = PredFeatures.data_BuyOrNot_FirstTime[train_features].values

y *= confidence

model = lgb.train(params, lgb.Dataset(X, y))

PredFeatures.data_BuyOrNot_FirstTime[train_label_BuyNum] = model.predict(X_pred, num_iteration=model.best_iteration)
y_pred = PredFeatures.data_BuyOrNot_FirstTime[train_label_BuyNum]

feature_impo_df = pd.DataFrame(np.vstack([train_features, model.feature_importance()]).T, columns=['name', 'impo'])
feature_impo_df['impo'] = feature_impo_df['impo'].astype(int)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  print(feature_impo_df.sort_values(by='impo', ascending=False))

# model.train(X, y)
# PredFeatures.data_BuyOrNot_FirstTime[train_label_BuyNum] = model.predict(X_pred)

###############################################################
params = {
  'task': 'train',
  'boosting_type': 'gbdt',
  'objective': 'regression',
  'metric': {'l2'},
  'num_leaves': 31,
  'learning_rate': 0.05,
  'feature_fraction': 0.9,
  'bagging_fraction': 0.8,
  'bagging_freq': 5,
  'verbose': 0
}
# model = SBBTree(params=params, stacking_num=5, bagging_num=3, bagging_test_size=0.33, num_boost_round=10000,
#                 early_stopping_rounds=200)

# train 当月首次购买时间预测 回归模型
train_label_FirstTime = 'Label_30_101_FirstTime'
X = TrainFeatures.data_BuyOrNot_FirstTime[TrainFeatures.data_BuyOrNot_FirstTime[train_label_FirstTime] > 0][
  train_features].values
y = TrainFeatures.data_BuyOrNot_FirstTime[TrainFeatures.data_BuyOrNot_FirstTime[train_label_FirstTime] > 0][
  train_label_FirstTime].values

X_pred = PredFeatures.data_BuyOrNot_FirstTime[train_features].values

model = lgb.train(params, lgb.Dataset(X, y))

PredFeatures.data_BuyOrNot_FirstTime[train_label_FirstTime] = model.predict(X_pred, num_iteration=model.best_iteration)
y_pred = PredFeatures.data_BuyOrNot_FirstTime[train_label_FirstTime]

feature_impo_df = pd.DataFrame(np.vstack([train_features, model.feature_importance()]).T, columns=['name', 'impo'])
feature_impo_df['impo'] = feature_impo_df['impo'].astype(int)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
  print(feature_impo_df.sort_values(by='impo', ascending=False))

# model.fit(X, y)
# PredFeatures.data_BuyOrNot_FirstTime[train_label_FirstTime] = model.predict(X_pred)

####################################################################
# submit
columns = ['user_id'] + [train_label_BuyNum] + [train_label_FirstTime]
out_submit = PredFeatures.data_BuyOrNot_FirstTime[columns].sort_values(['Label_30_101_BuyNum'], ascending=False)
out_submit[train_label_FirstTime] = out_submit[train_label_FirstTime].map(
  lambda day: datetime(2017, 5, 1) + timedelta(days=int(day + 0.49 - 1)))

# clip
out_submit[train_label_FirstTime] = out_submit[train_label_FirstTime].map(
  lambda day: datetime(2017, 5, 14) if day > datetime(2017, 5, 14) else day)

out_submit = out_submit[['user_id'] + [train_label_FirstTime]]
out_submit.columns = ['user_id', 'pred_date']
out_submit.head(50000).to_csv('../submit/predict.csv', index=False, header=True)
