import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, GridSearchCV
pd.options.mode.chained_assignment = None

from data_loader import DataLoader
from feature import get_feature
from label import get_label, label_to_output
from eval import score_eval


########## PROJECT CONFIGURATION ##########

PRINT_FEATURE_IMPORTANCE = False
VALIDATION = False
DATA_PATH = '../data'
RESULT_PATH = '../result'


########## PREPROCESSING ##########

# load and transform data table
data = DataLoader(DATA_PATH)

# various date ranges for train/test and validation train/validation test
featured_month_periods = [1, 3, 6]

def get_dates(label_begin_date):
	label_end_date = label_begin_date + relativedelta(months=1, days=-1)
	feature_begin_dates = [label_begin_date - relativedelta(months=i) for i in featured_month_periods]
	feature_end_date = label_begin_date - relativedelta(days=1)
	return label_end_date, feature_begin_dates, feature_end_date

if VALIDATION:

	test_label_begin_date = datetime(2017, 4, 1)
	test_label_end_date, test_feature_begin_dates, test_feature_end_date = get_dates(test_label_begin_date)

	train_label_begin_date = datetime(2017, 3, 1)
	train_label_end_date, train_features_begin_dates, train_feature_end_date = get_dates(train_label_begin_date)

else:

	test_label_begin_date = datetime(2017, 5, 1)
	test_label_end_date, test_feature_begin_dates, test_feature_end_date = get_dates(test_label_begin_date)

	train_label_begin_date = datetime(2017, 4, 1)
	train_label_end_date, train_feature_begin_dates, train_feature_end_date = get_dates(train_label_begin_date)


########## FEATURE EXTRACTION ##########


# get training feature and label
train_feature = get_feature(data, train_feature_begin_dates, train_feature_end_date, featured_month_periods)
train_label = get_label(data, train_label_begin_date, train_label_end_date)

# get test feature
test_feature = get_feature(data, test_feature_begin_dates, test_feature_end_date, featured_month_periods)


########## MODEL TRAINING ##########

x_train = train_feature.drop('user_id', axis=1)
y_train = train_label.drop('user_id', axis=1)
x_test = test_feature.drop('user_id', axis=1)

model_params = {
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

y_pred = None
for label in y_train.columns.values:
    model = lgb.train(model_params, lgb.Dataset(x_train, y_train[label]))
    y_pred = model.predict(x_test) if y_pred is None else np.vstack([y_pred, model.predict(x_test)])
y_pred = y_pred.T


########## RESULT CONVERSION ##########

# see feature importance
if PRINT_FEATURE_IMPORTANCE:
	feature_impo_df = pd.DataFrame(np.array([x_train.columns.values, model.feature_importance()]).T, columns=['name', 'impo'])
	feature_impo_df['impo'] = feature_impo_df['impo'].astype(int)
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	  print(feature_impo_df.sort_values(by='impo', ascending=False))

# add back user id
pred_df = pd.DataFrame(y_pred, columns=y_train.columns.values)
pred_df['user_id'] = data.user_df[['user_id']]
pred_df = pred_df[['user_id', 'pred_date', 'order_num']]
output_df = label_to_output(pred_df, test_label_begin_date)

if VALIDATION:

	ground_truth = data.order_df[(data.order_df.o_date >= test_label_begin_date) & \
        (data.order_df.o_date <= test_label_end_date)]. \
        groupby('user_id')['o_date']. \
        min(). \
        reset_index(). \
        rename(columns={'o_date': 'order_date'})

	# output validation score to screen
	score = score_eval(ground_truth, output_df)
	print(score)

else:

	# output prediction result to file
	time_now = datetime.now().strftime("_%m%d_%H%M")
	output_df.to_csv(RESULT_PATH + '/prediction' + time_now + '.csv', index=False)
