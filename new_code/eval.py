import numpy as np
import pandas as pd

# calculate s1 score
def s1(y, pred):
	length = len(pred) # 50000
	o = np.isin(pred.user_id, y.user_id)
	w = 1 / (1 + np.log(np.arange(length) + 1))
	return np.sum(w * o) / np.sum(w)

# calculate s2 score
def s2(y, pred):
	length = len(y) # len(ground-truth)
	df = y.merge(pred, on='user_id', how='inner')
	fu = 10 / (10 + (df['pred_date'] - df['order_date']).dt.days ** 2)
	return np.sum(fu) / length

# calculate overall score
def score_eval(y, pred):
	# y: ground-truth user dataframe, columns.values = ['user_id', 'order_date']
	# pred: 50000 ordered prediction dataframe, columns.values = ['user_id', 'pred_date']
	alpha = 0.4
	s1_score = s1(y, pred)
	s2_score = s2(y, pred)
	s_score = alpha * s1_score + (1 - alpha) * s2_score
	print('S1:', s1_score, 'S2:', s2_score)
	print('S:', s_score)
	return s_score
	