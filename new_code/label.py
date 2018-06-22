import pandas as pd
from datetime import datetime, timedelta


# convenient function for feature merging
def merge_df(df1, df2):
    return df2 if df1 is None else df1.merge(df2, on='user_id', how='left')

# get overall labels
def get_label(data, begin_date, end_date):

    # filter by date range
    order_df = data.order_df[(data.order_df.o_date >= begin_date) & (data.order_df.o_date <= end_date)]

    # filter by target cate
    cate_30_101_cond = (order_df.cate == 30) | (order_df.cate == 101)
    order_df = order_df[cate_30_101_cond]

    # transform date label representation and sort
    order_df.o_date = (order_df.o_date - begin_date).apply(lambda x: x.days).astype(int)
    order_df = order_df.sort_values(['user_id', 'o_date'])

    # start building label
    label_df = data.user_df[['user_id']]

    ########## LABEL START ##########

    # order num: 30 101
    label_order_num_df = order_df. \
                        groupby('user_id')['o_id']. \
                        nunique(). \
                        reset_index(). \
                        rename(columns={'o_id': 'order_num'})
    label_df = merge_df(label_df, label_order_num_df)

    # first day of order: 30 101
    label_pred_date_df = order_df. \
                        drop_duplicates('user_id', keep='first') \
                        [['user_id', 'o_date']]. \
                        rename(columns={'o_date': 'pred_date'})
    label_df = merge_df(label_df, label_pred_date_df)

    ########## LABEL END ##########

    label_df = label_df.fillna(0).sort_values(by='user_id')

    return label_df


def label_to_output(label_df, begin_date):

    # sort by order num
    output_df = label_df_. \
                sort_values(by='order_num', ascending=False). \
                drop('order_num', axis=1) \
                [:50000]

    # transform date representation
    output_df.pred_date = output_df.pred_date.apply(lambda x: begin_date + timedelta(x))
    
    return output_df
