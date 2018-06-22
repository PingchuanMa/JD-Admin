import pandas as pd
from datetime import datetime

count = 0

# convenient function for feature merging
def merge_df(df1, df2):
    global count
    count += 1
    print(count)
    return df2 if df1 is None else df1.merge(df2, on='user_id', how='left')


# get overall features
def get_feature(data, begin_dates, end_date, featured_month_periods):

    # initialize as user features
    feature_df = data.user_df.copy()

    # get feature from different periods and merge together
    for begin_date, month_period in zip(begin_dates, featured_month_periods):

        # get features from action table and order table in particular period
        action_feature_df = get_action_feature(data, begin_date, end_date, month_period)
        order_feature_df = get_order_feature(data, begin_date, end_date, month_period)
        action_order_feature_df = get_action_order_feature(data, begin_date, end_date, month_period)

        # merge action and order features
        feature_df = merge_df(feature_df, action_feature_df)
        feature_df = merge_df(feature_df, order_feature_df)
        feature_df = merge_df(feature_df, action_order_feature_df)

    return feature_df


# get action features
def get_action_feature(data, begin_date, end_date, month_period):

    action_feature_df = data.user_df[['user_id']]

    # filter by date range
    df = data.action_df[(data.action_df.a_date >= begin_date) & (data.action_df.a_date <= end_date)]

    # logical conditions
    cate_30_cond = (df.cate == 30)
    cate_101_cond = (df.cate == 101)
    cate_30_101_cond = (df.cate == 30) | (df.cate == 101)
    cate_others_cond = (df.cate != 30) & (df.cate != 101)

    # feature name prefix
    name = str(month_period) + 'mo_action_'
    name_30 = name + '30_'
    name_101 = name + '101_'
    name_30_101 = name + '30_101_'
    name_others = name + 'others_'
    name_all = name + 'all_'

    ########## FEATURE START ##########

    # action sum: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['a_num']. \
                sum(). \
                reset_index(). \
                rename(columns={'a_num': name_30_101 + 'sum'})
    action_feature_df = merge_df(action_feature_df, temp_df)

    # action count: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['a_num']. \
                count(). \
                reset_index(). \
                rename(columns={'a_num': name_30_101 + 'count'})
    action_feature_df = merge_df(action_feature_df, temp_df)

    # action date count: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['a_date']. \
                count(). \
                reset_index(). \
                rename(columns={'a_date': name_30_101 + 'date_count'})
    action_feature_df = merge_df(action_feature_df, temp_df)

    # average day gap between actions: 30 101
    temp_df = df[cate_30_101_cond]. \
                sort_values('a_date'). \
                groupby('user_id')['a_date']. \
                agg(lambda x: x.diff().mean(skipna=True).days). \
                reset_index(). \
                rename(columns={'a_date': name_30_101 + 'date_gap_mean'})
    action_feature_df = merge_df(action_feature_df, temp_df)

    # most frequent action day: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['a_day']. \
                agg(lambda x: x.value_counts().index[0]). \
                reset_index(). \
                rename(columns={'a_day': name_30_101 + 'day_freq'})
    action_feature_df = merge_df(action_feature_df, temp_df)

    ########## FEATURE END ##########

    return action_feature_df


# get order features
def get_order_feature(data, begin_date, end_date, month_period):

    order_feature_df = data.user_df[['user_id']]

    # filter by date range
    df = data.order_df[(data.order_df.o_date >= begin_date) & (data.order_df.o_date <= end_date)]

    # logical conditions
    cate_30_cond = (df.cate == 30)
    cate_101_cond = (df.cate == 101)
    cate_30_101_cond = (df.cate == 30) | (df.cate == 101)
    cate_others_cond = (df.cate != 30) & (df.cate != 101)

    # feature name prefix
    name = str(month_period) + 'mo_order_'
    name_30 = name + '30_'
    name_101 = name + '101_'
    name_30_101 = name + '30_101_'
    name_others = name + 'others_'
    name_all = name + 'all_'

    ########## FEATURE START ##########

    # order num: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['o_id']. \
                nunique(). \
                reset_index(). \
                rename(columns={'o_id': name_30_101 + 'num'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order num: 30
    temp_df = df[cate_30_cond]. \
                groupby('user_id')['o_id']. \
                nunique(). \
                reset_index(). \
                rename(columns={'o_id': name_30 + 'num'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order num: 101
    temp_df = df[cate_30_cond]. \
                groupby('user_id')['o_id']. \
                nunique(). \
                reset_index(). \
                rename(columns={'o_id': name_101 + 'num'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order num: others
    temp_df = df[cate_others_cond]. \
                groupby('user_id')['o_id']. \
                nunique(). \
                reset_index(). \
                rename(columns={'o_id': name_others + 'num'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order num: all
    temp_df = df. \
                groupby('user_id')['o_id']. \
                nunique(). \
                reset_index(). \
                rename(columns={'o_id': name_all + 'num'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order item count: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['sku_id']. \
                count(). \
                reset_index(). \
                rename(columns={'sku_id': name_30_101 + 'item_count'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order item count: 30
    temp_df = df[cate_30_cond]. \
                groupby('user_id')['sku_id']. \
                count(). \
                reset_index(). \
                rename(columns={'sku_id': name_30 + 'item_count'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order item count: 101
    temp_df = df[cate_101_cond]. \
                groupby('user_id')['sku_id']. \
                count(). \
                reset_index(). \
                rename(columns={'sku_id': name_101 + 'item_count'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order item count: others
    temp_df = df[cate_others_cond]. \
                groupby('user_id')['sku_id']. \
                count(). \
                reset_index(). \
                rename(columns={'sku_id': name_others + 'item_count'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order item num: all
    temp_df = df. \
                groupby('user_id')['sku_id']. \
                nunique(). \
                reset_index(). \
                rename(columns={'sku_id': name_all + 'item_num'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order date num: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['o_date']. \
                nunique(). \
                reset_index(). \
                rename(columns={'o_date': name_30_101 + 'date_num'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order item sum: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['sku_id']. \
                sum(). \
                reset_index(). \
                rename(columns={'sku_id': name_30_101 + 'item_sum'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # first day of order: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['o_day']. \
                min(). \
                reset_index(). \
                rename(columns={'o_day': name_30_101 + 'day_first'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # first day of order: 30
    temp_df = df[cate_30_cond]. \
                groupby('user_id')['o_day']. \
                min(). \
                reset_index(). \
                rename(columns={'o_day': name_30 + 'day_first'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # first day of order: 101
    temp_df = df[cate_101_cond]. \
                groupby('user_id')['o_day']. \
                min(). \
                reset_index(). \
                rename(columns={'o_day': name_101 + 'day_first'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # last day of order: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['o_day']. \
                max(). \
                reset_index(). \
                rename(columns={'o_day': name_30_101 + 'day_last'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # average day of order: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['o_day']. \
                mean(). \
                reset_index(). \
                rename(columns={'o_day': name_30_101 + 'day_mean'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order month num: 30 101
    if month_period > 1:
        temp_df = df[cate_30_101_cond]. \
                    groupby('user_id')['o_month']. \
                    nunique(). \
                    reset_index(). \
                    rename(columns={'o_month': name_30_101 + 'month_num'})
        order_feature_df = merge_df(order_feature_df, temp_df)

    # order date gap variance: 30 101
    valid_date_gap_cond = df[cate_30_101_cond]. \
                            groupby('user_id')['o_date']. \
                            nunique(). \
                            reset_index() \
                            ['o_date'] > 2
    temp_df = df[cate_30_101_cond]. \
                sort_values('o_date'). \
                groupby('user_id')['o_date']. \
                agg(lambda x: x.diff().dt.days.std(skipna=True)). \
                reset_index(). \
                rename(columns={'o_date': name_30_101 + 'date_gap_var'}) \
                [valid_date_gap_cond]
    order_feature_df = merge_df(order_feature_df, temp_df)

    # most frequent order area: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['o_area']. \
                agg(lambda x: x.value_counts().index[0]). \
                reset_index(). \
                rename(columns={'o_area': name_30_101 + 'area_freq'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # total order price: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['o_total_price']. \
                sum(). \
                reset_index(). \
                rename(columns={'o_total_price': name_30_101 + 'consume_total'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # average order date gap: 30 101
    temp_df = df[cate_30_101_cond]. \
                sort_values('o_date'). \
                groupby('user_id')['o_date']. \
                agg(lambda x: x.diff().mean(skipna=True).days). \
                reset_index(). \
                rename(columns={'o_date': name_30_101 + 'date_gap_mean'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # average order date gap: all
    temp_df = df. \
                sort_values('o_date'). \
                groupby('user_id')['o_date']. \
                agg(lambda x: x.diff().mean(skipna=True).days). \
                reset_index(). \
                rename(columns={'o_date': name_all + 'date_gap_mean'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # last month of order: 30 101
    if month_period > 1:
        temp_df = df[cate_30_101_cond]. \
                    groupby('user_id')['o_month']. \
                    max(). \
                    reset_index(). \
                    rename(columns={'o_month': name_30_101 + 'month_last'})
        order_feature_df = merge_df(order_feature_df, temp_df)

    # most frequent order day: 30 101
    temp_df = df[cate_30_101_cond]. \
                groupby('user_id')['o_day']. \
                agg(lambda x: x.value_counts().index[0]). \
                reset_index(). \
                rename(columns={'o_day': name_30_101 + 'day_freq'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # order score count: 30 101
    temp_df = df[(df.score_level > 0) & cate_30_101_cond]. \
                groupby('user_id')['score_level']. \
                count(). \
                reset_index(). \
                rename(columns={'score_level': 'score_count'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    # good order score count: 30 101
    temp_df = df[(df.score_level == 1) & cate_30_101_cond]. \
                groupby('user_id')['score_level']. \
                count(). \
                reset_index(). \
                rename(columns={'score_level': 'good_score_count'})
    order_feature_df = merge_df(order_feature_df, temp_df)

    ########## FEATURE END ##########

    return order_feature_df


# get action-order-combining features
def get_action_order_feature(data, begin_date, end_date, month_period):

    action_order_feature_df = data.user_df[['user_id']]

    # action-order-combining dataframe
    action_order_df = data.action_df[['user_id', 'sku_id', 'cate', 'a_date']]. \
                        merge(data.order_df[['user_id', 'sku_id', 'cate', 'o_date']], on=['user_id', 'sku_id', 'cate'], how='left')
    action_order_df = action_order_df[action_order_df.o_date >= action_order_df.a_date]
    action_order_df['date_gap'] = (action_order_df.o_date - action_order_df.a_date).apply(lambda x: x.days)

    # filter by date range
    df = action_order_df[(action_order_df.o_date >= begin_date) & (action_order_df.o_date <= end_date) & \
            (action_order_df.a_date >= begin_date) & (action_order_df.a_date <= end_date)]

    # logical conditions
    cate_30_cond = (df.cate == 30)
    cate_101_cond = (df.cate == 101)
    cate_30_101_cond = (df.cate == 30) | (df.cate == 101)
    cate_others_cond = (df.cate != 30) & (df.cate != 101)

    # feature name prefix
    name = str(month_period) + 'mo_action_order_'
    name_30 = name + '30_'
    name_101 = name + '101_'
    name_30_101 = name + '30_101_'
    name_others = name + 'others_'
    name_all = name + 'all_'

    ########## FEATURE START ##########

    # average action to order gap: all
    temp_df = df. \
                groupby('user_id')['date_gap']. \
                mean(). \
                reset_index()
    action_order_feature_df = merge_df(action_order_feature_df, temp_df)

    ########## FEATURE END ##########

    return action_order_feature_df


