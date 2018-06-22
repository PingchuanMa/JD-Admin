import pandas as pd
from datetime import datetime


class DataLoader():

    # data preprocessing and loading
    def __init__(self, data_path):

        # load data and format
        sku_info_df = pd.read_csv(data_path + '/jdata_sku_basic_info.csv')
        user_action_df = pd.read_csv(data_path + '/jdata_user_action.csv')
        user_info_df = pd.read_csv(data_path + '/jdata_user_basic_info.csv')
        user_score_df = pd.read_csv(data_path + '/jdata_user_comment_score.csv')
        user_order_df = pd.read_csv(data_path + '/jdata_user_order.csv')
        user_action_df.a_date = pd.to_datetime(user_action_df.a_date)
        user_score_df.comment_create_tm = pd.to_datetime(user_score_df.comment_create_tm)
        user_order_df.o_date = pd.to_datetime(user_order_df.o_date)

        # user table
        user_df = user_info_df

        # sku table
        sku_df = sku_info_df.drop(['para_1', 'para_2', 'para_3'], axis=1)
        sku_cate_df = sku_df[['sku_id', 'cate']].drop_duplicates()

        # action table
        action_df = user_action_df.merge(sku_cate_df, on='sku_id', how='left')
        action_df['a_month'] = action_df.a_date.dt.month
        action_df['a_day'] = action_df.a_date.dt.day

        # order table
        order_df = user_order_df.merge(sku_cate_df, on='sku_id', how='left')
        score_df = user_score_df.drop('comment_create_tm', axis=1)
        order_df = order_df.merge(score_df, on=['o_id', 'user_id'], how='left').fillna(-1)
        order_df.score_level = order_df.score_level.astype(int)
        order_df['o_month'] = order_df.o_date.dt.month
        order_df['o_day'] = order_df.o_date.dt.day
        sku_price_df = sku_df[['sku_id', 'price']].drop_duplicates()
        order_df = order_df.merge(sku_price_df, on='sku_id', how='left')
        order_df['o_total_price'] = order_df.price.values * order_df.o_sku_num.values
        order_df = order_df.drop('price', axis=1)

        self.user_df = user_df
        self.sku_df = sku_df
        self.action_df = action_df
        self.order_df = order_df
