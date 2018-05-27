# JD-Admin
JDATA Competition.

## Benchmark

0.2236 (S1: 0.3306, S2: 0.1523)

## Method

### Feature Engineering

1. user_basic_info表不作处理，存为**user_df**。

   sku_basic_info表中，将cate属性进行映射（30->1, 101->2, 其他->0），将price与para_1这两个连续值按百分比分别划分为五个band，属性值改为band的index（0-4），存为**sku_df**。

   user_action表不作处理，存为**action_df**。

   user_comment_score表与user_order表合并，可删除o_id冗余属性，额外再删除订单地区o_area与件数o_sku_num两个属性，订单未评分的score_level设为-1，存为**order_df**。

2. 对商品sku_id进行分类，分为四类（有交叉）：品类cate为30（sku_30_id），cate为101（sku_101_id），cate为30或101（sku_tg_id），cate非30或101（sku_ntg_id）。

   > 命名中的tg指target，ntg指非target。

3. 特征形式：在一段时间内（起始时间begin_date，终止时间end_date）对特定品类的商品（cate_types），用户对所有该类商品总和的浏览数（view_num）、关注数（star_num）、订单数（order_num）、好评数（good_score_num）、差评数（bad_score_num）。

   > 特征还有很多可以加，比如存在订单的天数。以及目前还没想好如何在特征里加入商品的属性信息，如para_1。

4. label形式：在一段时间内（起始时间begin_date，终止时间end_date）对特定品类的商品（cate_types），用户对所有该类商品总和的最早订单日期（pred_date）、订单数（order_num），以及在其中筛选出满足特征信息（feature_df）中也存在的用户，作为该特征信息的label。

   > 最早订单日期在label里其实存储形式是距离起始时间的天数，最后提交数据时会进行转换。
   >
   > 特征就是训练用的feature（x），label就是该特征对应的输出（y）。
   >
   > 筛选出来特征信息中也存在的用户是为了训练需要，特征与label在每一行上可以根据用户id一一对应。label的起始时间与终止时间间隔我一般设为一个月，为了与最后预测5月情况相对应，但是这样就导致用户很少，而且筛选完就更少，远不及50000条要求的输出，这点需要再仔细考虑！（最高优先级）

5. 通过label得到提交数据形式的方法：将label信息（label_df）的行按照订单数从大到小排序，然后删除订单数属性这一列，将最早订单日期晚于终止时间的日期改为一个月的中点日期。

   > 日期处理这部分有些hack，需要plausible的改进。

### Model Training

选用朴素的随机森林分类器，没调参（因为自动调参工具GridSearchCV不支持multiclass-multioutput形式的训练，这里我们的label是个multioutput）。

训练用的feature是从时间节点T之前六个月的记录里选取，label是从T未来一个月的记录里选取。T在2017年2.1，3.1，4.1里选取。时间range选取上可以再进行优化，比如baseline里是选了之前一个月、三个月、六个月的feature结合作为整体feature。而且当前没有用online learning的模式，对时序属性也没太多关注，只是简单的把不同时间点T对应的feature与label叠加作为整体的feature与label一次训练。预测时取5.1之前六个月的feature，预测5.1到5.31的label信息。