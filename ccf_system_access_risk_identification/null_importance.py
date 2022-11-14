# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@File    :   null_importance.py
@Time    :   2021/12/15 09:19:49
@Author  :   Liangshan Li
@Version :   1.0
@Contact :   isliliangshan@163.com
@Desc    :   Null Importance筛选特征
"""

import pandas as pd
import numpy as np 

import lightgbm as lgb


"""
该脚本实现了Null Importance功能，用于去除无效特征,
该脚本基于lightgbm版本，若想要使用其它模型或特征筛选,
请自定义get_feature_importance部分

"""


def get_random_state(random_state=None):
    """
    获取随机数
    
    @param  :
    -------
    random_state: int or None
             自定义随机数种子
    
    @Returns  :
    -------
    random_state: int
             随机数
    """
    if random_state is None:
        random_state = np.random.RandomState()
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    else:
        assert isinstance(random_state, np.random.RandomState)
    return random_state


def get_feature_importance(X, y, categorical_feats, shuffle, seed=None):
    """
    该函数用于产生特征重要性, 
    可自定义该函数，需要保证输入输出一致
    
    @param  :
    -------
    X: DataFrame
        训练特征

    y: Series
        训练目标

    shuffle: bool
        是否打乱

    seed: int
        随机数

    @Returns  :
    -------
    imp_df: DataFrame
        特征重要性
    """
    
    if shuffle:
        y = y.copy().sample(frac=1.0)


    dtrain = lgb.Dataset(X, y, categorical_feature=categorical_feats, free_raw_data=False, silent=True)

    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf':3,
        'verbose': -1,
        'seed': seed,
        'force_col_wise':True
    }

    clf = lgb.train(
                params=lgb_params, 
                train_set=dtrain, 
                num_boost_round=200, 
                categorical_feature=categorical_feats
                )

    imp_df = pd.DataFrame()
    imp_df['feature'] = list(X.columns)
    imp_df['importance'] = clf.feature_importance(importance_type='split')

    return imp_df


def get_imp_df(X, y, categorical_feats, act_nb_runs=1, nb_runs=80, func=None):
    """
    获取actual_imp_df及null_imp_df
    
    @param  :
    -------
    X: DataFrame
        训练特征

    y: Series
        训练目标

    act_nb_runs: int
        使用实际target训练次数

    nb_runs: int
        使用打乱的target训练次数

    func: funcation
        用于得到特征重要性的函数
    
    @Returns  :
    -------
    actual_imp_df: DataFrame
        特征重要性
    
    null_imp_df: DataFrame
        特征重要性
    """
    if func is None:
        func = get_feature_importance

    act_imp_df = pd.DataFrame()
    for i in range(act_nb_runs):
        imp_df = func(X, y, categorical_feats, shuffle=False, seed=i)
        imp_df['run'] = i + 1
        
        act_imp_df = pd.concat([act_imp_df, imp_df], axis=0)

    null_imp_df = pd.DataFrame()
    for i in range(nb_runs):
        imp_df = func(X, y, categorical_feats, shuffle=True, seed=i)
        imp_df['run'] = i + 1

        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)

    return act_imp_df, null_imp_df


def calc_imp_score(act_imp_df, null_imp_df, categorical_feats, thresholds=10):
    """
    计算特征重要性得分
    
    @param  :
    -------
    actual_imp_df: DataFrame
        特征重要性
    
    null_imp_df: DataFrame
        特征重要性

    thresholds: int
        过滤特征得分阈值，范围为0-100
    
    @Returns  :
    -------
    feats: list
        过滤后的特征
    """
    correlation_scores = []
    for _f in act_imp_df['feature'].unique():
        f_null_imps = null_imp_df.loc[null_imp_df['feature'] == _f, 'importance'].values
        f_act_imps  = act_imp_df.loc[act_imp_df['feature'] == _f, 'importance'].mean()

        imps_score = 100 * (f_null_imps < np.percentile(f_act_imps, 25)).sum()/f_null_imps.size
        correlation_scores.append((_f, imps_score))

    feats = [_f for _f, _score in correlation_scores if _score >= thresholds]
    categorical_feats = [_f for _f, _score in correlation_scores if (_score >= thresholds)&(_f in categorical_feats)]
    return feats, categorical_feats 


def get_null_importance(X, y, thresholds=10, act_nb_runs=1, nb_runs=80):
    """
    null importance筛选特征
    去除特征得分不超过thresholds的特征
    
    @param  :
    -------
    X: DataFrame
        训练特征

    y: Series
        训练目标

    thresholds: int
        特征过滤阈值，默认为40，范围0-100

    act_nb_runs: int
        使用实际target训练次数，默认为80

    nb_runs: int
        使用打乱的target训练次数，默认为1
    
    @Returns  :
    -------
    feats: int
        过滤后的特征
    """

    categorical_feats = [f for f in X.columns if X[f].dtype == 'object']

    for f_ in categorical_feats:
        X[f_], _ = pd.factorize(X[f_])
        X[f_] = X[f_].astype('category')


    act_imp_df, null_imp_df = get_imp_df(X, y, categorical_feats, act_nb_runs, nb_runs)

    new_feats, new_categorical_feats = calc_imp_score(act_imp_df, null_imp_df, categorical_feats, thresholds)
    return new_feats, new_categorical_feats



    
    

    
    



    
    
    
    
