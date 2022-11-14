#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   date_feats.py
@Time    :   2022/10/28 14:12:45
@Author  :   LiLiangshan
@Email   :   lilso@suning.com
@Copyright : 侵权必究
@Describe :  数据预处理模块-时间类特征衍生
'''

from time import time
import numpy as np 
import pandas as pd
from sklearn.preprocessing import OneHotEncoder 


"""
1. 特征衍生，时间类特征衍生函数
"""


def get_date_cycle_base(dt, col, n, one_hot=False, one_hot_drop=True):
    """
    功能:
    1.构造时间周期特征
    
    输入:
    dt: DataFrame, 衍生前的数据集
    col: str, 需要衍生的字段
    n: int, 时间周期
    one_hot: bool, 是否要one_hot
    drop: bool, 是否要删除特征
    
    输出:
    dt: DataFrame，衍生后的数据集
    
    维护记录:
    1. edited by 黎良山 2022/10/28
    """
    dt['{}_sin'.format(col)] = round(np.sin(2*np.pi / n * dt[col]), 6)
    dt['{}_cos'.format(col)] = round(np.cos(2*np.pi / n * dt[col]), 6)
    if one_hot:
        ohe = OneHotEncoder()
        X = OneHotEncoder().fit_transform(dt[col])
        df = pd.DataFrame(X, columns=[col + '_' + str(int(i)) for i in range(X.shape[1])])
        dt = pd.concat([dt, df], axis=1)
        if one_hot_drop:
            dt = dt.drop(col, axis=1)
    return dt 




def get_time_base(dt, cols, prefix=None, one_hot=False, one_hot_drop=True):
    """
    功能: 离散时间特征衍生
    
    输入:
    dt: DataFrame, 衍生前的数据集
    cols: str/list/None, 需要进行时间衍生的字段列表
    prefix: str/list/None, 衍生后特征名的前缀，默认为None，此时用输入cols作为前缀，否则必须提供所有cols的前缀
    one_hot: bool, 是否要对原始数据做one_hot 
    one_hot_drop: bool, 若对原始数据做one_hot，是否要删除原始数据
    
    输出:
    dt: DataFrame，衍生后的数据集
    
    维护记录:
    1. edited by 黎良山 2022/10/28
    """
    assert isinstance(dt, pd.DataFrame), 'dt must be DataFrame type'

    if isinstance(cols, str):
        cols = [cols]
        if prefix is None:
            prefix=cols 
        elif len(prefix)!=len(cols):
            raise ValueError("please check prefix, prefix must be None or list")
    elif isinstance(cols, list):
        cols = cols
        if prefix is None:
            prefix=cols 
        elif len(prefix)!=len(cols):
            raise ValueError("please check prefix, prefix must be None or list")
    else:
        raise ValueError("please check cols type, cols must be str or list")


    for i_prefix, col in zip(prefix, cols):
        dt[col] = pd.to_datetime(dt[col])

        dt['{}_year'.format(i_prefix)] = dt[col].dt.year              # 抽取出年
        dt['{}_month'.format(i_prefix)] = dt[col].dt.month            # 抽取出月
        dt['{}_day'.format(i_prefix)] = dt[col].dt.day                # 抽取出日
        dt['{}_hour'.format(i_prefix)] = dt[col].dt.hour              # 抽取出时
        dt['{}_minute'.format(i_prefix)] = dt[col].dt.minute          # 抽取出分
        dt['{}_second'.format(i_prefix)] = dt[col].dt.second          # 抽取出秒
        dt['{}_quarter'.format(i_prefix)] = dt[col].dt.quarter        # 抽取出季度
        dt['{}_dayofweek'.format(i_prefix)] = dt[col].dt.dayofweek    # 抽取出一周的第几天
        #dt['{}_weekofyear'.format(i_prefix)] = dt[col].dt.week        # 抽取出一年的第几周

        dt['{}_is_year_start'.format(i_prefix)] = dt[col].dt.is_year_start    # 判断日期是否为当年的第一天
        dt['{}_is_year_start'.format(i_prefix)] = dt[col].dt.is_year_end      # 判断日期是否为当年的最后一天
        dt['{}_is_month_start'.format(i_prefix)] = dt[col].dt.is_month_start  # 判断日期是否为当月的第一天
        dt['{}_is_month_end'.format(i_prefix)] = dt[col].dt.is_month_end      # 判断日期是否为当月的最后一天

        
        get_date_cycle_base(dt, '{}_second'.format(i_prefix), n=60, one_hot=one_hot, one_hot_drop=one_hot_drop)
        get_date_cycle_base(dt, '{}_minute'.format(i_prefix), n=60, one_hot=one_hot, one_hot_drop=one_hot_drop)
        get_date_cycle_base(dt, '{}_hour'.format(i_prefix), n=24, one_hot=one_hot, one_hot_drop=one_hot_drop)
        get_date_cycle_base(dt, '{}_day'.format(i_prefix), n=31, one_hot=one_hot, one_hot_drop=one_hot_drop)
        get_date_cycle_base(dt, '{}_dayofweek'.format(i_prefix), n=7, one_hot=one_hot, one_hot_drop=one_hot_drop)
        get_date_cycle_base(dt, '{}_month'.format(i_prefix), n=12, one_hot=one_hot, one_hot_drop=one_hot_drop)

    return dt 



def get_sequence_statis(dt, col, n, freq=3):
    """
    功能:基于历史数据构造长中期统计值

    输入:
    dt: DataFrame, 衍生前的数据集
    col: str, 统计数据的序列
    n: int, 共产生n个特征,取前(n+1)*freq条记录，统计特征
    freq: int, 间隔每个特征间隔n条记录
    
    输出:
    dt: DataFrame，衍生后的数据集
    
    维护记录:
    1. edited by 黎良山 2022/10/28
    """
    for i in range(1, n+1):
        tmp = dt[col].rolling(i*freq)
        dt['avg_{}_{}'.format(i*freq, col)] = tmp.mean()
        dt['median_{}_{}'.format(i*freq, col)] = tmp.median()
        dt['max_{}_{}'.format(i*freq, col)] = tmp.max()
        dt['min_{}_{}'.format(i*freq, col)] = tmp.min()
        dt['std_{}_{}'.format(i*freq, col)] = tmp.std()
        dt['skew_{}_{}'.format(i*freq, col)] = tmp.skew()
        dt['kurt_{}_{}'.format(i*freq, col)] = tmp.kurt()

    return dt 


def get_sequence_groupby_statis(dt, col, cate_cols, n, freq=3):
    """
    功能:
    1.基于历史数据与类别特征交叉构造长中期统计值

    输入:
    dt: DataFrame, 衍生前的数据集
    col: str, 统计数据的序列
    cate_cols: str/list, 用于交叉的特征列表
    n: int, 共产生n个特征,取前(n+1)*freq条记录，统计特征
    freq: int, 间隔每个特征间隔n条记录
    
    输出:
    dt: DataFrame，衍生后的数据集
    
    维护记录:
    1. edited by 黎良山 2022/10/28
    """
    for i in cate_cols:
        for j in range(1, n+1):
            tmp = dt.groupby([i], as_index=False)[col].rolling(j*freq)
            dt['avg_{}_{}_{}'.format(i,col, j*freq)] = tmp.mean()[col]
            dt['median_{}_{}_{}'.format(i,col, j*freq)] = tmp.median()[col]
            dt['max_{}_{}_{}'.format(i,col, j*freq)] = tmp.max()[col]
            dt['min_{}_{}_{}'.format(i,col, j*freq)] = tmp.min()[col]
            dt['std_{}_{}_{}'.format(i,col, j*freq)] = tmp.std()[col]
            dt['skew_{}_{}_{}'.format(i,col, j*freq)] = tmp.skew()[col]
            dt['kurt_{}_{}_{}'.format(i,col, j*freq)] = tmp.kurt()[col]
    
    return dt





