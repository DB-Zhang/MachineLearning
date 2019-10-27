# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 19:47
# @Author  : Dingbang Zhang
# @FileName: getData.py
# @Software: PyCharm Community Edition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn import preprocessing
dic_color = {"浅白":0,"青绿":1,"乌黑":2}
dic_root = {"硬挺":0,"稍蜷":1,"蜷缩":2}
dic_sensor = {"清脆":0,"沉闷":1,"浊响":2}
dic_vein = {"清晰":2,"模糊":0,"稍糊":1}
dic_shape = {"平坦":0,"稍凹":1,"凹陷":2}
dic_hardness = {"硬滑":1,"软粘":0}
dic_outcome = {"是":1,"否":0}

def get_numpy_data(data_frame,set2):
    """
    use for LDA,SVM,etc. that the algorithms need a scale vector but can not contain stange type data like string,dictionary etc.
    """
    list_data = []
    list_outcome = []
    origin = []
    enc = preprocessing.OneHotEncoder()  
    list_1= []
    
    for index,row in data_frame.iterrows():
        list_data.append([dic_color[row["色泽"]],dic_root[row["根蒂"]],dic_sensor[row["敲声"]],dic_vein[row["纹理"]],dic_shape[row["脐部"]],
                          dic_hardness[row["触感"]]])
        list_1.append([row["密度"],row["含糖率"]])
        list_outcome.append(dic_outcome[row["好瓜"]])
    for index,row in set2.iterrows():
        origin.append([dic_color[row["色泽"]],dic_root[row["根蒂"]],dic_sensor[row["敲声"]],dic_vein[row["纹理"]],dic_shape[row["脐部"]],
                          dic_hardness[row["触感"]]])

    enc.fit(origin)  #列表中代表四个样本
    # print(origin)
    # print(list_data)
    array = enc.transform(list_data).toarray() 
    # list_data = np.array(list_data)
    list_1 = np.array(list_1)
    list_data = np.column_stack((list_data,list_1))
    list_outcome = np.array(list_outcome)

    return (list_data,list_outcome)

def get_origin_data(data_frame):
    """
    use for Naive Bayes,etc. that the algorithms need a discrete vector and can contain stange type data like string,dictionary etc.
    """
    list_data = []
    list_outcome = []
    for index,row in data_frame.iterrows():
        list_data.append([row["色泽"],row["根蒂"],row["敲声"],row["纹理"],row["脐部"],
                          row["触感"],row["密度"],row["含糖率"]])
        list_outcome.append(dic_outcome[row["好瓜"]])
    list_outcome = np.array(list_outcome)
    list_data = np.array(list_data)
    return (list_data,list_outcome)

def get_watermelon_dataset(dataset_all,rate=0.8,shuffle=True):
    """
    In this part, the form of dataset_all is a pd.dataframe. Rate is the proportion of trainset.
    return  train_set: a numpy.array of train samples
            test_set: a numpy.array of test samples
    """
    row_num = dataset_all.shape[0]
    col_num = dataset_all.shape[1]
    
    train_num = rate * row_num
    
    train_set = []
    
    if shuffle:
        tmp_data = dataset_all.sample(frac=1).reset_index(drop=True)
    else:
        tmp_data = dataset_all
    
    train_set = tmp_data[:int(tmp_data.shape[0]*0.8)]
    test_set = tmp_data[int(tmp_data.shape[0]*0.8):]
    
    train_set = get_numpy_data(train_set,dataset_all)
    test_set = get_numpy_data(test_set,dataset_all)
    
    
    return train_set,test_set

def get_iris_dataset(dataset_all,rate=0.8,shuffle=True):
    """
    In this part, the form of dataset_all is a pd.dataframe. Rate is the proportion of trainset.
    The target of the func is to pre-process the IRIS dataset:Iris (http://archive.ics.uci.edu/ml/datasets/Iris)
    train_set: a list of train samples
    test_set: a list of test samples
    """
    row_num = dataset_all.shape[0]
    col_num = dataset_all.shape[1]
    
    train_num = rate * row_num
    
    train_set = []
    
    if shuffle:
        tmp_data = dataset_all.sample(frac=1).reset_index(drop=True)
    else:
        tmp_data = dataset_all
    
    train_set = tmp_data[:int(tmp_data.shape[0]*0.8)]
    test_set = tmp_data[int(tmp_data.shape[0]*0.8):]
    
    train_set = get_numpy_data(train_set,test_set)
    test_set = get_numpy_data(test_set,test_set)
    
    
    return train_set,test_set


class kfold_watermelon_dataset:
    """
    In this part, the form of dataset_all is a pd.dataframe. fold is the k fold train.
    train_set: a list of train samples
    test_set: a list of test samples
    """
    def __init__(self,dataset_all,fold=5,shuffle=True):
        self.row_num = dataset_all.shape[0]
        self.col_num = dataset_all.shape[1]
        self.fold = fold
        if shuffle:
            self.tmp_data = dataset_all.sample(frac=1).reset_index(drop=True)
        else:
            self.tmp_data = dataset_all
        
        self.times = 0
    
    def get_data(self):
        if self.times+1 == self.fold:
            self.times=0

        begin_pos = int(self.times*self.row_num/self.fold)
        stop_pos = min(int((self.times+1)*self.row_num/self.fold+1),self.row_num)

        self.test_set = self.tmp_data[begin_pos:stop_pos]
        havelist= []
        for _, row in self.test_set.iterrows():
            havelist.append(row["编号"])
        self.train_set = self.tmp_data[~self.tmp_data['编号'].isin(havelist)]
        
        train_set = get_origin_data(self.train_set)
        test_set = get_origin_data(self.test_set)
        
        
        return train_set,test_set
    
    def get_data_number(self):
        train_set = get_numpy_data(self.train_set,self.tmp_data)
        test_set = get_numpy_data(self.test_set,self.tmp_data)

        return train_set,test_set

class kfold_Iris_dataset:
    """
    In this part, the form of dataset_all is a pd.dataframe. fold is the k fold train.
    train_set: a list of train samples
    test_set: a list of test samples
    """
    def __init__(self,dataset_all,fold=5,shuffle=True):
        self.row_num = dataset_all.shape[0]
        self.col_num = dataset_all.shape[1]
        self.fold = fold
        if shuffle:
            self.tmp_data = dataset_all.sample(frac=1).reset_index(drop=True)
        else:
            self.tmp_data = dataset_all
        
        self.times = 0
    
    def get_data(self):
        pass