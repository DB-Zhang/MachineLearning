# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 13:35
# @Author  : Dingbang Zhang
# @FileName: part2.py
# @Software: PyCharm Community Edition


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
random.seed(0)
class Logistic:
    def __init__(self):
        pass

    def train(self,dataset,times,classnum = 3):
        self.classnum = classnum
        X, Y = dataset
        Y = Y.reshape((Y.shape[0],1))
        Y = self.rebuild(Y,classnum)
        One = np.ones((X.shape[0],1))
        X = np.column_stack((X,One))
        self.weight = np.random.rand(X.shape[1],classnum)
        for i in range(times):
            Out = self.forward(X)
            Out = np.exp(Out)/np.sum(np.exp(Out),axis=1,keepdims=True)
            self.weight -= 0.05 * (np.dot(X.T,Y) - X.T.dot(Out))
            
            outcome = Out.argmin(axis=1)
            groundtruth = Y.argmin(axis=1)
#             print(np.sum(outcome==groundtruth)/outcome.shape[0])



    def forward(self,X):
        return 1/(1+np.exp(X.dot(self.weight)))
    
    def rebuild(self,Y,classnum = 3):
        outcome = np.zeros((Y.shape[0],classnum))
        for num in range(Y.shape[0]):
            outcome[num][Y[num]] = 1
            
#         print(outcome)
        return outcome
    
    def validation(self,dataset):
        X, Y = dataset
        Y = Y.reshape((Y.shape[0],1))
        Y = self.rebuild(Y,self.classnum)
        One = np.ones((X.shape[0],1))
        X = np.column_stack((X,One))

        Out = self.forward(X)
        Out = np.exp(Out)/np.sum(np.exp(Out),axis=1,keepdims=True)

        outcome = Out.argmin(axis=1)
        groundtruth = Y.argmin(axis=1)
        #print(np.sum(outcome==groundtruth)/outcome.shape[0])

        return np.sum(outcome==groundtruth)/outcome.shape[0]