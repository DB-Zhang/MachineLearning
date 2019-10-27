# -*- coding: utf-8 -*-
# @Time    : 2019/10/21 19:47
# @Author  : Dingbang Zhang
# @FileName: part1.py
# @Software: PyCharm Community Edition

from getData import get_iris_dataset,get_watermelon_dataset,get_numpy_data,kfold_watermelon_dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn import preprocessing




parse=argparse.ArgumentParser()

def get_avg_corvariance(Input):
    """
    Input:a matrix of shape(n*l),while n is the number of samples,l is the dimension of each sample; type np.array
    """
    num = Input.shape[0]
    avg = np.sum(Input,axis=0)/num
    covarience = (Input-avg).T.dot((Input-avg))/(num-1)
    
    return avg,covarience

def LDAtrain(train_set):
    """
    train_set:type np.array
    return: weight, the center of class0, the center of class1
    """
    X,Y = train_set
    class0 = X[Y==0]
    class1 = X[Y==1]
    avg_0,covarience_0 = get_avg_corvariance(class0)
    avg_1,covarience_1 = get_avg_corvariance(class1) 
    weight = np.linalg.pinv(covarience_1+covarience_0).dot(avg_0-avg_1)
    
    return weight,(avg_0).dot(weight),(avg_1).dot(weight)

def LDAval(test_set,weight,mid0,mid1):
    """
    test_set:type np.array n*l
    weight: type np.array l*1
    mid0: class 0 center
    mid1: class 1 center
    """
    X,Y = test_set
    class0 = X[Y==0]
    class1 = X[Y==1]
    
    numall = class0.shape[0] + class1.shape[0]
    numcorr = 0
    for item in class0.dot(weight):
        if abs(item-mid0)<abs(item-mid1):
            numcorr += 1
    for item in class1.dot(weight):
        if abs(item-mid1)<abs(item-mid0):
            numcorr += 1
            
    return (numcorr/numall)

class Beyas:
    def __init__(self):
        self.probablity = []
        
    def train_watermelon(self,dataset):
        """
        dataset：type np.array n*l
        return : None
        """
        X,Y = dataset
        length = X.shape[1]
        pos_color = {"浅白":1,"青绿":1,"乌黑":1}
        pos_root = {"硬挺":1,"稍蜷":1,"蜷缩":1}
        pos_sensor = {"清脆":1,"沉闷":1,"浊响":1}
        pos_vein = {"清晰":1,"模糊":1,"稍糊":1}
        pos_shape = {"平坦":1,"稍凹":1,"凹陷":1}
        pos_hardness = {"硬滑":1,"软粘":1}
        pos_list = [pos_color,pos_root,pos_sensor,pos_vein,pos_shape,pos_hardness]
        pos_outcome = {"是":1,"否":0}
#         print(length)  #resule is 8

        self.weight = []
        for num in range(length):
            if num< len(pos_list):
                leng = len(pos_list[num])
                pos = pos_list[num]
                data = np.array(X[:,num])
                
                for name in pos:
                    Y_tmp = np.array(Y)[data==name]
                    vali = Y[data==name]
                    if len(vali)==0:
                        continue#prevent to devide by zero or dont choose this kind of example
                    rate = sum(Y_tmp)/len(Y_tmp)
                    pos[name]=rate
                    
                self.weight.append(pos)
            else:
                data0 = X[:,num][Y==0]
                data0 = np.array([float(i) for i in data0])
                mu0 = np.average(data0)
                sigma0 = np.sum(np.square(data0-mu0))/len(data0)
                
                data1 = X[:,num][Y==1]
                data1 = np.array([float(i) for i in data1])
                mu1 = np.average(data1)
                sigma1 = np.sum(np.square(data1-mu1))/len(data1)
                #print(mu0,sigma0,mu1,sigma1)
                self.weight.append([mu0,sigma0,mu1,sigma1])
            
    def validate(self,testset):
        X,Y = testset
        
        Y_hat = []
        for item in X:
            probablity0 = 1
            probablity1 = 1
            for num in range(len(item)):
                pos = self.weight[num]
                if (type(pos).__name__ == 'dict'):
                    probablity0 *= 1-pos[item[num]]
                    probablity1 *= pos[item[num]]
                else:
                    mu0,sigma0,mu1,sigma1 = pos
                    probablity0 *= 1/(np.sqrt(2*np.pi)*sigma0)*np.exp(-np.square(float(item[num])-mu0)/(2*np.square(sigma0)))
                    probablity1 *= 1/(np.sqrt(2*np.pi)*sigma1)*np.exp(-np.square(float(item[num])-mu1)/(2*np.square(sigma1)))
            if probablity0>probablity1:
                Y_hat.append(0)
            else:
                Y_hat.append(1)
        
        Y_hat = np.array(Y_hat)
        Y = np.array(Y)

        return np.sum(Y_hat==Y)/len(X)



def LDA(datasets):
    """
    using LDA algotithm
    train_set: a numpy.array of train samples
    test_set: a numpy.array of test samples
    """
    train_set,test_set = get_watermelon_dataset(datasets,rate=args.rate)
    weight,mid0,mid1 = LDAtrain(train_set)
    acc = LDAval(test_set,weight,mid0,mid1)
    print("using LDA on watermelon3 dataset,training set rate = ", args.rate,"  test accuracy  :",acc)

def KNB(datasets):
    """
    using K-fold Naive Bayes algotithm
    kfold_watermelon_dataset : a class with which iteration we can get different fold of trainset and testset
    train_set: a numpy.array of train samples, in this func , some of it's Input X's variables' type is string
    test_set: a numpy.array of test samples, in this func , some of it's Input X's variables' type is string
    """
    dataset = kfold_watermelon_dataset(datasets,args.fold,shuffle=True)
    for i in range(args.fold):
        naiveBayes = Beyas()
        train_set, test_set = dataset.get_data()
        naiveBayes.train_watermelon(train_set)
        acc = naiveBayes.validate(test_set)
        print("fold ", i ," accuracy : ", acc)

def SVM_func(datasets ,kernal = 'rbf'):
    train_set,test_set = get_watermelon_dataset(datasets,rate=args.rate)
    X,Y = train_set
    #print(X)
    X_normalized = preprocessing.normalize(X, norm='l2')

    svm_linear = SVC(kernel=kernal)
    svm_linear.fit(X_normalized, Y)
    X_test,Y_test = test_set
    X_test_normlized = preprocessing.normalize(X_test, norm='l2')
    print("the accuracy of SVM using kernal "+kernal+"  scores \t",svm_linear.score(X_test_normlized, Y_test))

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="type LDA or K-NB or SVM, the program will run a LDA or a Naive Bayes or SVM algorithm.")
    parse.add_argument("--rate", type=float, default=0.8, help="the preportion of training set, default is 0.8.")
    parse.add_argument("--fold", type=int, default=5, help="the num of folds in K-flod validation, default is 5.")

    watermalon = pd.read_csv("watermalon.csv",engine="python")
    iris = pd.read_csv("iris.data",header=None)

    args = parse.parse_args()

    if args.action=="LDA":
        LDA(watermalon)
    
    if args.action=="K-NB":
        KNB(watermalon)
            
    if args.action=="SVM":
        SVM_func(watermalon,'rbf')
        SVM_func(watermalon,'linear')
        SVM_func(watermalon,'poly')
        SVM_func(watermalon,'sigmoid')


        
        
