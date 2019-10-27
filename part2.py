# -*- coding: utf-8 -*-
# @Time    : 2019/10/22 00:17
# @Author  : Dingbang Zhang
# @FileName: part2.py
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
from part1 import get_avg_corvariance,LDAtrain,LDAval
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from logistic import Logistic

def LDAtrain_iris(train_set):
    """
    train_set:type np.array
    return: weight, the center of class0, the center of class1,the center of class2
    """
    X,Y = train_set
    class0 = X[Y==0]
    class1 = X[Y==1]
    class2 = X[Y==2]
    avg_0,covarience_0 = get_avg_corvariance(class0)
    avg_1,covarience_1 = get_avg_corvariance(class1) 
    avg_2,covarience_2 = get_avg_corvariance(class2) 
    weight = np.linalg.pinv(covarience_1+covarience_0+covarience_2).dot((avg_0-avg_1)+0.5*(avg_1-avg_2)+0.5*(avg_2-avg_0))
    
    return weight,(avg_0).dot(weight),(avg_1).dot(weight),(avg_2).dot(weight)

def LDAval_iris(test_set,weight,mid0,mid1,mid2):
    """
    test_set:type np.array n*l
    weight: type np.array l*1
    mid0: class 0 center
    mid1: class 1 center
    mid2: class 2 center
    """
    X,Y = test_set
    class0 = X[Y==0]
    class1 = X[Y==1]
    class2 = X[Y==2]
    
    numall = X.shape[0]
    numcorr = 0
    for item in class0.dot(weight):
        if abs(item-mid0)<abs(item-mid1) and (abs(item-mid0)<abs(item-mid2)):
            numcorr += 1
    for item in class1.dot(weight):
        if abs(item-mid1)<abs(item-mid0) and (abs(item-mid1)<abs(item-mid2)):
            numcorr += 1
    for item in class2.dot(weight):
        if abs(item-mid2)<abs(item-mid0) and (abs(item-mid2)<abs(item-mid1)):
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

    def train_Iris(self,dataset):
        """
        dataset：type np.array n*l
        return : None
        """
        X,Y = dataset
        length = X.shape[1]
#         print(length)  #resule is 8

        self.weight1 = []
        for num in range(length):
            data0 = X[:,num][Y==0]
            data0 = np.array([float(i) for i in data0])
            mu0 = np.average(data0)
            sigma0 = np.sum(np.square(data0-mu0))/len(data0)
            
            data1 = X[:,num][Y==1]
            data1 = np.array([float(i) for i in data1])
            mu1 = np.average(data1)
            sigma1 = np.sum(np.square(data1-mu1))/len(data1)
            
            data2 = X[:,num][Y==2]
            data2 = np.array([float(i) for i in data2])
            mu2 = np.average(data2)
            sigma2 = np.sum(np.square(data2-mu2))/len(data2)
            

            #print(mu0,sigma0,mu1,sigma1)
            self.weight1.append([mu0,sigma0,mu1,sigma1,mu2,sigma2])
            
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

    def validate_Iris(self,testset):
        X,Y = testset
        
        Y_hat = []
        for item in X:
            probablity0 = 1
            probablity1 = 1
            probablity2 = 1
            # print(self.weight1)
            for num in range(len(item)):
                
                mu0,sigma0,mu1,sigma1,mu2,sigma2 = self.weight1[num]
                probablity0 *= 1/(np.sqrt(2*np.pi)*sigma0)*np.exp(-np.square(float(item[num])-mu0)/(2*np.square(sigma0)))
                probablity1 *= 1/(np.sqrt(2*np.pi)*sigma1)*np.exp(-np.square(float(item[num])-mu1)/(2*np.square(sigma1)))
                probablity2 *= 1/(np.sqrt(2*np.pi)*sigma2)*np.exp(-np.square(float(item[num])-mu2)/(2*np.square(sigma2)))
            porbalis = [probablity0,probablity1,probablity2]
            Y_hat.append((porbalis.index(max(porbalis))))
        
        Y_hat = np.array(Y_hat)
        Y = np.array(Y)

        return np.sum(Y_hat==Y)/len(X)

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="watermelon or Iris, the program will run algorithm on this two dataset.")
    # parse.add_argument("--rate", type=float, default=0.8, help="the preportion of training set, default is 0.8.")
    parse.add_argument("--fold", type=int, default=5, help="the num of folds in K-flod validation, default is 5.")

    args = parse.parse_args()
    
    if args.action=="watermelon":
        watermalon = pd.read_csv("watermalon.csv",engine="python")
        dataset = kfold_watermelon_dataset(watermalon,args.fold,shuffle=True)
        Bayesout = []
        LDAout = []
        SVMout = []
        LogisticRg = []

        for i in range(args.fold):
            naiveBayes = Beyas()
            train_set, test_set = dataset.get_data()
            # print(train_set)
            # naive beyas
            naiveBayes.train_watermelon(train_set)
            acc = naiveBayes.validate(test_set)
            Bayesout.append(acc)
            # naive LDA
            train_set, test_set = dataset.get_data_number()
            weight,mid0,mid1 = LDAtrain(train_set)
            acc = LDAval(test_set,weight,mid0,mid1)
            LDAout.append(acc)
            # naive SVM
            train_x,train_y = train_set
            test_x,test_y = test_set
            X_normalized = preprocessing.normalize(train_x, norm='l2')
            svm_linear = SVC()
            svm_linear.fit(X_normalized, train_y)
            X_test_normlized = preprocessing.normalize(test_x, norm='l2')
            SVMout.append(svm_linear.score(X_test_normlized, test_y))
            #Logistic
            logi = Logistic()
            logi.train((train_x,train_y),100,classnum=2)
            LogisticRg.append(logi.validation((test_x,test_y)))

        print("using Naive Beyas on watermelon with ",args.fold,"flods accuracy :",Bayesout)
        print("using LDA on watermelon with ",args.fold,"flods accuracy :",LDAout)
        print("using SVM on watermelon with ",args.fold,"flods accuracy :",SVMout)
        print("using LogisticRegression on Iris with ",args.fold,"flods accuracy :",LogisticRg)

    if args.action=="Iris":
        iris = datasets.load_iris()
        folders = StratifiedKFold(n_splits=args.fold,random_state=0,shuffle=True)
        X_train, _, y_train, _ = train_test_split(iris.data, iris.target, test_size=0, random_state=0)
        Bayesout = []
        LDAout = []
        SVMout = []
        LogisticRg = []
        for train_index, test_index in folders.split(X_train,y_train):
            train_x = X_train[train_index]
            train_y = y_train[train_index]
            test_x = X_train[test_index]
            test_y = y_train[test_index]
            #beyas
            bayes = Beyas()
            bayes.train_Iris((train_x,train_y))
            Bayesout.append(bayes.validate_Iris((test_x,test_y)))
            #LDA
            weight, mid0, mid1, mid2 = LDAtrain_iris((train_x,train_y))
            LDAout.append(LDAval_iris((test_x,test_y),weight,mid0,mid1,mid2))
            #SVM
            X_normalized = preprocessing.normalize(train_x, norm='l2')
            svm_linear = SVC()
            svm_linear.fit(X_normalized, train_y)
            X_test_normlized = preprocessing.normalize(test_x, norm='l2')
            SVMout.append(svm_linear.score(X_test_normlized, test_y))
            #Logistic
            logi = Logistic()
            logi.train((train_x,train_y),100)
            LogisticRg.append(logi.validation((test_x,test_y)))


        print("using Naive Beyas on Iris with ",args.fold,"flods accuracy :",Bayesout)
        print("using LDA on Iris with ",args.fold,"flods accuracy :",LDAout)
        print("using SVM on Iris with ",args.fold,"flods accuracy :",SVMout)
        print("using LogisticRegression on Iris with ",args.fold,"flods accuracy :",LogisticRg)
