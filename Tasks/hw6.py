import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets, svm
import pandas as pd
import io
import requests
# url="http://www.amlbook.com/data/zip/features.train"
# s=requests.get(url).content
# c=pd.read_csv(io.StringIO(s.decode('utf-8')))
cols=['A', 'B',"C"]
def ReadData(url):
    data = pd.read_csv(url,header=None,names=cols, sep='\s+')
    data=pd.DataFrame(data)
    Xx=np.array(data)
    y=[d[0] for d in Xx]
    X=[d[1:] for d in Xx]
    return X,y

def ExtractDigit1(X,y,digit):
    X_d,y_d=[],[]
    for i in range(len(y)):
        X_d.append(X[i])
        if y[i]==digit:
            y_d.append(1)
        else:
            y_d.append(-1)
    return  X_d,y_d

def ExtractDigit2(X,y,digit1,digit2):
    X_d,y_d=[],[]
    for i in range(len(y)):
        
        if y[i]==digit1:
            X_d.append(X[i])
            y_d.append(1)

        if y[i]==digit2:
            X_d.append(X[i])
            y_d.append(-1)
    return  X_d,y_d

C_s = 10**np.linspace(-4,0, num=5)
X_train,y_train=ReadData("http://www.amlbook.com/data/zip/features.train")
X_test,y_test=ReadData("http://www.amlbook.com/data/zip/features.test")
scores = list()
scores_std = list()
# ex4-ex5
for digit in range(10):
    X_d,y_d=ExtractDigit1(X_train,y_train,digit)
    svc = svm.SVC(C=0.01,kernel='poly',degree=2)   
    svc.fit(X_d,y_d)
    print(digit,1.0-svc.score(X_d,y_d))
    #     this_scores = cross_val_score(svc, X, y, cv=5, n_jobs=1)
    #     print(C,np.mean(this_scores))

# ex6

for C in C_s:
    X_d,y_d=ExtractDigit2(X_train,y_train,1,5)
    X_k,y_k=ExtractDigit2(X_test,y_test,1,5)
    svc = svm.SVC(C,kernel='poly',degree=2)   
    svc.fit(X_d,y_d)
    print(C,1.0-svc.score(X_d,y_d),1.0-svc.score(X_k,y_k))
    #     this_scores = cross_val_score(svc, X, y, cv=5, n_jobs=1)
    #     print(C,np.mean(this_scores))

# ex6


