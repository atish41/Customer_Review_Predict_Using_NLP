# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 16:58:06 2023

@author: ATISHKUMAR
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loadd the text dataset file
dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\august\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

#cleaninf text
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
ps=PorterStemmer()
for i in range(0,1000):
    review=re.sub('^[a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
#creating tf-tdf model
from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#split the dataset into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#training  on naive bayes model
from sklearn.naive_bayes import BernoulliNB
classifier=BernoulliNB()
classifier.fit(x_train,y_train)

#test results
y_pred=classifier.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm

#accuracy score
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
ac