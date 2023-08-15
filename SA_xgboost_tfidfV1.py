# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:01:05 2023

@author: ATISHKUMAR
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\august\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

#clean the text
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
ps=PorterStemmer()
for i in range(0,1000):
    review=re.sub('[^a-zA-z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word)for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)

#apply bag of words model
from sklearn.feature_extraction.text import TfidfVectorizer
cv=TfidfVectorizer()
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#split into training and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0) 

import xgboost as xgb
#apply the xgboost ensamble technique to traingn set
#from sklearn.ensemble import xgboost
classifier=xgb.XGBClassifier()
classifier.fit(x_train,y_train)

#test the results
y_pred=classifier.predict(x_test)


#draw a confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm

#accuracy
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
ac
