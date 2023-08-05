# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 14:14:36 2023

@author: ATISHKUMAR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load the file
dataset=pd.read_csv(r'D:\Naresh_it_praksah_senapathi\august\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

#cleaning text
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
ps=PorterStemmer()
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
#creating bag of word model
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

#splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#training naive bayes model on training dataset
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#test the results
y_pred=classifier.predict(x_test)

#making confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm

#accuracy score
from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test,y_pred)
ac