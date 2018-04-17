
# coding: utf-8

# In[2]:


# Disease prediction
# Classification- Random Forest
# Using Age and Gender

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

df = pd.read_csv('/Users/Akther Hossen/Desktop/disease_data.csv')

df=df.fillna('0')

cols = [8,9,10,11,12,13,14,15,16]
df.drop(df.columns[cols],axis=1,inplace=True)

X = df.iloc[:,7:9].values
y = df.iloc[:,-1].values

#taking careof categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
labelencoder_X = LabelEncoder()
X[:,1]= labelencoder_X.fit_transform(X[:,1])
#print(X)
y= labelencoder_y.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 4)

clf = RandomForestClassifier(max_depth=3, random_state=0)
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)

print("Prediction Result: ",predicted)

print("Accuracy: ",accuracy_score(y_test, predicted, normalize = True)) #accuracy_score(train output, predict output)

