
# coding: utf-8

# In[5]:


# Disease prediction- Using Chief Complaints/naive bayes classifier
# Classification

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


df = pd.read_csv('/Users/Akther Hossen/Desktop/disease_data.csv')

df=df.fillna('0')

X = df.iloc[:,19:36].values

y = df.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
for i in range(0,17):
    X[:,i]= labelencoder_X.fit_transform(X[:,i])

labelencoder_y = LabelEncoder()
y= labelencoder_y.fit_transform(y)

for i in range(1,10):
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = i)

    model = GaussianNB()
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)
    print("Prediction Result: ",predicted)

    print("Accuracy: ",accuracy_score(y_test, predicted, normalize = True)) #accuracy_score(train output, predict output)
    print('\n')


# In[8]:


# Disease prediction
#DecisionTreeRegressor

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import warnings; warnings.simplefilter('ignore')
#importing the dataset
df = pd.read_csv('/Users/Akther Hossen/Desktop/disease_data.csv')

cols = [2,3,4,5,6]
df.drop(df.columns[cols],axis=1,inplace=True)

X = df.iloc[:,1:3].values


y = df.iloc[:,-1].values
print(len(y))


from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
labelencoder_X = LabelEncoder()
X[:,0]= labelencoder_X.fit_transform(X[:,0])
#print(X)
y= labelencoder_y.fit_transform(y)
#print(len(y))


from sklearn.cross_validation import train_test_split

#for i in range(1,10):
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20,random_state = 100)

regressor = DecisionTreeRegressor(max_depth=100)
regressor.fit(X_train,y_train)
predicted = regressor.predict(X_test)
#print("i = ",i)
print('R-squared test score: {:.3f}'.format(regressor.score(X_test,y_test))) 


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")
sns.tsplot(y_test[0:50])
sns.tsplot(predicted[0:50],color="indianred")
plt.xlabel('Number of Observation')
plt.ylabel('n')
plt.title('Disease Prediction')
plt.show()


# In[1]:


# Disease prediction/naive bayes classifier
# Classification

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('/Users/Akther Hossen/Desktop/disease_data.csv')

cols = [2,3,4,5,6]
df.drop(df.columns[cols],axis=1,inplace=True)
X = df.iloc[:,1:3].values
y = df.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
labelencoder_X = LabelEncoder()
X[:,0]= labelencoder_X.fit_transform(X[:,0])
#print(X)
y= labelencoder_y.fit_transform(y)
#print(len(y))

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state = 4)

model = GaussianNB()
model.fit(X_train,y_train)
predicted = model.predict(X_test)
print("Prediction Result: ",predicted)

print("Accuracy: ",accuracy_score(y_test, predicted, normalize = True))

