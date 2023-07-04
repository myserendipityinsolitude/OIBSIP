#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn import datasets
import pandas as pd
import numpy as np
iris = datasets.load_iris()


# In[4]:


iris.keys()


# In[5]:


ris = pd.read_csv("iris.csv")
print(ris)


# In[6]:


iris = ris
iris = pd.DataFrame(iris)
iris


# In[7]:


iris.groupby('Species')
iris


# In[8]:


iris.groupby('Species').size()


# In[9]:


iris.describe()


# In[10]:


import matplotlib.pyplot as plt
x = plt.boxplot(iris['SepalLengthCm'])


# In[11]:


import seaborn as sns
sns.countplot(x='Species',data=iris)


# In[12]:


setosa = iris[iris.Species == "Iris-setosa"]
versicolor = iris[iris.Species=='Iris-versicolor']
virginica = iris[iris.Species=='Iris-virginica']
fig, ax = plt.subplots()
fig.set_size_inches(13, 7) 
ax.scatter(setosa['PetalLengthCm'], setosa['PetalWidthCm'], label="Setosa", facecolor="blue")
ax.scatter(versicolor['PetalLengthCm'], versicolor['PetalWidthCm'], label="Versicolor", facecolor="green")
ax.scatter(virginica['PetalLengthCm'], virginica['PetalWidthCm'], label="Virginica", facecolor="red")

ax.set_xlabel("petal length (cm)")
ax.set_ylabel("petal width (cm)")
ax.grid()
ax.set_title("Iris petals")
ax.legend()


# In[13]:


sns.pairplot(iris, hue = "Species")


# Prediction starts here.....

# In[17]:


df = pd.read_csv("Iris.csv")
df.corr()


# In[20]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])


# In[23]:


from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.300)


# In[24]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)


# In[25]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[26]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[27]:


model.fit(x_train, y_train)


# In[28]:


print("Accuracy: ",model.score(x_test, y_test) * 100)


# In[36]:


from sklearn import metrics
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x_train,y_train)
training_prediction = log_reg.predict(x_train)

test_prediction = log_reg.predict(x_test)

print("Precision, Recall, Confusion matrix, in training\n")
print(metrics.confusion_matrix(y_test, test_prediction))


# In[ ]:




