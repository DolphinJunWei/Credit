#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


df= pd.read_csv("C:/Users/5193/Desktop/AI in Accounting and Finance/Individual Assignment/Credit Card Default II (balance).csv")


# In[3]:


print(df)


# In[4]:


df.describe()


# In[5]:


df.isna().sum()


# In[6]:


df.dropna()


# In[7]:


#remove not a number
for i in df.columns:
    df1 = pd.to_numeric(df[i], errors='coerce')
    df=df[df1.notnull()] #make it to null then remove null
print(df)


# In[8]:


import numpy as np
from scipy import stats


# In[9]:


z = stats.zscore(df.astype(np.float)) #zscore conversion need float
z = np.abs(z) #convert all to positive the parity is not important
f = (z < 3).all(axis=1) #3 is your choice, axis because=1 means by columns, f is a flag
df = df[f]


# In[10]:


print(df)


# In[11]:


Y = df.loc[:,["default"]]


# In[12]:


X = df.iloc[:, 0:3]


# In[13]:


print(Y, X)


# In[14]:


#split train test

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
print(X_train, X_test, Y_train, Y_test)


# In[15]:


from sklearn import linear_model
model = linear_model.LogisticRegression()
model.fit(X_train, Y_train)
pred = model.predict(X_train)


# In[16]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)


# In[17]:


pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)


# In[18]:


from sklearn import tree


# In[19]:


from sklearn import tree
model = tree.DecisionTreeClassifier(max_depth=3)
model.fit(X_train, Y_train)
pred = model.predict(X_train)


# In[20]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(pred, Y_train)
print(cm)
accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)


# In[21]:


pred = model.predict(X_test)
cm = confusion_matrix(pred, Y_test)
print(cm)
accuracy = (cm[0,0]+cm[1,1])/sum(sum(cm))
print(accuracy)


# In[22]:


import matplotlib.pyplot as plt

plt.subplots(figsize=(20, 10))
tree.plot_tree(model, fontsize=10)


# In[23]:


import math
from sklearn.model_selection import GridSearchCV

model = tree.DecisionTreeClassifier()
grid = GridSearchCV(estimator = model, param_grid = dict(max_depth = [i for i in range(1, 20)]))
grid_results = grid.fit(X, Y)
grid_results.best_params_


# In[24]:


import math
from sklearn.model_selection import GridSearchCV

model = tree.DecisionTreeClassifier()
grid = GridSearchCV(estimator = model, param_grid = dict( min_samples_split = [i for i in range(3, 20)]))
grid_results = grid.fit(X, Y)
grid_results.best_params_


# In[25]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import ensemble
model = ensemble.RandomForestClassifier(max_depth=3)
model.fit(X_train, Y_train)
pred = model.predict(X_train)
cm= confusion_matrix(Y_train, pred)
print(cm)
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))


# In[26]:


pred = model.predict(X_test)
cm= confusion_matrix(Y_test, pred)
print(cm)
print((cm[0,0]+cm[1,1])/(sum(sum(cm))))


# In[27]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(max_depth=3)
model.fit(X_train, Y_train)
pred = model.predict(X_train)
cm = confusion_matrix(Y_train, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)


# In[28]:


pred = model.predict(X_test)
cm = confusion_matrix(Y_test, pred)
print(cm)
accuracy = (cm[0,0] + cm[1,1])/sum(sum(cm))
print(accuracy)


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()
model.add(Dense(5, input_dim=3, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))
model.summary()


# In[ ]:


model.compile(loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, batch_size = 10, epochs= 10)


# In[ ]:


model.evaluate(X_test,Y_test)


# In[32]:


model.save("Credit")


# In[33]:


from flask import Flask


# In[34]:


app = Flask(__name__)


# In[35]:


from flask import request, render_template
from keras.models import load_model

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        income= request.form.get("income")
        age= request.form.get("age")
        loan= request.form.get("loan")
        print(income, age, loan)
        model = load_model("Credit")
        pred = model.predict([[float(income),float(age), float(loan)]])
        print(pred)
        s= "The predicted default probability is :" + str(pred)
        return(render_template("index.html", result=s))
    else:
        return(render_template("index.html", result="2"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




