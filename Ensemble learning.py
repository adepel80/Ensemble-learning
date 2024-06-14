#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
plt.style.use ("dark_background")

import os


# In[2]:


# k-nearestNeighbors(KNN)
from sklearn.datasets import load_iris
iris = load_iris()
X= iris.data
y= iris.target


# In[3]:


from sklearn.neighbors import KNeighborsClassifier
knn  = KNeighborsClassifier(n_neighbors=1)


# In[4]:


knn.fit(X,y)


# In[5]:


y_pred=knn.predict(X)


# In[6]:


from sklearn.metrics import accuracy_score
accuracy_score(y,y_pred)


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0, test_size= 0.5)


# In[8]:


from sklearn.neighbors import KNeighborsClassifier
knn  = KNeighborsClassifier(n_neighbors=1)


# In[9]:


knn.fit(X_train, y_train)


# In[10]:


y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)


# In[11]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=knn.classes_.tolist()))


# In[12]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report
#assuming 'knn' is your trained model, 'X_test' are your test features
predictions = knn.predict(X_test)
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
disp.plot()

plt.suptitle("Confusion Matrix for Iris Dataset")
plt.show()


# In[13]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth = 2, min_samples_leaf=4, random_state=42)
dt=  dt.fit(X_train, y_train)
#evaluate the model on the scond set of data
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)


# In[14]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=dt.classes_.tolist()))


# In[15]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report
#assuming 'knn' is your trained model, 'X_test' are your test features
predictions = dt.predict(X_test)
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
disp.plot()

plt.suptitle("Confusion Matrix for Iris Dataset")
plt.show()


# In[16]:


#bagging algorithm
from sklearn.ensemble import BaggingClassifier
BaggingClassifier


# In[17]:



from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
bag = BaggingClassifier(knn, max_samples =.5, max_features=2)


# In[18]:


bag.fit(X_train, y_train)


# In[19]:


#BaggingClassifier(base_estimator=KNeighborsClassifier(n_neighbors=3),
                  #max_features=2, max_samples=0.5, n_jobs=2, oob_score=True)


# In[20]:


#evaluae the model
y_pred = bag.predict(X_test)
accuracy_score(y_test, y_pred)


# In[21]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=bag.classes_.tolist()))


# In[22]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report
#assuming 'knn' is your trained model, 'X_test' are your test features
predictions = bag.predict(X_test)
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=bag.classes_)
disp.plot()

plt.suptitle("Confusion Matrix for Iris Dataset")
plt.show()


# In[23]:


from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('pinfo', 'RandomForestClassifier')


# In[24]:


rf = RandomForestClassifier(n_estimators=20)


# In[25]:


rf.fit(X_train, y_train)


# In[26]:


#evaluae the model
y_pred = rf.predict(X_test)
accuracy_score(y_test, y_pred)


# In[27]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=rf.classes_.tolist()))


# In[28]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report
#assuming 'knn' is your trained model, 'X_test' are your test features
predictions = rf.predict(X_test)
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot()

plt.suptitle("Confusion Matrix for Iris Dataset")
plt.show()


# In[29]:


from sklearn.ensemble import AdaBoostClassifier
get_ipython().run_line_magic('pinfo', 'AdaBoostClassifier')


# In[30]:


ada=AdaBoostClassifier(n_estimators=100)


# In[31]:


ada.fit(X_train, y_train)


# In[32]:


#evaluae the model
y_pred = ada.predict(X_test)
accuracy_score(y_test, y_pred)


# In[33]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred, labels=ada.classes_.tolist()))


# In[34]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import classification_report
#assuming 'knn' is your trained model, 'X_test' are your test features
predictions = ada.predict(X_test)
cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ada.classes_)
disp.plot()

plt.suptitle("Confusion Matrix for Iris Dataset")
plt.show()


# In[ ]:




