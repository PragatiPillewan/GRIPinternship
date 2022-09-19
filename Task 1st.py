#!/usr/bin/env python
# coding: utf-8

# # GRIP : The Sparks Foundation

# ## Data Science and Business Analytics Intern
# 

# *Author : Pragati Pillewan*

# ### Task 1: Prediction Using Supervised ML
# 

# #### In this task we have to predict the percentage score of an student based on study hours.This is a simple linear regression task which has two variables no. of hours studied and score. 
# 

# In[1]:


# Importing Libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# #### Reading Data from URL

# In[2]:


url ='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
data = pd.read_csv(url)


# #### Exploring Data

# In[4]:


print(data.shape)
data.head()


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.plot(kind = 'scatter', x = 'Hours' , y = 'Scores');
plt.show()


# In[8]:


data.corr(method = 'pearson')


# In[9]:


data.corr(method = 'spearman')


# In[10]:


hours = data['Hours']
scores = data['Scores']


# In[11]:


sns.distplot(hours)


# In[12]:


sns.distplot(scores)


# #### Linear Regression

# In[13]:


x = data.iloc[:, :-1].values
y = data.iloc[:,1].values


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =0.2, random_state = 0 )


# In[15]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[16]:


X_train


# In[17]:


y_train


# In[18]:


X_test


# In[19]:


y_test


# In[20]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)


# In[21]:


m = reg.coef_
c = reg.intercept_
line = m*x+c
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# In[22]:


y_pred = reg.predict(X_test)


# In[23]:


actual_predicted = pd.DataFrame({'Target' :y_test, 'Predicted' :y_pred})
actual_predicted


# In[24]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test - y_pred))
plt.show()


# ### What would be the predicted score if a student studies for 9.25 hours/ day?

# In[25]:


h = 9.25
s = reg.predict([[h]])
print("If a student studies for {} hours per day he/she will score {} % in exam. ".format(h,s))


# #### Model Evaluation

# In[27]:


from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolut Error:', metrics.mean_absolute_error(y_test, y_pred))
print('R2 Score:',r2_score(y_test,y_pred))


# In[ ]:




