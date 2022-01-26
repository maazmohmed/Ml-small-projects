#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import *


# In[7]:


df_train=pd.read_csv("TRAIN.csv")
df_test=pd.read_csv("TEST.csv")


# In[8]:


#inputs=df_train.drop(["Class"],axis="columns")
target=df_train["Class"]
inputs.shape


# In[9]:


df_test.shape


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing as per
scaler=per.MinMaxScaler(feature_range=(0,1))
rescaleData=scaler.fit_transform(inputs)
rescaleData=pd.DataFrame(rescaleData,index=inputs.index,columns=inputs.columns)
test_data=scaler.transform(df_test)
test_data=pd.DataFrame(test_data,index=df_test.index,columns=df_test.columns)
test_data


# In[5]:


rescaleData=rescaleData.drop(['Index'],axis='columns')


# In[6]:


test_data=test_data.drop(['Index'],axis='columns')
test_data


# In[10]:


clf1=setup(data=df_train,target='Class')


# In[12]:


compare_models()


# In[13]:


CatBoost=create_model('catboost')


# In[23]:


model=create_model('et')


# In[24]:


test_pred=model.predict(test_data)
test_pred


# In[ ]:




