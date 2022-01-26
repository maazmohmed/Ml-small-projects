#!/usr/bin/env python
# coding: utf-8

# In[141]:


import pandas as pd
df=pd.read_csv("new_train.csv")
df_test=pd.read_csv("new_test.csv")
df_test
df_date=pd.read_csv("new_test.csv")


# In[128]:


df_test=pd.DataFrame(df_test,columns=["Open-Stock-5","High-Stock-5","Low-Stock-5","VWAP-Stock-5","Volume-Stock-5","Turnover-Stock-5"])
df_test


# In[129]:


inputs1=pd.DataFrame(df, columns = ["Open-Stock-5","High-Stock-5","Low-Stock-5","VWAP-Stock-5","Volume-Stock-5","Turnover-Stock-5"])
target1=pd.DataFrame(df, columns = ["Close-Stock-5"])
inputs1


# In[130]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df=scaler.fit_transform(np.array(inputs1))
df


# In[131]:


df_t=scaler.transform(np.array(df_test))
df_t


# In[132]:


columns = ["Open-Stock-5","High-Stock-5","Low-Stock-5","VWAP-Stock-5","Volume-Stock-5","Turnover-Stock-5"]


# In[133]:


columns_test=["Open-Stock-5","High-Stock-5","Low-Stock-5","VWAP-Stock-5","Volume-Stock-5","Turnover-Stock-5"]


# In[134]:


df1=pd.DataFrame(data=df,columns=columns)
df1


# In[135]:


df_t=pd.DataFrame(data=df_t,columns=columns_test)
df_t


# In[123]:


import xgboost 


# In[72]:


model1=xgboost.XGBRegressor()


# In[91]:


model2=xgboost.XGBRegressor()
model3=xgboost.XGBRegressor()
model4=xgboost.XGBRegressor()
model5=xgboost.XGBRegressor()


# In[78]:


model1.fit(df1,target1)


# In[96]:


model2.fit(df1,target1)


# In[107]:


model3.fit(df1,target1)


# In[124]:


model4.fit(df1,target1)


# In[136]:


model5.fit(df1,target1)


# In[ ]:





# In[79]:


preds=model1.predict(df_t)
preds


# In[97]:


preds1=model2.predict(df_t)
preds1


# In[108]:


preds2=model3.predict(df_t)
preds2


# In[125]:


preds3=model4.predict(df_t)
preds3


# In[137]:


preds4=model5.predict(df_t)
preds4


# In[142]:


frame = { 'Date': df_date["Date"], 'Close-Stock-1':preds,'Close-Stock-2':preds1,'Close-Stock-3':preds2,'Close-Stock-4':preds3,'Close-Stock-5':preds4}
result = pd.DataFrame(frame)
result


# In[143]:


result.to_csv("submit.csv")


# In[ ]:




