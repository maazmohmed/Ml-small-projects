#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams
rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ["Normal", "Fraud"]


# In[92]:


df_train = pd.read_csv("TRAIN.csv")


# In[93]:


df_train.head()


# In[94]:


df_test = pd.read_csv("TEST.csv")


# In[95]:


df_train.columns
df_train.shape


# In[96]:


df_test.columns
df_test.shape


# In[97]:


df_train.isnull().sum()


# In[98]:


inputs=df_train.drop(["Class"],axis="columns")
target=df_train["Class"]
inputs.shape


# In[99]:



df_test.shape


# In[100]:


import sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing as per
scaler=per.MinMaxScaler(feature_range=(0,1))
rescaleData=scaler.fit_transform(inputs)
rescaleData=pd.DataFrame(rescaleData,index=inputs.index,columns=inputs.columns)


# In[101]:


test_data=scaler.transform(df_test)
test_data=pd.DataFrame(test_data,index=df_test.index,columns=df_test.columns)
test_data


# In[102]:


test_data=test_data.drop(["Index"],axis="columns")
test_data


# In[103]:


columns=inputs.columns
columns


# In[104]:


rescaleData=rescaleData.drop(["Index"],axis="columns")
rescaleData


# In[ ]:





# In[105]:


column1=df_train.columns
column1


# In[106]:


x_train.info()


# In[66]:


count_classes = pd.value_counts(x_train['Class'], sort = True)

count_classes.plot(kind = 'bar', rot=0)

plt.title("Transaction Class Distribution")

plt.xticks(range(2), LABELS)

plt.xlabel("Class")

plt.ylabel("Frequency")


# In[107]:


fraud = df_train[df_train['Class']==1]

normal = df_train[df_train['Class']==0]


# In[108]:


outlier_fraction = len(fraud)/float(len(normal))
outlier_fraction


# In[109]:


print(fraud.shape,normal.shape)


# In[33]:



f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
f.suptitle('Amount per transaction by class')
bins = 50
ax1.hist(fraud.Amount, bins = bins)
ax1.set_title('Fraud')
ax2.hist(normal.Amount, bins = bins)
ax2.set_title('Normal')
plt.xlabel('Amount ($)')
plt.ylabel('Number of Transactions')
plt.xlim((0, 20000))
plt.yscale('log')
plt.show();


# In[34]:


## Correlation
import seaborn as sns
#get correlations of each features in dataset
corrmat = df_train.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df_train[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# n_outliers = len(fraud)
# for i, (clf_name,clf) in enumerate(classifiers.items()):
#     #Fit the data and tag outliers
#     if clf_name == "Local Outlier Factor":
#         y_pred = clf.fit_predict(inputs)
#         scores_prediction = clf.negative_outlier_factor_
#     elif clf_name == "Support Vector Machine":
#         clf.fit(inputs)
#         y_pred = clf.predict(inputs)
#     else:    
#         clf.fit(inputs)
#         scores_prediction = clf.decision_function(inputs)
#         y_pred = clf.predict(inputs)
#     #Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions
#     y_pred[y_pred == 1] = 0
#     y_pred[y_pred == -1] = 1
#     n_errors = (y_pred != target).sum()
#     # Run Classification Metrics
#     print("{}: {}".format(clf_name,n_errors))
#     print("Accuracy Score :")
#     print(accuracy_score(target,y_pred))
#     print("Classification Report :")
#     print(classification_report(target,y_pred))

# In[114]:


test1_pred = model.predict(test_data)
test1_pred


# In[64]:


from sklearn.ensemble import GradientBoostingClassifier
model1=GradientBoostingClassifier(n_estimators=200,max_depth=12,learning_rate=0.2)
model1.fit(rescaleData,target)
model1.score(rescaleData,target)


# In[115]:


test1_pred[test1_pred == 1] = 0
test1_pred[test1_pred == -1] = 1
test1_pred


# In[ ]:


frauds = test_pred[test_pred==1]
normals = test_pred[test_pred==0]
print(frauds.shape,normals.shape)


# In[1]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import *


# In[116]:


pred=pd.DataFrame(test1_pred)
sub_df=pd.read_csv("sample_submission.csv")
datasets=pd.concat([sub_df['Index'],pred],axis=1)
datasets.columns=['Index','Class']
datasets.to_csv('submission.csv',index=False)


# In[ ]:




