#!/usr/bin/env python
# coding: utf-8

# In[1]:


#First we load the basic necessary packages
import pandas as pd
import numpy as np
import scipy.stats as ss
import copy
import matplotlib.pyplot as plt
import pandas_profiling
import sklearn.preprocessing as preprocessing
import seaborn as sns
import sklearn.model_selection as cross_validation
import sklearn.linear_model as linear_model
from sklearn.preprocessing import OneHotEncoder
from pandas.plotting import scatter_matrix

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist

from pylab import rcParams
import seaborn as sb

import sklearn
from sklearn.cluster import AgglomerativeClustering
import sklearn.metrics as sm


# In[2]:


import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
np.set_printoptions(precision=4, suppress=True)
plt.figure(figsize=(10,3))
plt.style.use('seaborn-whitegrid')


# In[3]:


elog = pd.read_csv("BPI Challenge 2017 - e.csv", sep=",")


# In[4]:


group_act = elog.groupby('Activity')['Resource'].count()


# In[5]:


print(group_act)


# In[6]:


group_elog = elog.groupby(['Resource','Activity']).agg({'Activity':'count'})
print(group_elog)


# In[7]:


group_elog.columns=['Count']

group_elog = group_elog.reset_index()


# Udh hitung jumlah aktivitas yang dilakukan, bisa hitung berapa distance antara user based on activities.

# In[8]:


print(group_elog)


# In[9]:


ggroup_elog = group_elog


# In[10]:


ggroup_elog = ggroup_elog.groupby('Resource')['Count'].sum()


# In[11]:


ggroup_elog.columns='Sum'
ggroup_elog = ggroup_elog.reset_index()


# In[12]:


print(ggroup_elog)


# In[13]:


result2 = pd.merge(group_elog, ggroup_elog, how='outer', on='Resource')
result = pd.merge(result2, group_act, how='outer', on='Activity')


# In[14]:


print(result)


# In[15]:


result['Percentage_Ind'] = result.apply(lambda row: row.Count_x/row.Count_y, axis=1)
result['Percentage_Act'] = result.apply(lambda row: row.Count_x/row.Resource_y, axis=1)


# In[16]:


print(result)


# In[17]:


result['Expert'] = np.where((result['Percentage_Ind'])>result['Percentage_Act'], 1, 0)


# In[18]:


print(result)


# In[19]:


allresult = pd.DataFrame(result)


# In[20]:


allresult


# In[21]:


xresult = pd.DataFrame(result, columns=['Resource_x','Activity','Expert'])


# In[22]:


xresult.rename(columns = {'Resource_x':'Resource'}, inplace = True)


# In[23]:


xresult


# In[24]:


xresult2 = pd.merge(elog, xresult, how='outer', on=['Resource','Activity'])


# In[25]:


xresult2


# In[26]:


xresult3 = pd.DataFrame(xresult2, columns=(['Case ID','Activity','Resource','Expert']))
xresult3


# In[27]:


axa = xresult3.groupby('Case ID')['Activity'].count()


# In[28]:


axe= xresult3.groupby('Case ID')['Expert'].sum()


# In[29]:


axae = pd.merge(axa,axe, how='outer', on='Case ID')


# In[30]:


axae


# In[31]:


axae['Expert_Rate'] = axae.apply(lambda row: row.Expert/row.Activity, axis=1)


# In[32]:


axae

