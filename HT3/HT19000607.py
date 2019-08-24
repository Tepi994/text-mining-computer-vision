#!/usr/bin/env python
# coding: utf-8

# In[203]:


import pandas as pd
import numpy as np
import re
import os


# In[204]:


os.listdir("C:/Users/jctep/OneDrive/Documents/GitHub/text-mining-computer-vision/HT3/Datos/")
files = os.listdir("C:/Users/jctep/OneDrive/Documents/GitHub/text-mining-computer-vision/HT3/Datos/")


# In[205]:


data = pd.read_csv(str("Datos/"+files[0]), sep=" ", header=None, index_col=None)
data.head()


# In[34]:


str("Datos/"+files[0])


# In[54]:


li = []
for filename in files:
    df = pd.read_csv(str("Datos/"+filename), index_col = None, header = None)
    li.append(df)

data = pd.concat(li, axis = 0, ignore_index = True)
data.columns = ['raw_date']


# In[55]:


len(data)


# In[117]:


data.head()


# In[197]:


data.raw_date = data.raw_date.replace(regex=r'[Jj]an', value='01')
data.raw_date = data.raw_date.replace(regex=r'[Ff]eb', value='02')
data.raw_date = data.raw_date.replace(regex=r'[Mm]ar', value='03')
data.raw_date = data.raw_date.replace(regex=r'[Aa]pr', value='04')
data.raw_date = data.raw_date.replace(regex=r'[Mm]ay', value='05')
data.raw_date = data.raw_date.replace(regex=r'[Jj]un', value='06')
data.raw_date = data.raw_date.replace(regex=r'[Jj]ul', value='07')
data.raw_date = data.raw_date.replace(regex=r'[Aa]ug', value='08')
data.raw_date = data.raw_date.replace(regex=r'[Ss]ep', value='09')
data.raw_date = data.raw_date.replace(regex=r'[Oo]ct', value='10')
data.raw_date = data.raw_date.replace(regex=r'[Nn]ov', value='11')
data.raw_date = data.raw_date.replace(regex=r'[Dd]ec', value='12')
data.raw_date = data.raw_date.str.replace('.','/')
data.raw_date = data.raw_date.str.replace('-','/')


# In[198]:


df= data.raw_date.str.split('/',expand= True)


# In[199]:


df.columns = ['day','month','year']


# In[200]:


df.to_csv('fulldata.csv',index=None)


# In[206]:


df['day'].astype(int).mean()
df['month'].astype(int).mean()
df['year'].astype(int).mean()


# In[210]:


df['day'] = df['day'].astype(int)
df['month'] = df['month'].astype(int)
df['year'] = df['year'].astype(int)


# In[211]:


df.describe()


# In[212]:


df.mean()


# In[ ]:




