#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import dates
from datetime import datetime


# In[2]:


# 1-import data


# In[3]:


data = pd.read_csv(r"C:\Users\Elnakib\OneDrive\Desktop\data\walmart-sales-dataset-of-45stores.csv")
df = pd.DataFrame(data)
df


# In[4]:


#data clean


# In[5]:


df.info()


# In[6]:


#I get the count of null values in a column. But the default axis for.sum () is None, or 0 - which should be summing across the columns.


# In[7]:


df.isnull().sum() 


# In[8]:



df = df.drop_duplicates()
df


# In[9]:


#The data set contains wrong data
#we must do type casting


# In[35]:


df['Date'] = pd.to_datetime(df['Date'])


print(df.to_string())


# In[11]:


df.info()


# In[12]:


#Which store has maximum sales


# In[13]:


maxSales = data.groupby('Store')['Weekly_Sales'].sum()


# In[14]:


sorted_maxSales = maxSales.sort_values(ascending=False)
sorted_maxSales


# In[15]:


print("Store that has Highest sales is : {} , and The Value of Sales is {}:".format(maxSales.idxmax(),maxSales.max()))


# In[16]:


plt.figure(figsize=(15,5))
sns.barplot(data=data,x=data.Store,y=data.Weekly_Sales)
#plt.bar(data=data,x="Store",height='Weekly_Sales')
plt.show()


# In[17]:


# Which store has maximum standard deviation i.e., the sales vary a lot


# In[18]:


max_std = data.groupby('Store')['Weekly_Sales'].std()
max_std.idxmax()


# In[19]:


# Some holidays have a negative impact on sales. Find out holidays that have higher sales than the mean sales in the non-holiday season for all stores together.


# In[20]:


data_normal_day=data[data.Holiday_Flag==0]
data_normal_day


# In[21]:


data_holiday_day=data[data.Holiday_Flag==1]
data_holiday_day


# In[22]:


data_holiday_positive_impact = data_holiday_day[(data_holiday_day.Weekly_Sales) > (data_normal_day.Weekly_Sales.mean())]
data_holiday_positive_impact


# In[23]:


#########################################################################################


# In[24]:


df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df2010=df.loc[(df['Date'] >= '2010-01-01')& (df['Date'] < '2010-12-31')]
df2011=df.loc[(df['Date'] >= '2011-01-01')& (df['Date'] < '2011-12-31')]
df2012=df.loc[(df['Date'] >= '2012-01-01')& (df['Date'] < '2012-12-31')]


# In[25]:


df2010ByMonth=df2010.groupby(df2010.Date.dt.month)['Weekly_Sales'].sum()
df2010ByMonth=pd.DataFrame(df2010ByMonth)
df2010ByMonth.columns=["Monthly_Sales_for2010"]

df2011ByMonth=df2011.groupby(df2011.Date.dt.month)['Weekly_Sales'].sum()
df2011ByMonth=pd.DataFrame(df2011ByMonth)
df2011ByMonth.columns=["Monthly_Sales_for2011"]

df2012ByMonth=df2012.groupby(df2012.Date.dt.month)['Weekly_Sales'].sum()
df2012ByMonth=pd.DataFrame(df2012ByMonth)
df2012ByMonth.columns=["Monthly_Sales_for2012"]


# In[26]:


plt.bar(df2010ByMonth.index , height=df2010ByMonth["Monthly_Sales_for2010"] )

plt.xlabel("Month")
plt.ylabel("Sales per month")
plt.title("Sales per month 2010")
plt.show()


# In[27]:


plt.bar(df2011ByMonth.index , height=df2011ByMonth["Monthly_Sales_for2011"] )
plt.xlabel("Month")
plt.ylabel("Sales per month")
plt.title("Sales per month 2011")
plt.show()


# In[28]:



plt.bar(df2012ByMonth.index , height=df2012ByMonth["Monthly_Sales_for2012"] )
plt.xlabel("Month")
plt.ylabel("Sales per month")
plt.title("Sales per month 2012")
plt.show()


# In[29]:



df2 = df2011ByMonth.join(df2010ByMonth)
df3 = df2.join(df2012ByMonth)
ax = df3.plot.bar(rot=0)
plt.show()


# In[30]:



df["Date"]=pd.to_datetime(df["Date"])
df["quarter"]=df["Date"].dt.quarter
df["semster"]=np.where(df["quarter"].isin([1,2]),1,2)

plt.title("Sales of semester in all years")
plt.xlabel("Semester")
plt.ylabel("Total Sales/$ ")
plt.text(1,3500000,"1 = First half of the year \n2 = Second half of the year",color="r")
plt.bar(df["semster"],height=df["Weekly_Sales"],width=0.1 , align='center')
plt.show()


# In[31]:



plt.hist(df["Temperature"])
plt.ylabel("weekly sales")
plt.xlabel("Temperature")
plt.show()


# In[32]:


sns.scatterplot(x = df["Weekly_Sales"] ,y  = df["Unemployment"])
plt.show()


# In[33]:


sns.scatterplot(x = df["CPI"] ,y  = df["Weekly_Sales"])
plt.show()


# In[ ]:





# In[ ]:




