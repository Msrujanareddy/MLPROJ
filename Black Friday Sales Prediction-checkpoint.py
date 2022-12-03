#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries and Loading data

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)


# In[9]:


data = pd.read_csv("C:\\Users\\makut\\Downloads\\black friday sales\\Black-Friday-Sales-Prediction-master\\Data\\BlackFridaySales.csv")


# In[10]:


data.head()


# In[11]:


data.shape


# In[12]:


data.info()


# ## Checking Null values

# In[13]:


data.isnull().sum()


# ## Null Value in percentage

# In[14]:


data.isnull().sum()/data.shape[0]*100


# # Unique elements in each attributes

# In[15]:


data.nunique()


# # EDA

# ## Target Variable Purchase

# In[16]:


sns.histplot(data["Purchase"],color='r')
plt.title("Purchase Distribution")
plt.show()


# In[17]:


sns.boxplot(data["Purchase"])
plt.title("Boxplot of Purchase")
plt.show()


# In[18]:


data["Purchase"].skew()


# In[19]:


data["Purchase"].kurtosis()


# In[20]:


data["Purchase"].describe()


# ### Gender

# In[21]:


sns.countplot(data['Gender'])
plt.show()


# In[22]:


data['Gender'].value_counts(normalize=True)*100


# In[23]:


data.groupby("Gender").mean()["Purchase"]


# ### Marital Status

# In[24]:


sns.countplot(data['Marital_Status'])
plt.show()


# In[25]:


data.groupby("Marital_Status").mean()["Purchase"]


# In[26]:


data.groupby("Marital_Status").mean()["Purchase"].plot(kind='bar')
plt.title("Marital_Status and Purchase Analysis")
plt.show()


# ### Occupation

# In[27]:


plt.figure(figsize=(18,5))
sns.countplot(data['Occupation'])
plt.show()


# In[28]:


occup = pd.DataFrame(data.groupby("Occupation").mean()["Purchase"])
occup


# In[29]:


occup.plot(kind='bar',figsize=(15,5))
plt.title("Occupation and Purchase Analysis")
plt.show()


# ### City_Category

# In[30]:


sns.countplot(data['City_Category'])
plt.show()


# In[31]:


data.groupby("City_Category").mean()["Purchase"].plot(kind='bar')
plt.title("City Category and Purchase Analysis")
plt.show()


# ### Stay_In_Current_City_Years

# In[32]:


sns.countplot(data['Stay_In_Current_City_Years'])
plt.show()


# In[33]:


data.groupby("Stay_In_Current_City_Years").mean()["Purchase"].plot(kind='bar')
plt.title("Stay_In_Current_City_Years and Purchase Analysis")
plt.show()


# ### Age

# In[34]:


sns.countplot(data['Age'])
plt.title('Distribution of Age')
plt.xlabel('Different Categories of Age')
plt.show()


# In[35]:


data.groupby("Age").mean()["Purchase"].plot(kind='bar')


# In[36]:


data.groupby("Age").sum()['Purchase'].plot(kind="bar")
plt.title("Age and Purchase Analysis")
plt.show()


# ### Product_Category_1

# In[37]:


plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_1'])
plt.show()


# In[38]:


data.groupby('Product_Category_1').mean()['Purchase'].plot(kind='bar',figsize=(18,5))
plt.title("Product_Category_1 and Purchase Mean Analysis")
plt.show()


# In[39]:


data.groupby('Product_Category_1').sum()['Purchase'].plot(kind='bar',figsize=(18,5))
plt.title("Product_Category_1 and Purchase Analysis")
plt.show()


# ### Product_Category_2

# In[40]:


plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_2'])
plt.show()


# ### Product_Category_3

# In[41]:


plt.figure(figsize=(18,5))
sns.countplot(data['Product_Category_3'])
plt.show()


# In[42]:


data.corr()


# ## HeatMap

# In[43]:


sns.heatmap(data.corr(),annot=True)
plt.show()


# In[44]:


data.columns


# In[45]:


df = data.copy()


# In[46]:


df.head()


# In[47]:


# df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace(to_replace="4+",value="4")


# In[48]:


#Dummy Variables:
df = pd.get_dummies(df, columns=['Stay_In_Current_City_Years'])


# ## Encoding the categorical variables

# In[49]:


from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()


# In[50]:


df['Gender'] = lr.fit_transform(df['Gender'])


# In[51]:


df['Age'] = lr.fit_transform(df['Age'])


# In[52]:


df['City_Category'] = lr.fit_transform(df['City_Category'])


# In[53]:


df.head()


# In[54]:


df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')
df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')


# In[55]:


df.isnull().sum()


# In[56]:


df.info()


# ## Dropping the irrelevant columns

# In[57]:


df = df.drop(["User_ID","Product_ID"],axis=1)


# ## Splitting data into independent and dependent variables

# In[58]:


X = df.drop("Purchase",axis=1)


# In[59]:


y=df['Purchase']


# In[60]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# ## Modeling

# # Random Forest Regressor

# In[61]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
# create a regressor object 
RFregressor = RandomForestRegressor(random_state = 0)  


# In[ ]:


RFregressor.fit(X_train, y_train)


# In[ ]:


rf_y_pred = RFregressor.predict(X_test)


# In[ ]:


mean_absolute_error(y_test, rf_y_pred)


# In[ ]:


mean_squared_error(y_test, rf_y_pred)


# In[ ]:


r2_score(y_test, rf_y_pred)


# In[ ]:


from math import sqrt
print("RMSE of Random Forest Model is ",sqrt(mean_squared_error(y_test, rf_y_pred)))


# # XGBoost Regressor

# In[ ]:


get_ipython().system('pip install xgboost')


# In[ ]:


from xgboost.sklearn import XGBRegressor


# In[ ]:


xgb_reg = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)

xgb_reg.fit(X_train, y_train)


# In[ ]:


xgb_y_pred = xgb_reg.predict(X_test)


# In[ ]:


mean_absolute_error(y_test, xgb_y_pred)


# In[ ]:


mean_squared_error(y_test, xgb_y_pred)


# In[ ]:


r2_score(y_test, xgb_y_pred)


# In[ ]:


from math import sqrt
print("RMSE of XGBoost Model is ",sqrt(mean_squared_error(y_test, xgb_y_pred)))


# In[ ]:




