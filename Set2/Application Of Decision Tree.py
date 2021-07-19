#!/usr/bin/env python
# coding: utf-8

# In[304]:


# Import Dependencies
get_ipython().run_line_magic('matplotlib', 'inline')

# data analysis and wrangling
import math, time, random, datetime

# Data Manipulation
import numpy as np
import pandas as pd

# Visualization 
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-whitegrid')                        # The default theme is DarkGrid Theme .........

# Preprocessing
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn import model_selection, tree, preprocessing, metrics, linear_model
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# Importing the dataset: 

# In[305]:


df = pd.read_csv('Downloads/Card_Attrition_Challenge_DT0.csv', sep=";")
df_test = pd.read_csv('Downloads/Card_Attrition_Challenge_DT0_Test.csv', sep=";")


# Checking for duplicate values: 

# In[306]:


sum(df.duplicated())


# In[307]:


sum(df_test.duplicated())


# Hence there are no duplicate values in our database... 

# In[308]:


print(df.isnull().any())
print('*'*40)
print(df_test.isnull().any())


# So there are missing values in 2 features, in Customer_Age and another in Credit_Limit.....

# In[309]:


print(df.describe(include=['O']))
print('*'*40)
print(df_test.describe(include=['O']))


# In[310]:


print(df['Marital_Status'].value_counts())
print('--'*40)
print(df['Income_Category'].value_counts())
print('--'*40)
print(df['Card_Category'].value_counts())


# Now let's fill up the missing values: 

# In[311]:


df['Customer_Age'].isnull().sum()


# In[312]:


df['Credit_Limit'].isnull().sum()


# There are several methods, like filling the missing values with mean or median or mode etc. But the best method in this case is to remove those missing rows since there are only 14 entries and 2. So removing them will not affect our results much.

# In[313]:


df = df.dropna()
df_test = df_test.dropna()


# In[314]:


print(df.isnull().sum())
print('*'*40)
print(df_test.isnull().sum())


# Great now there are no missing values 

# Now lets check for the ratio of transactions in final quater of 2012 and first quater of 2013 to check if there has been significant change in the transaction activity.  

# In[315]:


#df['SumOfColumns_Q4_12_TransCount'] = df['Trans_Count_Oct12'] + df['Trans_Count_Nov12'] + df['Trans_Count_Dec12']
SumOfColumns_Q4_12_TransCount = df['Trans_Count_Oct12'] + df['Trans_Count_Nov12'] + df['Trans_Count_Dec12']
SumOfColumns_Q4_12_TransCount_test = df_test['Trans_Count_Oct12'] + df_test['Trans_Count_Nov12'] + df_test['Trans_Count_Dec12']


# In[316]:


#df['SumOfColumns_Q1_13_TransCount'] = df['Trans_Count_Jan13'] + df['Trans_Count_Feb13'] + df['Trans_Count_Mar13']
SumOfColumns_Q1_13_TransCount = df['Trans_Count_Jan13'] + df['Trans_Count_Feb13'] + df['Trans_Count_Mar13']
SumOfColumns_Q1_13_TransCount_test = df_test['Trans_Count_Jan13'] + df_test['Trans_Count_Feb13'] + df_test['Trans_Count_Mar13']


# In[317]:


#df['SumOfColumns_Q4_12_TransAmount'] = df['Trans_Amt_Oct12'] + df['Trans_Amt_Nov12'] + df['Trans_Amt_Dec12']
SumOfColumns_Q4_12_TransAmount = df['Trans_Amt_Oct12'] + df['Trans_Amt_Nov12'] + df['Trans_Amt_Dec12']
SumOfColumns_Q4_12_TransAmount_test = df_test['Trans_Amt_Oct12'] + df_test['Trans_Amt_Nov12'] + df_test['Trans_Amt_Dec12']


# In[318]:


#df['SumOfColumns_Q1_13_TransAmount'] = df['Trans_Amt_Jan13'] + df['Trans_Amt_Feb13'] + df['Trans_Amt_Mar13']
SumOfColumns_Q1_13_TransAmount = df['Trans_Amt_Jan13'] + df['Trans_Amt_Feb13'] + df['Trans_Amt_Mar13']
SumOfColumns_Q1_13_TransAmount_test = df_test['Trans_Amt_Jan13'] + df_test['Trans_Amt_Feb13'] + df_test['Trans_Amt_Mar13']


# In[319]:


#df['Ratio_TransCount'] = df['SumOfColumns_Q1_13_TransCount']/df['SumOfColumns_Q4_12_TransCount']
df['Ratio_TransCount'] = SumOfColumns_Q1_13_TransCount/SumOfColumns_Q4_12_TransCount
df_test['Ratio_TransCount'] = SumOfColumns_Q1_13_TransCount_test/SumOfColumns_Q4_12_TransCount_test


# In[320]:


#df['Ratio_TransAmount'] = df['SumOfColumns_Q1_13_TransAmount']/df['SumOfColumns_Q4_12_TransAmount']
df['Ratio_TransAmount'] = SumOfColumns_Q1_13_TransAmount/SumOfColumns_Q4_12_TransAmount
df_test['Ratio_TransAmount'] = SumOfColumns_Q1_13_TransAmount_test/SumOfColumns_Q4_12_TransAmount_test


# In[321]:


df.head()


# In[322]:


df_test.head()


# In[323]:


Y_train = df['Attrition_Flag']
Y_test = df_test['Attrition_Flag']


# In[324]:


df_new = df.drop(['Attrition_Flag'], axis = 1)
df_new_test = df_test.drop(['Attrition_Flag'], axis = 1)


# In[325]:


df_new.head(15)


# Now let's finalize our dataframe from the prescribed calculations, considering numerical dataset: 

# In[326]:


df_final = df_new[['Total_Revolving_Bal', 'Ratio_TransCount', 'Months_Inactive_12_mon', 'Total_Relationship_Count', 'Contacts_Count_12_mon']].copy()
df_final_test = df_new_test[['Total_Revolving_Bal', 'Ratio_TransCount', 'Months_Inactive_12_mon', 'Total_Relationship_Count', 'Contacts_Count_12_mon']].copy()


# In[327]:


df_final.head()


# In[328]:


df_final_test.head()


# In[329]:


X_train = df_final
X_test = df_final_test


# In[330]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[332]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[331]:


df_new.info()

