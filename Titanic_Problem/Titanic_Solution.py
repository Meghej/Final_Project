#!/usr/bin/env python
# coding: utf-8

# ### KAGGLE - TITANIC PROBLEM 

# IMPORTING REQUIRED LIBRARIES AND PACKAGES: 

# In[350]:


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


# In[351]:


#IMPORTING THE DATABASE (TRAINING AS WELL AS TESTING) and Submission format
df_train = pd.read_csv('Downloads/train.csv')
df_test = pd.read_csv('Downloads/test.csv')
gender_submission = pd.read_csv('Downloads/gender_submission.csv') 


# In[352]:


df_train.head()


# In[353]:


#df_both will be useful when we need to perform operations on both the datasets simultaneously 
df_both = [df_train, df_test]


# Here the features that are available in the dataset are as follows: 
# <br>
# 1 PassengerId <br>
# 2 Pclass <br>
# 3 Name <br>
# 4 Sex <br>
# 5 Age <br>
# 6 SibSp <br>
# 7 Parch <br>
# 8 Ticket <br>
# 9 Fare <br>
# 10 Cabin <br> 
# 11 Embarked <br>

# Now the first task is to seperate the data as per it's type : 
# <br>
# Categorical
# <br>
# Numerical 
# <br>
# Doubtful

# Categorical : Name, Sex, Ticket, Cabin, Embarked
# <br>
# Numerical : PassengerId, Age, SibSp, Parch, Fare, Survived, Pclass

# Now we are targetting to estimate the target variable: Survived and hence we will remove it from the consideration for now. 

# In[354]:


df_train.describe()


# Now we need to think about the data which is unavailable. 

# In[355]:


df_train.info()
print('_'*40)
df_test.info()


# We can clearly see that Cabin feature is less than 50% in terms of availability. Hence its better to mark it out of consideration. 

# In[356]:


df_train['Age'].isnull().values.any()


# In[357]:


#Now let us check for the uniqueness of the database
df_train.describe(include=['O'])


# There are duplicate values present in ticket section, which indicates that it should also be removed from our consideration (even intuitively, ticket name shall not be affecting the survivor, though now we have numbers to support our intuition)

# Hence the features that we shall take into consideration are : Sex, Embarked, Age, Pclass, Parch, SibSp, Fare

# We will also need to complete or adjust the features which are missing but do play an important role in preparing our model like Age and Embarked. But first lets analyze the variables and get better insight with their effects on survival. 

# In[358]:


df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# We can see that 1st class people have a better chance of survival than 2nd/3rd class. 

# In[359]:


df_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)                      


# Females are having better survival chance in comparision to males. 

# In[360]:


df_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Correlation is not linear !  

# In[361]:


df_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


# Correlation is not linear !

# ANALYSING DATA: 

# In[362]:


#fig = plt.figure(figsize=(20,1))
sns.countplot(y='Survived', data=df_train);
print(df_train.Survived.value_counts())


# Total Number of Survivors are 342, approximately 38 percent of the total passengers. 

# In[363]:


graph = sns.FacetGrid(df_train, col='Survived')
graph.map(plt.hist, 'Age', bins=10)


# Observation: People in between the age of 20 to 40 are the ones who survived the most. 

# In[364]:


grid = sns.FacetGrid(df_train, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=10)
grid.add_legend()


# In[365]:


sns.displot(df_train.Pclass)


# In[366]:


sns.countplot(y="Sex", data=df_train);


# In[367]:


grid = sns.FacetGrid(df_train, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid.map(sns.barplot,'Sex', 'Fare', alpha=1, ci=None)
grid.add_legend()


# Wrangline the data: 

# Let's Drop the columns we eliminated during our analysis : 

# In[368]:


print("Initially", df_train.shape, df_test.shape)

df_train = df_train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)
df_test = df_test.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)

print("After Modification", df_train.shape, df_test.shape)


# In[369]:


df_train.head()


# Converting Categorical Features into numbers: 

# In[370]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_train['Sex'] = le.fit_transform(df_train['Sex'])
df_test['Sex'] = le.fit_transform(df_test['Sex'])


# In[371]:


df_train.head()


# Now let us work towards completing the missing information for some of the features. One solution is to remove those entries and other is to fill it by some guess (mean value maybe...)

# Now here the approach we are using to fill the age value will be based upon assigning the random value in between : <br> 
# $\mu$ - $\sigma$ to $\mu$ + $\sigma$  

# In[372]:


df_both = [df_train, df_test]


# In[373]:


df_train['Pclass'].isnull().values.any()


# In[374]:


df_train['Age'].isnull().values.any()


# In[375]:


df_train['Age'].isnull().values.any()


# In[376]:


df_both = [df_train, df_test]

for data in df_both:
    mean_val = df_train["Age"].mean()
    std_val = df_train["Age"].std()
    null_val = data["Age"].isnull().sum()
    # compute random numbers between the mean_val, std_val and null_val
    rand_age = np.random.randint(mean_val - std_val, mean_val + std_val, size = null_val)
    # fill NaN values in Age column with random values generated
    age_slice = data["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    data["Age"] = age_slice
    data["Age"] = df_train["Age"].astype(int)


# In[377]:


df_train["Age"].isnull().sum()


# Now since we are done with Age, lets focus on other missing values. 

# In[378]:


df_train["Embarked"].isnull().sum()


# Now since there are only 2 values out of 891, hence we can assign it to be anything, it will not affect the results much. 

# In[379]:


df_train["Embarked"] = df_train["Embarked"].fillna('S')


# In[380]:


df_train.info()
df_test.info()


# In[381]:


#need to adjust fare in test dataset (only 1 value)
df_test = df_test.fillna(df_test['Fare'].mean())


# In[382]:


df_train.info()
df_test.info()


# In[383]:


df_test["Embarked"]= LabelEncoder().fit_transform(df_test["Embarked"])
df_train["Embarked"]= LabelEncoder().fit_transform(df_train["Embarked"])


# In[384]:


df_train.head()


# Spliting the Train & Test datasets

# In[385]:


X_train = df_train.drop("Survived", axis=1)
Y_train = df_train["Survived"]


# In[386]:


X_test  = df_test.copy()
X_train.shape, Y_train.shape, X_test.shape


# Training the Classification model

# 1. Logistic Regression: 
#     

# In[399]:


log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
Y_pred = log_reg.predict(X_train)
acc_log = round(log_reg.score(X_train, Y_train) * 100, 2)
acc_log


# 2. SVM

# In[403]:


svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
acc_svc


# 3. KNN

# In[404]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
acc_knn


# 4. Gaussian Naive Bayes

# In[405]:


gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian


# 5. Perceptron

# In[406]:


perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
acc_perceptron


# 6. Linear SVC

# In[407]:


linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
acc_linear_svc


# 7. Stochastic Gradient Descent

# In[409]:


sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# 8. Decision Tree 

# In[411]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# 9. Random Forest

# In[412]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[413]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# Conclusion : Clearly we can see that both Decision Tree and Random Forest can be implemented and the score recieved is also pretty decent.
# 
# You'll have a better chance if you're a woman or a child. You have a better chance of surviving if you have a higher class ticket than if you have a third class ticket. In comparison to Southampton or Queenstown, a man is more likely to live if he embarks in Cherbourg. If you travel with 1 or 3 persons instead of 0 or more than 3, your chances of survival increase.

# In[ ]:




