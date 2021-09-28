#!/usr/bin/env python
# coding: utf-8

# <img src="static/img/rosslog.png" align="left">  

# # Rossmann Stores Predictions

# This notebook covers basics of data cleaning, EDA, feature engineering, and data modelling. For more in-deph data analysis, please consider Kaggle Kernels: https://www.kaggle.com/c/rossmann-store-sales/kernels
# 
# **IMPORTANT: Make sure you do not skip any cells. In that case you risk not being able to accomplish given tasks.** 
# 

# ## Rossmann Data Cleaning 

# You are provided with historical sales data for 1,115 Rossmann stores. The task is to forecast the "Sales" column for the test set. In this first part of the notebook we will walk you through basic steps that need to be done before start with baseline modelling.

# ### Load libraries and data

# In[1]:


# NumPy for numerical computing
import numpy as np

# Pandas for DataFrames
import pandas as pd
from pandas import DataFrame
from pandas import Grouper

# Matplotlib for visualization
from matplotlib import pyplot as plt

# Display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# Seaborn for easier visualization
import seaborn as sns


# In[2]:


# Load data from CSV
train_dataset = pd.read_csv("train.csv")
train_dataset.head()


# In[3]:


stores = pd.read_csv("store.csv")


# The train data set contains sales by day for each store with the following columns:
# 
# - Store - a unique id number
# - DayOfWeek/Date - the day of the week (1-7) and date (YYYY-MM-DD) for a sales data point
# - Sales - the sales for a given day
# - Customers - the number of customers on a given day. This column is highly correlated with sales and is not present in the test set.
# - Open - Values: 0 = closed, 1 = open
# - Promo - indicates whether a store was running a sales promotion that day
# - StateHoliday - Values: a = public holiday, b = Easter holiday, c = Christmas, 0 = None
# - SchoolHoliday - indicates if a store was affected by the closure of public schools on that day

# ###### Question 1

# In[4]:


# YOUR CODE GOES HERE
# Print first five rows of the "stores" dataset
stores.head()


# The stores data set contains additional columns about each store that does not vary by day:
# 
# - StoreType - differentiates between 4 different store models: a, b, c, d
# - Assortment - describes the level of products available: a = basic, b = extra, c = extended
# - CompetitionDistance - distance in meters to the nearest competitor
# - CompetitionOpenSince[Month/Year] - month/year when the nearest competitor was opened
# - Promo2 - indicator for a recurring promotion: 0 = store not participating, 1 = participating
# - Promo2Since[Week/Year] - calendar week/year when the store started participating in Promo2
# - PromoInterval - describes the intervals when Promo2 is started. E.g. "Feb,May,Aug,Nov" means each round starts in those months of any given year for that store

# ###### Question 2

# In[5]:


# YOUR CODE GOES HERE
# Chech out the shape of the datasets. 
# We have a lot of data observations to play around.
train_dataset.shape, stores.shape


# Take a look and understand different data types.

# In[6]:


# Column datatypes
print(train_dataset.dtypes,'\n')
print(stores.dtypes)


# ### Drop unwanted observations

# Our goal during the data cleaning phase is to fix any problems with the provided data sets, such as:
# 
# - Fix inconsistent data
# - Replace missing data with reasonable values
# - Remove Data that cannot be fixed
# - Convert categorical variables to numeric values
# - Check for outlying values, and correct them if necessary
# 
# We needed to fix these types of problems so that our prediction models could be fit as accurately as possible.

# In[7]:


# Check for duplicates
train_dataset = train_dataset.drop_duplicates()
stores = stores.drop_duplicates()

# Print shape after removing duplicates
train_dataset.shape, stores.shape


# There were no duplicates. Now, let us check closed stores, stores with no customers, and stores where sales values equals zero.

# In[8]:


# Drop closed observation
train_dataset = train_dataset[train_dataset.Open != 0]
train_dataset.shape


# In[9]:


train_dataset.StateHoliday.unique()


# ###### Question 3

# In[10]:


# YOUR CODE GOES HERE
# Print the number of stires where Customers=0
len(train_dataset[train_dataset.Customers == 0].groupby('Store'))


# In[11]:


train_dataset[train_dataset.Customers == 0].sort_values(by=['Store']).head()


# ###### Question 4

# In[12]:


# YOUR CODE GOES HERE
# Print the number of stires where Sales=0
len(train_dataset[train_dataset.Sales == 0].groupby('Store'))


# After checking the data, we decide to drop Sales == 0 observations:

# In[13]:


train_dataset = train_dataset[train_dataset.Sales != 0]


# In[14]:


train_dataset.shape


# ### Feature engineering

# We are going to create new feature called "AvgPurchasing" we will use in the second section of this notebook. Than, we summarize numerical features to understand our data better.

# In[15]:


train_dataset['AvgPurchasing'] = train_dataset.Sales / train_dataset.Customers


# In[16]:


# Summarize numerical features
train_dataset.describe()


# In[17]:


# Promo2Since[Year/Week] 
# Describes the year and calendar week 
# when the store started participating in Promo2

stores.describe()


# ### Missing values of numerical features

# Let us check for missing values and fill in any add indicator variable for missing data

# In[18]:


print(train_dataset.select_dtypes(exclude=['object']).isnull().sum(),'\n')
print(stores.select_dtypes(exclude=['object']).isnull().sum())


# In[19]:


# For the competion data, check the 3 missing CompetitionDistance
stores[stores['CompetitionDistance'].isnull()]


# In[20]:


# Fill and flag the missing numeric data

stores.CompetitionOpenSinceMonth.fillna(0, inplace=True)
stores.CompetitionOpenSinceYear.fillna(0, inplace=True)
stores.CompetitionDistance.fillna(0, inplace=True)

# Indicator variable for missing numeric data

stores['CompetitionOpenSinceMonth_missing'] = stores.CompetitionOpenSinceMonth.isnull().astype(int)
stores['CompetitionOpenSinceYear_missing'] = stores.CompetitionOpenSinceYear.isnull().astype(int)
stores['CompetitionDistance_missing'] = stores.CompetitionDistance.isnull().astype(int)

# Just fill the nan with 0 

stores.Promo2SinceWeek.fillna(0, inplace=True)
stores.Promo2SinceYear.fillna(0, inplace=True)
stores.PromoInterval.fillna(0, inplace=True)


# In[21]:


stores.isnull().sum()


# ### Categorical features cleaning

# Notice that we have two zero values: one integer and one string. Let us fix this and have one string zero value for all StateHoliday representations.

# In[22]:


# Display unique values of 'basement'
train_dataset.StateHoliday.unique()


# In[23]:


train_dataset.groupby('StateHoliday').count()


# ###### Question 5

# In[24]:


# YOUR CODE
# Your result shoud look like this: 
### array(['0', 'a', 'b', 'c'], dtype=object)
train_dataset.StateHoliday = train_dataset.StateHoliday.replace(0,'0')


# Check missing values for categorical variables

# In[25]:


train_dataset.StateHoliday.unique()


# In[26]:


# Display number of missing values by categorical feature
print(train_dataset.select_dtypes(include=['object']).isnull().sum(), '\n')
print(stores.select_dtypes(include=['object']).isnull().sum())


# ### Joining the "Train" and "Store" tables 

# Before joining our tables, we need to set both of our table indexes. Let us set the Store values as our new table indexes.

# ###### Question 6

# In[27]:


# YOUR CODE
# Set index in train_dataset to corespond to the train_dataset['Store'] values
# Set index in stores to corespond to the stores['Store'] values
# Drop Store column in train_dataset
train_dataset.set_index('Store')
stores.set_index('Store')


# In[28]:


df_combined = train_dataset.merge(stores, left_on='Store', right_on='Store')
#df_combined = df_combined.reset_index(drop=True)
df_combined.head()


# ### Convert store type and assortiman from char to int

# StateHoliday, StoreType, and Assortment, needs to be transformed into one-hot-encoding after all the cleaning and feature engineering. Let us explore each of this features unique values and replace them with int values. For example:
# 
# 1. StoreType ['a', 'b', 'c', 'd'] needs to be converted into [1, 2, 3, 4]
# 2. Assortment ['a', 'b', 'c'] needs to be converted into [1, 2, 3]
# 3. StateHoliday ['0', 'a', 'b', 'c'] needs to be converted into [0 ,1, 2, 3]

# In[29]:


df_combined.StoreType.unique()


# In[30]:


df_combined.Assortment.unique()


# In[31]:


df_combined.StateHoliday.unique()


# ###### Question 7

# In[32]:


## YOUR CODE
# Transforme into one-hot-encoding 
# Result should look like: 

# Reminder, the reason we need to convert it to numeric values, is because one-hot-encoding works 
# on the values, where it combines the column label with the list of values in the column. So, for 
# example, if the column label is "ABCDEFG" and the values are "AAAA,BBBB,CCCC", we would get three 
# additional columns, with labels such as "ABCDEFG_AAAA", "ABCDEFG_BBBB" and "ABCDEFG_CCCC", which 
# would be pretty bad to handle (especially, if we are building Flask variables in our form ;). 
# Instead, using numbers for the values, we would get a nice variable "ABCDEFG_1", "ABCDEFG_2" 
# and "ABCDEFG_3", which is much easier to manipulate!

#### StoreType ['a', 'b', 'c', 'd'] needs to be converted into [1, 2, 3, 4]
#### Assortment ['a', 'b', 'c'] needs to be converted into [1, 2, 3]
#### StateHoliday ['0', 'a', 'b', 'c'] needs to be converted into [0 ,1, 2, 3]

df_combined.StateHoliday = df_combined.StateHoliday.map({'0': 0, 'a': 1, 'b': 2, 'c': 3})
df_combined = df_combined.join(pd.get_dummies(df_combined.StateHoliday, prefix="StateHoliday"))

df_combined.StoreType = df_combined.StoreType.map({'a': 1, 'b': 2, 'c': 3, 'd': 4}).fillna(0)
df_combined = df_combined.join(pd.get_dummies(df_combined.StoreType, prefix="StoreType"))

df_combined.Assortment = df_combined.Assortment.map({'a': 1, 'b': 2, 'c': 3}).fillna(0)
df_combined = df_combined.join(pd.get_dummies(df_combined.Assortment, prefix="Assortment"))


# In[33]:


# Check out your previous code
print("StoreType: ", np.sort(df_combined.StoreType.unique()))
print("Assortment: ", np.sort(df_combined.Assortment.unique()))
print("StateHoliday: ", np.sort(df_combined.StateHoliday.unique()))


# ## EDA

# To determine what type of models and predictors might work best for predicting sales, we studied which factors are causing the most variance in sales.

# #### Variations based on Store Attributes

# One of the biggest sources of variance in sales is based on the store number. In the graph below we show the average sales for the first ten stores in the data set. We can see that the average sales are quite different for each store and that there is no linear pattern. We do see stores with similar sales levels, suggesting the stores may fall into groups.
# 
# The scatterplot below that shows the average sales for all the stores. We can again see that some stores have similar levels of sales, so it's likely we can group stores together when making sales predictions, which is something for which tree models are well suited.
# 
# Each store number is effectively a category with its own particular sales level. When predicting future sales, we would like a single model that can make predictions for each store rather than having to fit a separate model for each store.

# In[34]:


df_subset = df_combined[(df_combined['Store'] < 11)]
sns.set(font_scale=2)

# Plot store sales for stores 1 to 10
fig, ax = plt.subplots(1,1)

p3 = sns.barplot(x='Store', y='Sales', data=df_subset, ax=ax)
ax.set(xlabel='Store Number')
ax.set_ylabel('Average Sales')
ax.set_title('Sales for ten stores')

plt.show()

# Create a plot of average sales per store id
avg_sales_per_store = df_combined[['Sales', 'Store']].groupby('Store').mean()
avg_sales_per_store.reset_index().plot(kind='scatter', x='Store', y='Sales')
plt.xlim(0,960)
plt.title("Average Sales by Store Number")
plt.show()


# The data set has categorical variables for different store types and the assortment of products for sale at each store. We see that these categories are a source of variance in sales.

# #### Impact of Number of Customers

# After exploration of the train and test sets, we ascertained that only the training data contains the number of Customers feature. The visualization below shows the positive correlation between the number of customers and sales (and also outliers). This feature is highly correlated with sales, but it’s not available until after the sales occur (i.e. it’s not in the test set). But exploring this feature helps make a case for including a proxy for the Customers feature in our final predictors (see the Feature Engineering section).
# 
# We also see that there’s added information with the behavior of customers per StoreType. We see that StoreType d (shown in green) that projects to the upper left quadrant. This means that for the same level of Sales, StoreType d requires fewer customers than StoreType b (blue). There’s less clarity of StoreType impact in the middle of the cone shaped scatter, but we can certainly see the impact of Customers and Sales conditional on the StoreType mostly in the outer regions of the visualization. Thus StoreType is a relevant feature, but as a function of Customers, as we can see that around 2000 to 3000 Customers there’s a break in the cluster pattern of the middle section of the cone. Therefore, further substantiation for us to include a proxy of the Customers feature in the model.

# In[35]:


sns.lmplot(x='Customers', y='Sales', data=df_combined, hue='StoreType',fit_reg=False)


# In[36]:


g = sns.FacetGrid(df_combined, col="StoreType")
g.map(plt.scatter, "Customers", "Sales")


# ## Rossmann Data Modeling

# Due to the regressive predictive nature of The Rossmann Project, we decided to approach our model methodology with a Linear Regression base model with feature selection. Note: you can also consider Ridge regression to evaluate the effects of regularization on the predictive performance. 

# *Note: The time dimension has a tremendous impact in this project. The Kaggle competition consisted in predicted the next 6 weeks (or 42 days), our splitting lead to have a testing set very large of over 900 days, explaining the fast degradation of the explained variance. These issues are the same the world of finance and economics are facing. For example the usage of an AR(n) to predict future outcomes quickly converge to a stationary states. In a real world, one would adjust the model based on the observed error. This is the spirit of tools such as the 'Kalman Filter' that are usually implemented in these domains.* 
# 
# *One other potential approach would consist in implementing more complicated model that naturally combine time series and decision tree, such as ART (Autoregressive Trees). The proposed method are outside of the scope of this course and thus we will focus on some other tehniques here.*

# ### Linear Regression

# For our first baseline model, we decided to use linear regression. We used one hot encoding to convert categorical values into indicator columns. The store number is effectively a category with over 1000 different categories. We know that sales is not linear with the arbitrary numbers assigned to stores, so if this were our final model, it might have been reasonable to fit a separate regression model for each store. But since this is a baseline model, we treated the high dimension categories as if they were numeric values.
# 
# In the next section we will explore how different features affect overall score. You will be challenged to spot the differences and explain what given results mean.

# #### Load libraries

# In[37]:


from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# Let us drop some temporal features given that we are not performing time-series analysis.

# In[38]:


#drop PromoInterval (includes Mar,Jun,Sept,Dec)
#drop Date -> no time-series prediction

no_Promo_Date = df_combined.drop(['PromoInterval', 'Date'], axis=1)

# create tagret (Sales) variable 
target = no_Promo_Date['Sales']


# #### Linear Regression: *ver. 1*

# In[39]:


# Drop Sales and AvgPurchasing, but leave Customers
drop_sales_avg_df = no_Promo_Date.drop(['Sales', 'AvgPurchasing'], axis=1)

# Split the data into train/test
X_train, X_test, y_train, y_test = train_test_split(drop_sales_avg_df, target, test_size=0.7)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

# Explained variance score: 1 is perfect prediction
print('R2 score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.figure(figsize=(20, 5))

plt.plot(y_test.values, color='red', linewidth=0.1)
plt.plot(y_pred, color='blue', linewidth=0.1)


# In short, explain what the above plot and its specific trend.

# ###### Question 8

# ** Your answer: **
# 
# 
# 
# 
# 
# 

# **This is a pretty decent coverage. The R2 score is quite high, and the plot shows pretty 
# close coverate between the predicted and the actual values.**
# 
# **In light of the other linear regressions, however, it seems there is room for improvement.**

# #### Linear Regression: *ver. 2*

# ###### Question 9

# In[40]:


# YOUR CODE
# Drop Sales, but leave Customers and AvgPurchasing
# Elaborate about given R2 score

# Drop Sales and AvgPurchasing, but leave Customers
drop_sales_only_df = no_Promo_Date.drop(['Sales'], axis=1)

# Split the data into train/test
X_train, X_test, y_train, y_test = train_test_split(drop_sales_only_df, target, test_size=0.7)

# Create linear regression object
regr9 = linear_model.LinearRegression()

# Train the model using the training sets
regr9.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr9.predict(X_test)

# Explained variance score: 1 is perfect prediction
print('R2 score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.figure(figsize=(20, 5))

plt.plot(y_test.values, color='red', linewidth=0.1)
plt.plot(y_pred, color='blue', linewidth=0.1)


# In[41]:


# Export this algorithm, to be used in the website
import joblib
joblib.dump(regr9, 'regr9.pkl')


# ** Your answer: **

# **This seems to be the best regression, yet. The R2 number is quite high. Actually, 
# suspiciously high, which might indicate some sort of overfitting. **
# 
# **However, after visual inspection, there is barely any red area, which indicates, 
# that the predicted values cover the actual values pretty closely.**

# #### Linear Regression: *ver. 3*

# In[ ]:


# Drop Sales and Customers
drop_sales_cust_df = no_Promo_Date.drop(['Sales', 'Customers'], axis=1)

# Split the data into train/test
X_train, X_test, y_train, y_test = train_test_split(drop_sales_cust_df, target, test_size=0.2)

# Create linear regression object
regr3 = linear_model.LinearRegression()

# Train the model using the training sets
regr3.fit(X_train, y_train)

# Make predictions using the testing set
y_pred = regr3.predict(X_test)

# Explained variance score: 1 is perfect prediction
print('R2 score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.figure(figsize=(20, 5))

plt.plot(y_test.values, color='red', linewidth=0.1)
plt.plot(y_pred, color='blue', linewidth=0.1)


# ###### Question 10

# ** Your answer: **

# **This is a super bad version. The R2 score is very low 
# (compared to the other two versions of linear regression) and 
# the prediction is clearly too small against the actual values.**
# 
# _(i.e. blue barely covers the red lines)_

# ### Tree Based Models

# It has been well established that bagging and other resampling techniques can be used to reduce the variance in model predictions. As several replicates of the original data set are created using random selection with replacement, at every step, each derivative data set is then used to construct a new model and the models are gathered together into an ensemble. To make a prediction, all of the models in the ensemble are polled and their results are averaged in the case of regression, which is our specific approach.
# 
# As well, it has been well established that a powerful modeling algorithm that makes good use of bagging is Random Forests, which works by training numerous decision trees each based on a different resampling of the original training data. The random forest algorithm improves on bagging by training each tree on a random sample of the available features, to prevent each tree from choosing the same predictors.
# 
# In Random Forests the bias of the full model is equivalent to the bias of a single decision tree, which itself has high variance. By creating many of these trees, a forest, and then averaging them, the variance of the final model can be greatly reduced over that of a single tree. In practice the only limitation we encountered on the size of the forest is computing time, as an infinite number of trees could be trained without ever increasing bias and with a continual - if asymptotically declining - decrease in the variance.
# 
# It is for the aforementioned that we considered Random Forests as a Baseline Ensemble and its constituent, the Decision Tree Regressor as a baseline model as well.

# ### Decision Trees

# Decision trees were implemented as to evaluate the base consituent of the Random Forest ensemble. As per our research on previous models, Random Forest is a very good performance candidate for a baseline ensemble model for Rossmann.
# 
# Decision Trees are also evaluated as a non-parametric baseline model, which enriches the comparative analysis of the Linear Regression Models.
# 
# For fitting the decision tree model, we do not need to create dummy variables for categorical columns, since trees are able to make use of the factorized category values we created during data cleaning.

# ### Random Forest

# Next we considered Random Forest models. 

# In[ ]:


# Load libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split

# Clean your data
clean_df = df_combined.drop(['Sales', 'Customers', 'AvgPurchasing', 'PromoInterval', 'Date'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(clean_df, target, test_size=0.2)


# In[ ]:


def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
    gs.fit(X, y)

    return gs


# #### Parameter Tuning via CV GridSearch

# The paramters to be tuned are:
# 
# - max_depth
# - n_estimartors
# - max_features
# - random_state

# IMPORTANT: This might take a while!!!

# In[ ]:


# tuning n_estimators
parameters = {'max_depth': [1,3,5], # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
              'n_estimators': [10,20,30], # [ 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
              'max_features': ['auto'], # [None, 'auto', 'sqrt', 'log2']
              'random_state': [None], # [None, 10, 44, 100]
}

rf = RandomForestRegressor(max_depth=2, random_state=10)
bstGridSearch = cv_optimize(rf, parameters, X_train, y_train, n_jobs=-1, n_folds=5)
#


# In[ ]:


bstGridSearch.best_params_


# In[ ]:


print('\nThe tuned parameters in the RF w/ customers, via CV grid search are:\n')
print('Max tree depth: {}\nNumber of Estimators {}\nMax Features: {}\nRandom State {}'.                                                                     format(bstGridSearch.best_params_.max_depth,
                                                                     bstGridSearch.best_params_.n_estimators,
                                                                     bstGridSearch.best_params_.max_features,
                                                                     bstGridSearch.best_params_.random_state) )


# ###### Question 11

# In[ ]:


# YOUR CODE
# Run RandomForestRegressor with best params
# Elaborate about given R2 score
rf.fit(X_train,y_train)
y_predict = rf.predict(X_test)
print('R2 score: %.2f' % r2_score(y_test, y_pred))


# ###### Question 12

# #### What is the best model? ** Your answer: **

# In[ ]:





# ## Export models 

# In our final step, we will choose the best model and explore some important prediction elements. Our app accepts features and gives predictions based on users inputs. Let us see what kind of data input is neccessarry (this is important for you to setup the proper input on the Flask side). 

# For example, take one simple observation (we are using first data row):

# In[ ]:


# Print first row
X_test.iloc[0]


# Predict based on your first row:

# In[ ]:


# Test for your app
rf.predict(np.array(X_test.iloc[0].reshape(1, -1)))


# In[ ]:


rf.fit(X_test)


# Similar to the above line:

# In[ ]:


rf.predict(np.array([1, 1, 1, 0, 1, 109, 1, 1, 290.0, 10.0, 2011.0, 1, 40.0, 2014.0, 0, 0, 0]).reshape(1, -1))


# We can conclude that your input needs to be **numpy array**. Now, we can export pur model and get ready for the second part of the homework.

# In[ ]:


# because warning, check the docs, and use it directly
# https://joblib.readthedocs.io/en/latest/persistence.html#use-case
import joblib
joblib.dump(rf, 'rm.pkl')

