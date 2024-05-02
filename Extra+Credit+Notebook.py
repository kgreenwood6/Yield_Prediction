
# coding: utf-8

# # Yield Harvest: Cultivating Insights Through Linear Regression
# 

# ### A high yield is a primary focus in the agricultural industry. Models that predict yield can assist in understanding optimal conditions to attain the highest crop output. The goal of this assignment is to create a linear regression model which will use multiple variables to predict crop yield. 

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display

get_ipython().magic('matplotlib inline')

import requests
import os
import hashlib
import io


# In[2]:


# Load data and view first 5 rows 

ag_df = pd.read_csv('agricultural_yield_t.csv')
ag_df.head()


# ### This dataset consists of 7 variables and 16,000 observations. All variables are continious apart from Seed_Variety which is categorical. In following steps we will attempt to create a predictive model for Yield using the remaining variables. 
# 
# ### The data can be found here: https://www.kaggle.com/datasets/blueloki/synthetic-agricultural-yield-prediction-dataset

# ### Exploratory Data Analysis

# In[3]:


#Descriptive statistics of each variable 

ag_df.describe()


# In[4]:


#Rename columns for easier reference
#ag_df.rename(columns={"Fertilizer_Amount_kg_per_hectare": "Fertilizer", "Rainfall_mm": "Rain", "Irrigation_Schedule": "Irrigate", "Yield_kg_per_hectare": "Yield"})


# In[5]:


ag_df[['Soil_Quality', 'Fertilizer_Amount_kg_per_hectare','Sunny_Days']].boxplot()


# In[6]:


ag_df[['Rainfall_mm', 'Irrigation_Schedule', 'Yield_kg_per_hectare']].boxplot()


# ### The outliers indicated on the graph above do not flag reason for concern and will not be removed or adjusted. 

# In[7]:


# Check for duplicate values
ag_df.duplicated().sum()

# no duplicate values 


# In[8]:


#Check for null values 
ag_df.isnull().sum()

#no null values 


# Since there are no missing values, this dataset will not require any imputation 

# In[9]:


ag_df.dtypes


# In[10]:


sns.heatmap(ag_df.corr())


# ### This dataset was quite clean and did not require imputation or removal of duplicate values. 
# 
# ### Now that we have a better understanding of our data - we move onto the analysis 
# 
# ### We will start off with a Simple Linear Regression 

# Fertilizer companies can often push their own agendas of making sales targets and potentially over-sell to farmers. The first step of this analysis will be a Simple Linear Regression to gain insight into the relationship between fertiliizer use and yield. Can the amount of fertilizer used per hectare predict the yield?

# In[11]:


# Finding alpha and beta 
# y = alpha(x) + beta 

x = ag_df['Fertilizer_Amount_kg_per_hectare']
y = ag_df['Yield_kg_per_hectare']

from numpy import ones


m = len(x)
u = ones(m)
    
alpha = x.dot(y) - u.dot(x)*u.dot(y)/m
alpha /= x.dot(x) - (u.dot(x)**2)/m
    
beta = u.dot(y - alpha*x)/m
    ###
print("alpha:", alpha)
print("beta:", beta)



# In[12]:


#Best fit linear model 

plt.scatter(x, y, color = "green",
            marker = "o", s = 30)
 
    # predicted response vector
y_hat = beta + alpha*x
 
    # plotting the regression line
plt.plot(x, y_hat, color = "black")
 
    # labels
plt.xlabel('Fertilizer (x)')
plt.ylabel('Yield (y)')
 
    # function to show plot
plt.show()


# In[13]:


#Function to determine R-squared of linear model 

def r_sq(y, y_hat):
    ###
    from statistics import mean
    
    SSR = np.sum((y - y_hat)**2)
    y_line = mean(y)
    SST = np.sum((y - y_line)**2)
    
    return 1 - (SSR/SST)


# In[14]:


print(r_sq(y, y_hat))


# The R-squared value for this model is 8.1% - which is not high at all. This means that only 8.1% of the data's variability is explained by this model. Based off of this model I would not advise that one can predict the yield based off of fertilizer application. Fertilizer companies can take this into account when it comes to sales and marketing techniques. 

# ### Multiple Linear Regression to Predict Yield
# 
# The previous model was not strong in predicting Yield. We will now incorporate all the variables to create a multiple linear regression model to see if we can improve this.

# In[15]:


#Reminder of the data 

ag_df.head()


# In[16]:


# Separating the predictor from response variables 
X_og = ag_df.drop(columns=['Yield_kg_per_hectare']).values

Y = ag_df['Yield_kg_per_hectare'].values


# Add a column of ones to X for the intercept term
X = np.column_stack((np.ones(len(X_og)), X_og))

#Find Gramian of data matrix X
C = X.T.dot(X)
#Find b 
b = X.T.dot(Y)
theta_est = np.linalg.solve(C, b)

print(f"Estimated Thetas:\n{theta_est}")



# ### The equation for this mulitple linear regression model is:
# ### Yield = 1.54(Soil_Quality) + 300.46(Seed_variety) + 0.81(Fertilizer) + 1.99(Sunny_Days) -
# ### 0.51(Rainfall) + 49.99(Irrigation_Schedule) + 48.33

# In[17]:


# Finding the predicted values 

y_pred = theta_est[0] + np.sum(theta_est[1:] * X_og, axis=1)
print(f"predicted response:\n{y_pred}")


# ### We have used the abovementioned formula to predict the Yield using the given variables
# 
# ### Now lets see how the model performs

# In[18]:


# Calculating R_squared by finding SSR and SST 
from statistics import mean
m_SSR = np.sum((y - y_pred)**2)
my_line = mean(y)
m_SST = np.sum((y - my_line)**2)
    
print(1 - (m_SSR/m_SST))


# ### With an R_squared of 0.94, we can say that 94% of the variability in the Yield response variable is explained by the model 

# ### Project Summary 
# 
# The aim of the assignment was to create a prediction model for crop yield using linear regression. Exploratory data analysis was performed - giving insight into the data's anatomy. The first model was a simple linear regression which used fertilizer to predict yield, an interesting study given the commercial aspect of the predictor variable. This model was not a suitable model to draw conclusions on Yield with an R_sqaured value of 8.1%. We then move onto the multiple linear regression model which used all the variables in the dataset to predict Yield. This model was much stronger with an R_sqaured of 94%. It is recommended that farmers start to adopt analytics and models such as this to assist in decision making for their trade. Limitations to this study was that it is unclear whether this data is for a particular crop or many crops - the implementation of this model may not be as accurate if it is used on a crop that is not within the data set. 
