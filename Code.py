#!/usr/bin/env python
# coding: utf-8

# In[23]:


#read a CSV file
import pandas as pd
#use to look at data frame(ex: .info)
import numpy as np
#used for heatmap to evaluate nulls
import seaborn as sns
#when addressing null values with a data set
import math as ma


# In[2]:


data = pd.read_csv('C:/Users/kavya.patel/Documents/DataScience/House Prices - Advanced Regression Techniques(Kaggle)/train.csv')


# In[3]:


#look at first 5 rows of data
data.head()


# In[4]:


#look at shape of dataset
data.shape


# In[5]:


#see all rows in data set
pd.set_option("display.max.columns", None)
#change all decimal places to 2
pd.set_option("display.precision",2)


# In[6]:


#this displays the efect of te functions int eh previous cell
data.tail()


# In[7]:


data.info()


# In[8]:


#https://seaborn.pydata.org/generated/seaborn.heatmap.html
sns.heatmap(data.isnull(),yticklabels=False,cbar=False)


# In[ ]:


# I feel that these feilds should get deleted 
# since they have so many null values
"""
Alley         91 non-null object
FireplaceQu   770 non-null object
PoolQC        7 non-null object
Fence         281 non-null object
MiscFeature   54 non-null object
"""


# In[9]:


#looking at basic statistics 
data.describe()
#this will only look at numeric colums 
#if you want to do it for object colums you will need to add (include=object)
    #this will give you descriptive statistics


# In[10]:


#feature enginnering
#in our heatmap lotFrontage is the first one that has missing nulls
#lets count how many null values it has
data.isnull().sum()
#LotFrontage only has 259 null values
#that is not enough to just get rid of the column
#try to see if you can fill in the nulls with another value


# In[12]:


#fill in any null values in the LotFrontage feild with 
#the average of all the non-null values in the LotFrontage feild
data['LotFrontage']=data['LotFrontage'].fillna(data['LotFrontage'].mean())


# In[13]:


#next on our heatmap is Alley!!
#Look at the result of block 19
#Alley has 1369 null values...wayyyy more then LotFrontage
#Lets calculate the % of null values Alley has
Alley_null_per = (1369/len(data))*100
print(Alley_null_per)
# wow 93% of the values are null
#we should get rid of this feild


# In[14]:


#https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html
#axis is askign if you want to drop this from the index
data.drop(['Alley'],axis=1,inplace=True)


# In[15]:


#BsmtCond = Basement condition
#https://numpy.org/doc/stable/reference/generated/numpy.sum.html
BsmtCond_null_per = ((data['BsmtCond'].isnull().sum())/len(data))*100
print(BsmtCond_null_per)
#only 2% of the values are null so lets do null imputation!


# In[ ]:


#lets double check if our null values are still there
data['BsmtCond_int'].isnull().sum()
#oh wait why do we have zero nulls??
#Python will consider an object labeled as 'NA' as the same thing as a null
#so when we manually converted the column objects to intergers...
#all the null values(2%) were converted to a integers
#we do NOT want that because according to our research we woudl like to do our null imputations based on mode
#at the moment our nulls are recognized as 0 which stands fo no basement.
#we want our nulls to be Ex which is our mode
--------------------------------
#So what we can do is convert our null values in the 'BsmtCond' to "NAN"
#this will allow for NAN values to stay NAN when we run our for loop again


# In[ ]:


#how do we know what to use in our null imputation?
#mode, medium, mean
# we will look at the distribution of the values
#sns.distplot(data['BsmtCond'])
#oh no we can not plot the distribution because the values are objects
#we can not simply convert them to integers becuase python does not have that ability
# we will have to manuually create a column


# In[ ]:


#create a array rather then a empty string because when doing the following for loop you will override the previous number 
#and so your string will end up wth just one number
#CODE-----------------------
"""
data['BsmtCond_int'] = ""
for v in data['BsmtCond']:
    if v == 'NA':
        data['BsmtCond_int'] = 0
    elif v == 'Ex':
        data['BsmtCond_int'] = 1
    elif v == 'Gd':
        data['BsmtCond_int'] = 2
    elif v == 'TA':
        data['BsmtCond_int'] = 3
    elif v == 'Fa':
        data['BsmtCond_int'] = 4
    elif v == 'Po':
        data['BsmtCond_int'] = 5
"""


# In[ ]:


#we need to create a empty array that is NOT apart of the data becuase python did not want a empty array in the data set
#data['BsmtCond_int'] = [] --this will give you error
#so first we create a empty array OUTSIDE the dataset
#then we populate it
#then we add hte populated array into the dataset
#CODE-----------------------
"""
BsmtCond_int = []
for v in data['BsmtCond']:
    if v == 'Ex':
        BsmtCond_int.append(5)
    elif v == 'Gd':
        BsmtCond_int.append(4)
    elif v == 'TA':
        BsmtCond_int.append(3)
    elif v == 'Fa':
        BsmtCond_int.append(2)
    elif v == 'Po':
        BsmtCond_int.append(1)
    elif v == 'NA':
        BsmtCond_int.append(0)
"""
#you will still have an error because you have not accounted for the nulls in your for loop so the array is
#a different lenght then the dataset
#you need to add in another line that assigns your nulls something even if it is another null


# In[25]:


#create a for loop that asigns integer values to each possible outcomes
#COME BACK TO THIS AS TO WHY IT DID NOT WORK
"""
BsmtCond_int = []
for v in data['BsmtCond']:
    if v == 'Ex':
        BsmtCond_int.append(5)
    elif v == 'Gd':
        BsmtCond_int.append(4)
    elif v == 'TA':
        BsmtCond_int.append(3)
    elif v == 'Fa':
        BsmtCond_int.append(2)
    elif v == 'Po':
        BsmtCond_int.append(1)
    elif v == 'NA':
        BsmtCond_int.append(0)
    elif ma.isnan(v) == 'True':
         BsmtCond_int.append('nan')
"""


# In[52]:


BsmtCond_int = []
v = data['BsmtCond'].to_string()
if v == "TA":
    print('true')
#     #BsmtCond_int.append(5)
# #elif v.str == 'Gd':
#     BsmtCond_int.append(4)
# #elif v.str == 'TA':
#     BsmtCond_int.append(3)
# #elif v.str == 'Fa':
#     BsmtCond_int.append(2)
# #elif v.str == 'Po':
#     BsmtCond_int.append(1)
# #elif v.str == 'NA':
#     BsmtCond_int.append(0)
# #elif pd.isna(v.str):
#     BsmtCond_int.append('nan') 


# In[50]:


len(v)


# In[45]:


print(v)


# In[35]:


len( BsmtCond_int)


# In[37]:


len(data.BsmtCond)


# In[ ]:


#create a empty array
#it does not want a regular array it wants a 
data['BsmtCond_int'] = BsmtCond_int


# In[ ]:


print(data[['BsmtCond','BsmtCond_int']])


# In[ ]:


sns.distplot(data['BsmtCond_int'])
#wow it looks like the distribution only shows one value
#We need to double check what is going on


# In[ ]:


#checking how many count per values within a feild 
data['BsmtCond'].value_counts()


# In[ ]:


TA_per = (1311/len(data['BsmtCond']))*100
Gd_per = (65/len(data['BsmtCond']))*100
Fa_per = (45/len(data['BsmtCond']))*100
Po_per = (2/len(data['BsmtCond']))*100
print(TA_per,Gd_per,Fa_per,Po_per)
#three of our 4 values are under 50% so lets use mode rather then mean for null imputations


# In[53]:


get_ipython().system('git')

