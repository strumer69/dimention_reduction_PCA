#!/usr/bin/env python
# coding: utf-8

# # Principal Component Analysis
# #### Let's discuss PCA! 
# Notice PCA isn't exactly a full machine learning algorithm, but instead an  unsupervised learning algorithm,
# 
# Remember that PCA is just a transformation of your data and attempts to find out what features explain the most variance in your data. For example:
# 
# # Libraries

# In[4]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # The Data
# Let's work with the cancer data set again since it had so many features

# In[5]:


from sklearn.datasets import load_breast_cancer


# In[6]:


cancer = load_breast_cancer()


# In[7]:


cancer.keys()


# In[8]:


print(cancer['DESCR'])


# In[9]:


df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
#(['DESCR', 'data', 'feature_names', 'target_names', 'target'])


# In[10]:


df.head()


# In[11]:


df.info()


# # PCA Visualization
# #### As we've noticed before it is difficult to visualize high dimensional data,
# #### we can use PCA to find the first two principal components, and visualize the data in this new, two-dimensional space, with a single scatter-plot. 
# 
# #### Before we do this though, we'll need to scale our data so that each feature has a single unit variance.

# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


scaler = StandardScaler()
scaler.fit(df)


# In[14]:


scaled_data=scaler.transform(df)


# In[15]:


scaled_data


# In[16]:


type(scaled_data)


# In[17]:


type(scaler)


# In[18]:


scaled_data.shape


# PCA with Scikit Learn uses a very similar process to other preprocessing functions that come with SciKit Learn. We instantiate a PCA object, find the principal components using the fit method, then apply the rotation and dimensionality reduction by calling transform().
# 
# We can also specify how many components we want to keep when creating the PCA object.

# In[19]:


from sklearn.decomposition import PCA


# In[20]:


pca = PCA(n_components=2)


# In[21]:


pca=pca.fit(scaled_data)


# In[22]:


x_pca=pca.transform(scaled_data)


# In[23]:


scaled_data.shape


# In[24]:


x_pca.shape


# #### Great! We've reduced 30 dimensions to just 2! Let's plot these two dimensions out!

# In[25]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[26]:


sns.scatterplot(x=x_pca[:,0],y=x_pca[:,1],hue=cancer['target'])
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')


# In[ ]:




