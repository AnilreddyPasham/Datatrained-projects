#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\practice\baseball.csv")


# In[4]:


data.head()


# In[5]:


data.columns


# In[6]:


data.isnull().sum()


# In[9]:


count = data['W'].value_counts()
sns.set_context(font_scale=1.5)
plt.figure(figsize=(8,7))
sns.barplot(count.index, count.values, alpha=0.8, palette="bwr")
plt.ylabel('Count')
plt.xlabel('W')
plt.title('Number of poisonous/edible mushrooms')
plt.show()


# In[12]:


plt.hist(data['W'])
plt.xlabel('wins')
plt.title('histogram of wins')


# In[17]:


data['W'].mean()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


def assign_win_bins(w):
    if w >= 50 and w <= 59:
        return 1
    if w >= 60 and w <= 69:
        return 2
    if w >= 70 and w <= 79:
        return 3
    if w >= 80 and w <= 89:
        return 4
    if w >= 90 and w <= 100:
        return 5


# In[21]:


data['win_bins'] = data['W'].apply(assign_win_bins)


# In[ ]:





# In[ ]:





# In[22]:


X = data.drop(['W'], axis=1)  
y = data["W"]

X.head()


# In[25]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, linecolor='black', fmt='.2f')


# In[24]:


sns.scatterplot(x='RA',y='ERA', data=data)


# In[26]:


X.drop(['RA','ERA'],axis=1,inplace=True)


# In[28]:


X.head()


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=40)


# In[32]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X2 = sc.fit_transform(X)
X_test = sc.transform(X_test)


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3, random_state=40)


# In[34]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca=pca.fit_transform(X_train) 
X_test_pca=pca.transform(X_test)
print("Original shape:",X_train.shape)
print('Shape of PCA data:',X_train_pca.shape)


# In[39]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[40]:


lr = LinearRegression(normalize=True)
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)


# In[41]:


mae = mean_absolute_error(y_test, predictions)


# In[42]:


print(mae)


# In[43]:


from sklearn.linear_model import RidgeCV


# In[45]:


rrm = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0), normalize=True)
rrm.fit(X_train, y_train)
predictions_rrm = rrm.predict(X_test)


# In[46]:


mae_rrm = mean_absolute_error(y_test, predictions_rrm)
print(mae_rrm)


# In[ ]:




