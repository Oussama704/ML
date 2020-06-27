#!/usr/bin/env python
# coding: utf-8

# In[188]:


import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
import numpy as np
import seaborn as sns


# # Visualiser les données

# In[95]:


df=pd.read_csv('train.csv',delimiter=";")
df.head()


# In[96]:


df.describe()
df.shape


# In[97]:


df.columns


# In[98]:


distribution=df.SalePrice
distribution.plot.hist(bins=50)


# In[99]:


df.corr()


# In[100]:


corr_matrice=df.corr()
corr_price=corr_matrice['SalePrice']
sns.heatmap(corr_matrice)


# In[101]:


corr_matrice[corr_matrice['SalePrice'].abs()>=0.5]['SalePrice'].sort_values(ascending=False)
#Les variables dont l'objjectif y dépend


# In[102]:


plt.scatter(df['OverallQual'],df['SalePrice'])
plt.xlabel('OverallQual')
plt.ylabel('SalePrice')
plt.show()
# cette visualisation montre la forte corrélation entre la SalePrice et OverallQual


# In[103]:


plt.scatter(df['GrLivArea'],df['SalePrice'])
plt.xlabel('GrLivArea ')
plt.ylabel('SalePrice')
plt.show()

#dès que le GrLivArea sera plus grande moins que la corrélation sera forte ce qui bien significatif dans notre données 


# # Pre_Processing 

# In[105]:


df.select_dtypes(include=['object'])


# In[106]:


df['SalePrice'].plot.hist(bins=50)
#On remarque qu'on a une valeur extreme donc on va faire un ilternative avec la fonction log


# In[107]:


#log
np.log(df['SalePrice']).plot.hist(bins=50)


# In[108]:


#Ajouter une autre variable
df['SalePrice_Log']=np.log(df['SalePrice'])
df.head()


# In[109]:


#Vleur Catégorique


# In[110]:


#Selectionner les Valeurs Catégoriques.
categ_var=df.select_dtypes(include=['object'])
categ_var_columns=categ_var.columns
categ_var.head()


# In[111]:


#transformer les valeurs catégoriques en des valeurs numériques.
categ_var_update=pd.get_dummies(df.select_dtypes(include=['object']))
categ_var_update.head()


# In[112]:


num_var=df[[var for var in df if var not in categ_var]]
num_var.head()


# In[113]:


new_data=pd.concat([categ_var_update,num_var],axis=1)


# In[114]:


new_data


# In[115]:


#Nombre de coloune qu'on a ajouter
new_data.shape


# In[116]:


df.shape


# In[117]:


#MISSING VALUES


# In[118]:


#Visualiser les données manquantes.
#Trur signif que la variable a une valeur inconnu
df.isna().head()


# In[119]:


new_data[[var for var in new_data if var not in categ_var_update]].fillna(0.0)


# # Modeling 
# 

# In[120]:


#On a plus de variable inconnu 
new_data.isna().head()


# In[160]:


from sklearn.preprocessing import normalize
data=new_data.copy().fillna(0.0)
data.head()


# In[161]:


target=data.pop('SalePrice_Log')
target.head()


# In[162]:


del data['SalePrice']


# In[163]:


features=data
features


# In[164]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)


# In[165]:


from sklearn.linear_model import LinearRegression, RidgeCV
model=RidgeCV()

model.fit(x_train,y_train)


# In[134]:


'SalePrice' in features


# In[191]:


y_pred = model.predict(x_test)
x_train.shape


# In[168]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)


# In[187]:


plt.figure(figsize=(40,20))
plt.scatter(y_test,y_pred,c='red')
plt.scatter(y_train,model.predict(x_train),c='g',alpha=0.3)
#Le cas ou y_test=y_pred coloré en blue
plt.plot(y_test,y_test)

plt.show()


# # End

# In[ ]:




