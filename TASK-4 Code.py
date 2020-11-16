#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
import seaborn as snb
from sklearn.preprocessing import StandardScaler


# In[2]:


# data['merchant_name'] = data['merchant_name'].str.decode('latin1').encode('utf-8')
df = pd.read_csv('T:\PROJETCS\TSF\TASK2\data.csv',encoding='latin1')
df[["region", "region_txt"]].drop_duplicates()


# In[5]:


# df['one']=1
df.head(10)
df6=df.groupby(df["iyear"]).count().reset_index()
df6=df6[['iyear','one']]
df6.rename(columns={'one':'Attacks'},inplace=True)
df6
# df2=df.filter(['iyear'],['one'],axis=1)


# In[6]:


df6.plot(x="iyear", y='Attacks',style='-', figsize=(10,10))
plt.title('Year vs Attacks')  
plt.xlabel('Year')  
plt.ylabel('Attacks')  
# plt.xticks(np.arange(min('iyear'), max('iyear'), 1))
x_ticks = np.arange(1970, 2018, 5)
plt.xticks(x_ticks)
plt.show()


# In[7]:


plt.rcParams['figure.figsize']=[10,10]
# y_ticks = np.arange(0, 18000, 2000)
# plt.yticks(y_ticks)
df6.plot.bar(x="iyear", y='Attacks',width=0.75,)
plt.title('Year vs Attacks')  
plt.xlabel('Year')  
plt.ylabel('Attacks')  

# plt.xticks(np.arange(min('iyear'), max('iyear'), 1))

# plt.show()


# In[8]:


df['attacks']=df6['Attacks']


# In[10]:


# df['one']=1
df.head(10)
# df4=df.groupby(["iyear","region_txt"]).size().reset_index().groupby('region_txt')[[0]].max()
df5=df.groupby(df["region_txt"]).count().reset_index()
# df4.rename( columns={'Unnamed: 0':'new column name'}, inplace=True )
# df4.rename(columns={'region_txt':'Attacks'},inplace=True,errors="raise")
df5.reset_index()

# df4=df3[['iyear','one']]


# In[13]:


# df['two']=1
df.head(10)
df7=df.groupby(df["region_txt"]).count().reset_index()
df7=df7[['region_txt','two']]
df7.rename(columns={'two':'Attacks'},inplace=True)
df7


# In[14]:


plt.rcParams['figure.figsize']=[10,10]
# y_ticks = np.arange(0, 18000, 2000)
# plt.yticks(y_ticks)
barchart=df7.plot.bar(x="region_txt", y='Attacks',width=0.75)
# reg=df7["region_txt"]
# atk=df7['Attacks']
# barchart=plt.bar(reg,atk,label="REG VS ATK")
# plt.title('Country vs Attacks')  
plt.xlabel('Country')  
plt.ylabel('Attacks') 
# ax.invert_yaxis() 


# In[15]:


df


# In[16]:


df[["attacktype1_txt", "attacktype1"]].drop_duplicates()
df["attacktype1_txt"].value_counts()


# In[18]:


df["attacktype1_txt"].value_counts()


# In[19]:


df[["region", "region_txt"]].drop_duplicates()


# In[20]:


df["region_txt"].value_counts()


# In[21]:


d_west = df["nkill"][((df['region'] == 1) | (df['region']== 8))]
d_west = [k for k in d_west if k > 0]
d_west = [k for k in d_west if k < 100]
d_west
prior_mean = np.mean(d_west)
prior_std = np.std(d_west)

print(prior_mean, prior_std)


# In[22]:


us_aa_o = df["nkill"][((df['country'] == 217) & (df['attacktype1']== 2))]
us_aa_o = [k for k in us_aa_o if k > 0]


# In[23]:


we_aa_o = df["nkill"][((df['region_txt'] == "Western Europe") & (df['attacktype1']== 2))]
we_aa_o = [k for k in we_aa_o if k > 0]


# In[24]:


print(np.mean(us_aa_o), np.mean(we_aa_o))


# In[30]:


def find_1993(df):
    bombs = []
    regions = df["region_txt"].unique()
    for r in regions:
        region = df[df["region_txt"] == r]
        region = pd.crosstab(index=region["iyear"], columns=[region["attacktype1_txt"]],margins=True)
        region.reset_index(inplace=True)
        region.drop(region.tail(1).index, inplace=True)
        region["iyear"] = region["iyear"].map(lambda x: int(x))
        region = region[(region["iyear"] >= 1983) & (region["iyear"] <= 2003)]
        year_BE = region[["iyear", "Bombing/Explosion"]]
        
        with pm.Model() as reg:
            std = pm.HalfNormal("std", sd=10) # std tf the residuals
            intercept = pm.Normal("intercept", mu=0, sd=10) #beta_0
            beta = pm.Normal("beta", mu=0, sd=10) #beta_1
            E_BE = pm.Normal("E_BE", mu=intercept + (beta * year_BE["iyear"].values), sd=std, observed=year_BE["Bombing/Explosion"].values)
            
        with reg:
            map_estimate = pm.find_MAP()
            
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(year_BE["iyear"].values, year_BE["Bombing/Explosion"].values, color="cyan",alpha=0.5)
        ax.plot(year_BE["iyear"].values, map_estimate['intercept'] + year_BE["iyear"].values*map_estimate['beta'])
        plt.title("region %r" %r)
        plt.show()
        
        y = map_estimate['intercept'] + year_BE["iyear"].values*map_estimate['beta']
        pred_df = pd.DataFrame()
        pred_df["iyear"] = year_BE["iyear"]
        pred_df["pred"] = y
        pred_1992 = pred_df[(pred_df["iyear"] == 1992) | (pred_df["iyear"] == 1994)]
        bombs_1992_r = pred_1992["pred"].mean()
        bombs.append(bombs_1992_r)
    return bombs


# In[31]:


bombs = find_1993(df)


# In[36]:


def find_2003(df):
    bombs1 = []
    regions = df["region_txt"].unique()
    for r in regions:
        region = df[df["region_txt"] == r]
        region = pd.crosstab(index=region["iyear"], columns=[region["attacktype1_txt"]],margins=True)
        region.reset_index(inplace=True)
        region.drop(region.tail(1).index, inplace=True)
        region["iyear"] = region["iyear"].map(lambda x: int(x))
        region = region[(region["iyear"] >= 1993) & (region["iyear"] <= 2013)]
        year_BE = region[["iyear", "Bombing/Explosion"]]
        
        with pm.Model() as reg:
            std = pm.HalfNormal("std", sd=10) # std tf the residuals
            intercept = pm.Normal("intercept", mu=0, sd=10) #beta_0
            beta = pm.Normal("beta", mu=0, sd=10) #beta_1
            E_BE = pm.Normal("E_BE", mu=intercept + (beta * year_BE["iyear"].values), sd=std, observed=year_BE["Bombing/Explosion"].values)
            
        with reg:
            map_estimate = pm.find_MAP()
            
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(year_BE["iyear"].values, year_BE["Bombing/Explosion"].values, color="cyan",alpha=0.5)
        ax.plot(year_BE["iyear"].values, map_estimate['intercept'] + year_BE["iyear"].values*map_estimate['beta'])
        plt.title("region %r" %r)
        plt.show()
        
        y = map_estimate['intercept'] + year_BE["iyear"].values*map_estimate['beta']
        pred_df = pd.DataFrame()
        pred_df["iyear"] = year_BE["iyear"]
        pred_df["pred"] = y
        pred_2002 = pred_df[(pred_df["iyear"] == 2002) | (pred_df["iyear"] == 2004)]
        bombs_2002_r = pred_2002["pred"].mean()
        bombs1.append(bombs_2002_r)
    return bombs1


# In[37]:


bombs1 = find_2003(df)


# In[38]:


bombs1


# In[35]:


bombs


# In[39]:


sum(bombs)


# In[40]:


with pm.Model() as model:
    us_aa_mean = pm.Normal('US_Armed_Assault_Mean', prior_mean, sd=50)
    we_aa_mean = pm.Normal('WE_Armed_Assault_Mean', prior_mean, sd=50)
    
    us_aa_std = pm.HalfNormal('US_Armed_Assault_STD',sd=prior_std)
    we_aa_std = pm.HalfNormal('WE_Armed_Assault_STD',sd=prior_std)
    
    US_AA = pm.Normal('US_Armed_Assault', mu = us_aa_mean, sd = us_aa_std, observed =us_aa_o )
    US_BB = pm.Normal('WE_Armed_Assault', mu = we_aa_mean, sd = we_aa_std, observed = we_aa_o)
    
    mean_delta = pm.Deterministic('mean_delta', us_aa_mean - we_aa_mean)
    std_delta = pm.Deterministic('std_delta',us_aa_std - we_aa_std)
    effect_size = pm.Deterministic('effect_size', mean_delta / np.sqrt((us_aa_std**2 + us_aa_std**2)/2))


# In[43]:


with model:
    trace = pm.sample(1000)


# In[44]:


pm.plot_posterior(trace,
                  varnames=['US_Armed_Assault_Mean', 'WE_Armed_Assault_Mean',
                            'US_Armed_Assault_STD', 'WE_Armed_Assault_STD'],
                  color='#87ceeb');


# In[45]:


pm.plot_posterior(trace,
                  varnames=['mean_delta','std_delta','effect_size'],
                  color='#87ceeb', ref_val=0);


# In[ ]:




