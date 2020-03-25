#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#read and combine weekly billing data
df1=pd.read_excel('WklyBilling_20170625_20180623 (2).xlsx')
df2=pd.read_excel('WklyBilling_20180624_20190622.xlsx')
df3= pd.read_excel('WklyBilling_20190623_20191123.xlsx')
df=pd.concat([df1,df2,df3])


# In[3]:


#drop unuseful features
df.drop(['Scratchers Full Pack Returns','Scratcher Partial Pack Returns','Validations','Commisions','Adjustments',
        'Promotions','Bank Card Sales','End Date','Sweep Date'],axis=1,inplace=True)


# In[4]:


#drop invalid retailer num
df=df[df['Retailer Number']!=0]


# In[5]:


#drop negative sweep amount
df=df[df['Sweep Amount']>0]


# In[6]:


#read NSF list
nf= pd.read_excel('oldlist.xlsx')
nf.head()


# In[7]:


# create a dataframe with total weeky biling volume 
# and average weekly biling amount for each retailer
dfsub= df[['Retailer Number','Sweep Amount']]
dfc=dfsub.groupby(['Retailer Number']).count().reset_index()
dfc.columns = ['Retailer Number','Total_weekly_sweep']
dfsub=dfsub.groupby(['Retailer Number']).mean().reset_index()
dfsub=pd.merge(dfsub, dfc, left_on='Retailer Number',right_on = 'Retailer Number',how='left')


# In[8]:


dfsub.head()


# In[9]:


#get a list of unique NSF retailer number
nsf=pd.DataFrame(set(nf['Retailer No']))
nsf.columns=['Retailer Num']
nsf.head()


# In[10]:


#merge NSF list to weekly billing data
merge=pd.merge(dfsub, nsf, left_on="Retailer Number",right_on="Retailer Num",how='left')


# In[11]:


#check which retailer has NSF history
merge['NSF'] = merge['Retailer Num'].apply(lambda x: 1 if not pd.isnull(x) else 0)


# In[12]:


#drop duplicated retailer number
merge.drop(['Retailer Num'],axis=1,inplace=True)


# In[13]:


merge.head(20)


# In[14]:


#read the chain&retailer
#this dataset consisted with all retailer datasetd that AZL shared
chaindata = pd.read_excel('ASU Retailers with Chain Detail.xlsx')
chaindata.drop(['BusinessName','Street1','Street2','LicenseCancelled'],axis=1,inplace=True)


# In[15]:


chaindata.head()


# In[16]:


#read a list of counties in AZ and corresponding zip code
county=pd.read_csv('counties.csv')
chaindata=chaindata.merge(county,left_on='Zip_code',right_on='zipcode',how='left')


# In[17]:


#for the retailer's zipcode has no matching result, count as Out of AZ
chaindata['county'].fillna(value='Out of AZ',inplace=True)


# In[18]:


chaindata.head()


# In[19]:


#drop zipcode, it's useless at this point
chaindata.drop(['zipcode','Zip_code'],axis=1,inplace=True)


# In[20]:


#merge the chain&retailer dataset with weekly billing dataset
merge1=pd.merge(merge, chaindata, left_on="Retailer Number",
                right_on="RetailerNum",how='left')


# In[21]:


#only keep the billing data that has a matching retailer records
merge1=merge1[merge1['RetailerNum'].notnull()]


# In[22]:


merge1.head(50)


# In[23]:


merge1.drop(['RetailerNum'],axis=1,inplace=True)


# In[24]:


merge1['RetailerStatusID'] = merge1['RetailerStatusID'].astype(int)
merge1['Sales_terminals_count']=merge1['Sales_terminals_count'].astype(int)
merge1['Business_type_code']=merge1['Business_type_code'].astype(int)


# In[25]:


#calculate year of partnership for each retailer
merge1['year'] = pd.DatetimeIndex(merge1['LicenseIssued']).year
merge1['year']=merge1['year'].apply(lambda x:2020-x)
merge1.rename(columns={'year':'year_parterned'},inplace=True)
merge1.drop(['LicenseIssued'],axis=1,inplace=True)


# In[26]:


#drop city and chainname
merge1.drop(['City'],axis=1,inplace=True)
merge1.drop(['ChainName'],axis=1,inplace=True)


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns
business_type = merge1.Business_type_code.value_counts()
sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=business_type.index, y=business_type.values)
plt.show()


# In[28]:


nsf = merge1.NSF.value_counts()
sns.set_style("darkgrid")
plt.figure(figsize=(10,4))
sns.barplot(x=nsf.index, y=nsf.values)
plt.show()


# In[29]:


merge1['NSF'].value_counts()


# In[30]:


merge1.head()


# In[31]:


year1_10 = merge1.year_parterned[(merge1.year_parterned <= 10) & (merge1.year_parterned >= 1)]
year11_20 = merge1.year_parterned[(merge1.year_parterned <= 20) & (merge1.year_parterned >= 11)]
year21_30 = merge1.year_parterned[(merge1.year_parterned <= 30) & (merge1.year_parterned >= 21)]
year31above = merge1.year_parterned[merge1.year_parterned >= 31]

x = ["1-10","11-20","21-30","31+"]
y = [len(year1_10.values),len(year11_20.values),len(year21_30.values),len(year31above.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=x, y=y, palette="rocket")
plt.title("Number of Retailers and Years Paternered")
plt.xlabel("year")
plt.ylabel("Number of retailer")
plt.show()


# In[33]:


merge1['county'].value_counts()[:10].plot(kind='barh')


# In[34]:


merge1[['Business_type_code', 'Sweep Amount']].groupby(['Business_type_code'], as_index=False).mean().sort_values(by='Sweep Amount', ascending=False)


# In[35]:


merge1[['county', 'Sweep Amount']].groupby(['county'], as_index=False).mean().sort_values(by='Sweep Amount', ascending=False)


# In[36]:


merge1[['Business_type_code', 'Total_weekly_sweep']].groupby(['Business_type_code'], as_index=False).mean().sort_values(by='Total_weekly_sweep', ascending=False)


# In[37]:


merge1[['NSF', 'Sweep Amount']].groupby(['NSF'], as_index=False).mean().sort_values(by='Sweep Amount', ascending=False)


# In[38]:


merge1[['NSF', 'Total_weekly_sweep']].groupby(['NSF'], as_index=False).mean().sort_values(by='Total_weekly_sweep', ascending=False)


# In[39]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown='ignore')

ohe_df = pd.DataFrame(ohe.fit_transform(merge1[['Business_type_code']]).toarray())


# In[40]:


dum=pd.get_dummies(merge1['county'])


# In[41]:


merge1 = merge1.join(ohe_df)
merge1 = merge1.join(dum)
merge1.drop(['county'],axis=1,inplace=True)


# In[42]:


btc=merge1['Business_type_code']
merge1.drop(['Business_type_code'],axis=1,inplace=True)


# In[43]:


merge1.isnull().sum()


# In[44]:


from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler


# In[45]:


columnTransformer1 = ColumnTransformer([('num', MinMaxScaler(), [1])], remainder='passthrough')


# In[46]:


merge1['Sweep Amount']=columnTransformer1.fit_transform(merge1)


# In[47]:


columnTransformer2 = ColumnTransformer([('num1', MinMaxScaler(), [2])], remainder='passthrough')


# In[48]:


merge1['Total_weekly_sweep']=columnTransformer2.fit_transform(merge1)


# In[49]:


merge1=merge1[merge1['Sweep Amount'].notnull()]


# In[51]:


retailernum=merge1['Retailer Number']
merge1.drop(['Retailer Number'],axis=1,inplace=True)


# In[54]:


merge1.columns


# In[55]:


merge1.columns=['Sweep Amount','Total_weekly_sweep','NSF','RetailerStatusID',
                'Sales_terminals_count','year_parterned',"bc0","bc1","bc2","bc3",
                "bc4","bc5","bc6","bc7","bc8","bc9","bc10","bc11","bc12","bc13",
               'Apache',               'Cochise',
                    'Coconino',                  'Gila',
                      'Graham',              'Greenlee',
                      'La Paz',              'Maricopa',
                      'Mohave',                'Navajo',
                   'Out of AZ',                  'Pima',
                       'Pinal',            'Santa Cruz',
                     'Yavapai',                  'Yuma']


# In[56]:


merge1[["bc0","bc1","bc2","bc3",
                "bc4","bc5","bc6","bc7","bc8","bc9","bc10","bc11","bc12","bc13"]]=merge1[["bc0","bc1","bc2","bc3",
                "bc4","bc5","bc6","bc7","bc8","bc9","bc10","bc11","bc12","bc13"]].fillna(value=0)


# In[57]:


from sklearn.cluster import KMeans
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(merge1)
    Sum_of_squared_distances.append(km.inertia_)


# In[58]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()
#k=4


# In[59]:


# Import the KElbowVisualizer method 
from yellowbrick.cluster import KElbowVisualizer

# Instantiate a scikit-learn K-Means model
model = KMeans(random_state=0)

# Instantiate the KElbowVisualizer with the number of clusters and the metric 
visualizer = KElbowVisualizer(model, k=(2,6), metric='silhouette', timings=False)

# Fit the data and visualize
visualizer.fit(merge1)    
visualizer.poof()   


# In[74]:


kmeans = KMeans(n_clusters=3, random_state=0).fit(merge1)


# In[75]:


# Get the cluster centroids
print(kmeans.cluster_centers_)
    
# Get the cluster labels
print(kmeans.labels_)


# In[76]:


# Calculate silhouette_score
from sklearn.metrics import silhouette_score

print(silhouette_score(merge1, kmeans.labels_))


# In[77]:


kmeans.cluster_centers_.shape


# In[78]:


from sklearn.cluster import DBSCAN
from sklearn import preprocessing


# In[79]:


def plot_corr(df):
    corr = df.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[80]:


plot_corr(merge1)


# In[67]:


#nomalized kmeans
normalized_vectors = preprocessing.normalize(merge1)
scores = [KMeans(n_clusters=i+2).fit(normalized_vectors).inertia_ for i in range(10)]
sns.lineplot(np.arange(2, 12), scores)
plt.xlabel('Number of clusters')
plt.ylabel("Inertia")
plt.title("Inertia of Cosine k-Means versus number of clusters")


# In[81]:


#normalized kmeans
normalized_kmeans = KMeans(n_clusters=3)
normalized_kmeans.fit(normalized_vectors)


# In[82]:


#DBSCAN
min_samples = merge1.shape[1]+1 
dbscan = DBSCAN(eps=3.5, min_samples=min_samples).fit(merge1)


# In[83]:


from sklearn.decomposition import PCA
def prepare_pca(n_components, data, kmeans_labels):
    names = ['x', 'y', 'z']
    matrix = PCA(n_components=n_components).fit_transform(data)
    df_matrix = pd.DataFrame(matrix)
    df_matrix.rename({i:names[i] for i in range(n_components)}, axis=1, inplace=True)
    df_matrix['labels'] = kmeans_labels
    
    return df_matrix

pca_merge1 = prepare_pca(3, merge1, normalized_kmeans.labels_)
sns.scatterplot(x=pca_merge1.x, y=pca_merge1.y, hue=pca_merge1.labels, palette="Set2")


# In[84]:


print('kmeans: {}'.format(silhouette_score(merge1, kmeans.labels_, metric='euclidean')))
print('Cosine kmeans: {}'.format(silhouette_score(normalized_vectors, normalized_kmeans.labels_, metric='cosine')))


# In[89]:


from sklearn.cluster import AgglomerativeClustering


# In[98]:


clustering = AgglomerativeClustering(n_clusters=3,
                                     affinity='cosine',
                                    linkage='average').fit(merge1)


# In[99]:


clustering.labels_


# In[100]:


print(silhouette_score(merge1, clustering.labels_))


# In[ ]:




