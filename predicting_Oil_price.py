# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 06:23:55 2019

@author: elava
"""

# Reset the Console
%reset -f

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
oil_price = pd.read_csv('Oil and Gas 1932-2014.csv', sep=',')
oil_price.shape
oil_price.head(10)
oil_price.tail(10)
# change the cty_name to country
oil_price.rename(columns={'cty_name':'country'},inplace=True)

# Summary Statistics
oil_price.describe()
oil_price.info()

oil_price.columns
oil_price.dtypes

# Null Values
oil_price.isnull().sum() # NaN values
print((oil_price==0).sum()) # Zero values
oil_price.drop_duplicates(inplace=True , keep='first')
oil_price.shape # No duplicate rows

# Exploring data missingness
import missingno as msno
missing_data = oil_price.columns[oil_price.isnull().any()].tolist()
msno.matrix(oil_price[missing_data],color=(0.6, 0.3,0.4))
msno.bar(oil_price[missing_data], color='coral')
msno.heatmap(oil_price[missing_data])

import seaborn as sns
total = oil_price.isnull().sum().sort_values(ascending=False)
percent = (oil_price.isnull().sum()/oil_price.isnull().count()).sort_values(ascending=False)
missing_dataset = pd.concat([total, percent], axis=1, keys = ['Total','Percent'])
f, ax = plt.subplots(figsize=(15,6))
plt.xticks(rotation='90')
sns.barplot(x=missing_dataset.index, y=missing_dataset['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_dataset.head(10)


# Finding missing values in the dataset

# Mark zero values as missing or NaN
#(oil_price==0).sum()
#oil_price.replace(0, np.NaN,inplace=True)
#oil_price.isnull().sum()

#missing_data_id = np.where(oil_price['id'].isnull())

# Drop the null values in id column
oil_price['id'].isnull().sum()
# Some countries are divided and combined. 
# Here data is before that, and ISO-id is changed.
# Oil Production is not done with these countries, so remove them.
#dataset = dataset[pd.notnull(dataset['id'])]
oil_price.dropna(subset=['id'],inplace=True)
oil_price.reset_index(inplace=True)
oil_price.drop(['index'],inplace=True,axis=1)
oil_price.isnull().sum()
oil_price.shape

# KNN Imputation
from fancyimpute import KNN
# fancyimpute remove the column names
data_set = oil_price.drop(['iso3numeric','id','eiacty','sovereign'],axis=1)
dataset_cols = list(data_set)
# Use 5 nearest rows which have a feature to fill in each row missing features
imputed_knn = pd.DataFrame(KNN(k=5).fit_transform(data_set.iloc[:,2:]))
imputed_knn.columns = dataset_cols[2:]

# Checking null values
imputed_knn.isnull().sum() # No NaN values
(oil_price==0).sum()
(imputed_knn==0).sum() # Here No.of Zero's increases [396]

dataset = pd.concat([data_set.iloc[:,:2],imputed_knn],axis=1)
#dataset['year'] = dataset['year'].iloc[,:4]

dataset.head(10)
dataset.info()
dataset.describe()
dataset.shape
dataset.dtypes

# Measures of Dispersion
np.var(dataset)
np.std(dataset)

#dataset['sovereign'].value_counts()

# Skewness and Kurtosis
from scipy.stats import skew, kurtosis, norm
skew(dataset.drop(['country','year'], axis=1))
kurtosis(dataset.drop(['country','year'], axis=1))

''' # Dividing the dataset into 3 parts
part_1 = new_data.iloc[:5000,:]
part_2 = new_data.iloc[5000:10000,:]
part_3 = new_data.iloc[10000:15521,:]

from fancyimpute import NuclearNormMinimization, SoftImpute, BiScaler

# Imputation using nuclearnormminization
impute_part_1_nnm = NuclearNormMinimization().fit_transform(part_1)
impute_part_2_nnm = NuclearNormMinimization().fit_transform(part_2)
impute_part_3_nnm = NuclearNormMinimization().fit_transform(part_3)

new_data_normalized = BiScaler().fit_transform(new_data)
impute_data_softimpute = SoftImpute().fit_transform(new_data_normalized)
# print mean sq. error for the imputation methods above
nnm_mse = ((impute_data_nnm[missing_mask]-new_data[missing_mask]) ** 2).mean()
print('Nuclear norm minimization MSE: %f' % nnm_mse)

softimpute_mse = ((impute_data_softimpute[missing_mask]-new_data[missing_mask])**2).mean()
print("Softimpute MSE: %f" % softimpute_mse) 

knn_mse = ((imputed_dataset_knn[missing_mask]-new_dataset[missing_mask])**2).mean()
print('knnimpute MSE: %f' % knn_mse) '''


# Scipy
import scipy
print('scipy: %s' % scipy.__version__)
# Numpy
import numpy
print('scipy: %s' % numpy.__version__)
# Matplotlib
import matplotlib
print('scipy: %s' % matplotlib.__version__)
# Pandas
import pandas
print('scipy: %s' % pandas.__version__)
# Scikit learn
import sklearn
print('scipy: %s' % sklearn.__version__)
# Statsmodels
import statsmodels
print('scipy: %s' % statsmodels.__version__)

# Line Plot

dataset['oil_prod32_14'].plot()
dataset['oil_prod32_14'].hist()
dataset['oil_prod32_14'].plot(kind='kde')

# Histogram 
plt.hist(dataset['oil_prod32_14']);plt.title('Histogram of Total Oil Production from 1932-2014');plt.xlabel('Total Oil Production (bbl/year)');plt.ylabel('Frequency')
plt.hist(dataset['oil_price_2000']);plt.title('Histogram of Price of Total Oil Production, in 2000 US $');plt.xlabel('Price of Total Oil Production ($ per barrel)');plt.ylabel('Frequency')
plt.hist(dataset['oil_price_nom']);plt.title('Histogram of Nominal Price of Total Oil Production, in US $');plt.xlabel('Nominal Price of Total Oil Production');plt.ylabel('Frequency')
plt.hist(dataset['oil_value_nom']);plt.title('Histogram of Nominal Value of Total Oil Production, in US $');plt.xlabel('Nominal Value of Total Oil Production');plt.ylabel('Frequency')
plt.hist(dataset['oil_value_2000']);plt.title('Histogram of Value of Total Oil Production, in 2000 US $');plt.xlabel('Value of Total Oil Production, 2000');plt.ylabel('Frequency')
plt.hist(dataset['oil_value_2014']);plt.title('Histogram of Value of Total Oil Production, in 2014 US $');plt.xlabel('Value of Total Oil Production, 2014');plt.ylabel('Frequency')
plt.hist(dataset['gas_prod55_14']);plt.title('Histogram of Total Gas Production from 1955-2014 (Billion Cubic Meters)');plt.xlabel('Total Gas Production (BCM)');plt.ylabel('Frequency')
plt.hist(dataset['gas_price_2000_mboe']);plt.title('Histogram of Price of Total Gas Production, in 2000 US $');plt.xlabel('Price of Total Gas Production ($ per mboe)');plt.ylabel('Frequency')
plt.hist(dataset['gas_price_2000']);plt.title('Histogram of Price of Total Gas Production, in 2000 US $');plt.xlabel('Price of Total Gas Production ($ per gallon)');plt.ylabel('Frequency')
plt.hist(dataset['gas_price_nom']);plt.title('Histogram of Nominal Price of Total Gas Production, in US $');plt.xlabel('Nominal Price of Total Gas Production');plt.ylabel('Frequency')
plt.hist(dataset['gas_value_nom']);plt.title('Histogram of Nominal Value of Total Gas Production, in US $');plt.xlabel('Nominal Value of Total Gas Production');plt.ylabel('Frequency')
plt.hist(dataset['gas_value_2000']);plt.title('Histogram of Value of Total Gas Production, in 2000 US $');plt.xlabel('Value of Total Gas Production');plt.ylabel('Frequency')
plt.hist(dataset['gas_value_2014']);plt.title('Histogram of Value of Total Gas Production, in 2014 US $');plt.xlabel('Value of Total Gas Production');plt.ylabel('Frequency')
plt.hist(dataset['oil_gas_value_nom']);plt.title('Histogram of Nominal Value of Total Oil and Gas Production, in US $');plt.xlabel('Nominal Value of Total Oil and Gas Production');plt.ylabel('Frequency')
plt.hist(dataset['oil_gas_value_2000']);plt.title('Histogram of Value of Total Oil and Gas Production, in 2000 US $');plt.xlabel('Value of Total Oil and Gas Production ($ per mboe)');plt.ylabel('Frequency')
plt.hist(dataset['oil_gas_value_2014']);plt.title('Histogram of Value of Total Oil and Gas Production, in 2014 US $');plt.xlabel('Value of Total Oil and Gas Production ($ per mboe)');plt.ylabel('Frequency')
plt.hist(dataset['oil_gas_valuePOP_nom']);plt.title('Histogram of Nominal Value of Total Oil and Gas POP, in US $');plt.xlabel('Nominal Value of Total Oil and Gas POP');plt.ylabel('Frequency')
plt.hist(dataset['oil_gas_valuePOP_2000']);plt.title('Histogram of Value of Total Oil and Gas POP, in 2000 US $');plt.xlabel('Value of Total Oil and Gas POP ($ per mboe)');plt.ylabel('Frequency')
plt.hist(dataset['oil_gas_valuePOP_2014']);plt.title('Histogram of Value of Total Oil and Gas POP, in 2014 US $');plt.xlabel('Value of Total Oil and Gas POP ($ per mboe)');plt.ylabel('Frequency')
plt.hist(dataset['oil_exports']);plt.title('Histogram of Total Amount of Oil Exported');plt.xlabel('Amount of Oil Exported (bbl/year)');plt.ylabel('Frequency')
plt.hist(dataset['net_oil_exports']);plt.title('Histogram of Net Amount of Oil Exported');plt.xlabel('Net Amount of Oil Exported');plt.ylabel('Frequency')
plt.hist(dataset['net_oil_exports_mt']);plt.title('Histogram of Net Amount of Oil Exported, in mt');plt.xlabel('Net Amount of Oil Exported (million tonnes)');plt.ylabel('Frequency')
plt.hist(dataset['net_oil_exports_value']);plt.title('Histogram of Value of Net Oil Export, in US $');plt.xlabel('Value of Net Oil Export');plt.ylabel('Frequency')
plt.hist(dataset['net_oil_exports_valuePOP']);plt.title('Histogram of Net Oil Export POP, in US $');plt.xlabel('Net Oil Export POP');plt.ylabel('Frequency')
plt.hist(dataset['gas_exports']);plt.title('Histogram of Total Amount of Gas Exported, in Cubic meters');plt.xlabel('Amount of Gas Exported (cubic meters)');plt.ylabel('Frequency')
plt.hist(dataset['net_gas_exports_bcf']);plt.title('Histogram of Net Amount of Gas Exports, in Billion cubic feet');plt.xlabel('Amount of Gas Export (bcf)');plt.ylabel('Frequency')
plt.hist(dataset['net_gas_exports_mboe']);plt.title('Histogram of Net Amount of Gas Exports, in mboe');plt.xlabel('Amount of Gas Export (mboe)');plt.ylabel('Frequency')
plt.hist(dataset['net_gas_exports_value']);plt.title('Histogram of Value of Net Gas Exports, in US $');plt.xlabel('Value of Net Gas Export');plt.ylabel('Frequency')
plt.hist(dataset['net_gas_exports_valuePOP']);plt.title('Histogram of Value of Net Gas Export POP, in US $');plt.xlabel('Value of Net Gas Export POP');plt.ylabel('Frequency')
plt.hist(dataset['net_oil_gas_exports_valuePOP']);plt.title('Histogram of Value of Net Oil and Gas Export POP, in US $');plt.xlabel('Value of Net Oil and Gas Export POP');plt.ylabel('Frequency')
plt.hist(dataset['population']);plt.title('Histogram of Population');plt.xlabel('Population');plt.ylabel('Frequency')
plt.hist(dataset['pop_maddison']);plt.title('Histogram of Maddison Estimated Population');plt.xlabel('Maddison Population');plt.ylabel('Frequency')
plt.hist(dataset['mult_nom_2000'])
plt.hist(dataset['mult_nom_2014'])

# Count Plot
#sns.countplot(dataset['sovereign']).set_title('Countplot of sovereign')

# Normal Q-Q plot
plt.plot(dataset['oil_prod32_14']);plt.legend('oil_prod32_14')


plt.plot(dataset[['oil_prod32_14','oil_price_2000','oil_price_nom','oil_value_nom','oil_value_2000',
                  'oil_value_2014','gas_prod55_14','gas_price_2000_mboe','gas_price_2000','gas_price_nom',
                  'gas_value_nom','gas_value_2000','gas_value_2014','oil_gas_value_nom','oil_gas_value_2000',
                  'oil_gas_value_2014','oil_gas_valuePOP_nom','oil_gas_valuePOP_2000','oil_gas_valuePOP_2014',
                  'oil_exports','net_oil_exports','net_oil_exports_mt','net_oil_exports_value','net_oil_exports_valuePOP',
                  'gas_exports','net_gas_exports_bcf','net_gas_exports_mboe','net_gas_exports_value','net_gas_exports_valuePOP',
                  'net_oil_gas_exports_valuePOP','population','pop_maddison','mult_nom_2000','mult_nom_2014','mult_2000_2014']]);
plt.legend(list(['oil_prod32_14','oil_price_2000','oil_price_nom','oil_value_nom','oil_value_2000','oil_value_2014','gas_prod55_14',
                'gas_price_2000_mboe','gas_price_2000','gas_price_nom','gas_value_nom','gas_value_2000','gas_value_2014','oil_gas_value_nom',
                'oil_gas_value_2000','oil_gas_value_2014','oil_gas_valuePOP_nom','oil_gas_valuePOP_2000','oil_gas_valuePOP_2014','oil_exports',
                'net_oil_exports','net_oil_exports_mt','net_oil_exports_value','net_oil_exports_valuePOP','gas_exports','net_gas_exports_bcf',
                'net_gas_exports_mboe','net_gas_exports_value','net_gas_exports_valuePOP','net_oil_gas_exports_valuePOP','population','pop_maddison',
                'mult_nom_2000','mult_nom_2014','mult_2000_2014']))


oil_prod32_14 = np.array(dataset['oil_prod32_14'])
oil_price_2000 = np.array(dataset['oil_price_2000'])
oil_price_nom = np.array(dataset['oil_price_nom'])
oil_value_nom = np.array(dataset['oil_value_nom'])
oil_value_2000 = np.array(dataset['oil_value_2000'])
oil_value_2014 = np.array(dataset['oil_value_2014'])
gas_prod55_14 = np.array(dataset['gas_prod55_14'])
gas_price_2000_mboe = np.array(dataset['gas_price_2000_mboe'])
gas_price_2000 = np.array(dataset['gas_price_2000'])
gas_price_nom = np.array(dataset['gas_price_nom'])
gas_value_nom = np.array(dataset['gas_value_nom'])
gas_value_2000 = np.array(dataset['gas_value_2000'])
gas_value_2014 = np.array(dataset['gas_value_2014'])
oil_gas_value_nom = np.array(dataset['oil_gas_value_nom'])
oil_gas_value_2000 = np.array(dataset['oil_gas_value_2000'])
oil_gas_value_2014 = np.array(dataset['oil_gas_value_2014'])
oil_gas_valuePOP_nom = np.array(dataset['oil_gas_valuePOP_nom'])
oil_gas_valuePOP_2000 = np.array(dataset['oil_gas_valuePOP_2000'])
oil_gas_valuePOP_2014 = np.array(dataset['oil_gas_valuePOP_2014'])
oil_exports = np.array(dataset['oil_exports'])
net_oil_exports = np.array(dataset['net_oil_exports'])
net_oil_exports_mt = np.array(dataset['net_oil_exports_mt'])
net_oil_exports_value = np.array(dataset['net_oil_exports_value'])
net_oil_exports_valuePOP = np.array(dataset['net_oil_exports_valuePOP'])
gas_exports = np.array(dataset['gas_exports'])
net_gas_exports_bcf = np.array(dataset['net_gas_exports_bcf'])
net_gas_exports_mboe = np.array(dataset['net_gas_exports_mboe'])
net_gas_exports_value = np.array(dataset['net_gas_exports_value'])
net_gas_exports_valuePOP = np.array(dataset['net_gas_exports_valuePOP'])
net_oil_gas_exports_valuePOP = np.array(dataset['net_oil_gas_exports_valuePOP'])
population = np.array(dataset['population'])
pop_maddison = np.array(dataset['pop_maddison'])
mult_nom_2000 = np.array(dataset['mult_nom_2000'])
mult_nom_2014 = np.array(dataset['mult_nom_2014'])
mult_2000_2014 = np.array(dataset['mult_2000_2014'])

from scipy import stats
stats.probplot(oil_prod32_14, dist='norm',plot=plt)
stats.probplot(oil_price_2000, dist='norm',plot=plt)
stats.probplot(oil_price_nom, dist='norm',plot=plt)
stats.probplot(oil_value_nom, dist='norm',plot=plt)
stats.probplot(oil_value_2000, dist='norm',plot=plt)
stats.probplot(oil_value_2014, dist='norm',plot=plt)
stats.probplot(gas_prod55_14, dist='norm',plot=plt)
stats.probplot(gas_price_2000_mboe, dist='norm',plot=plt)
stats.probplot(gas_price_2000, dist='norm',plot=plt)
stats.probplot(gas_price_nom, dist='norm',plot=plt)
stats.probplot(gas_value_nom, dist='norm',plot=plt)
stats.probplot(gas_value_2000, dist='norm',plot=plt)
stats.probplot(gas_value_2014, dist='norm',plot=plt)
stats.probplot(oil_gas_value_nom, dist='norm',plot=plt)
stats.probplot(oil_gas_value_2000, dist='norm',plot=plt)
stats.probplot(oil_gas_value_2014, dist='norm',plot=plt)
stats.probplot(oil_gas_valuePOP_nom, dist='norm',plot=plt)
stats.probplot(oil_gas_valuePOP_2000, dist='norm',plot=plt)
stats.probplot(oil_gas_valuePOP_2014, dist='norm',plot=plt)
stats.probplot(oil_exports, dist='norm',plot=plt)
stats.probplot(net_oil_exports, dist='norm',plot=plt)
stats.probplot(net_oil_exports_mt, dist='norm',plot=plt)
stats.probplot(net_oil_exports_value, dist='norm',plot=plt)
stats.probplot(net_oil_exports_valuePOP, dist='norm',plot=plt)
stats.probplot(gas_exports, dist='norm',plot=plt)
stats.probplot(net_gas_exports_bcf, dist='norm',plot=plt)
stats.probplot(net_gas_exports_mboe, dist='norm',plot=plt)
stats.probplot(net_gas_exports_value, dist='norm',plot=plt)
stats.probplot(net_gas_exports_valuePOP, dist='norm',plot=plt)
stats.probplot(net_oil_gas_exports_valuePOP, dist='norm',plot=plt)
stats.probplot(population, dist='norm',plot=plt)
stats.probplot(pop_maddison, dist='norm',plot=plt)
stats.probplot(mult_nom_2000, dist='norm',plot=plt)
stats.probplot(mult_nom_2014, dist='norm',plot=plt)
stats.probplot(mult_2000_2014, dist='norm',plot=plt)


# Normal Probability Distribution

x_oil_prod32_14 = np.linspace(np.min(oil_prod32_14), np.max(oil_prod32_14))
y_oil_prod32_14 = stats.norm.pdf(x_oil_prod32_14, np.median(x_oil_prod32_14), np.std(x_oil_prod32_14))
plt.plot(x_oil_prod32_14, y_oil_prod32_14);plt.xlim(np.min(oil_prod32_14), np.max(oil_prod32_14));plt.title('Normal Probability Distribution of Total Oil Production from 1932-2014');plt.xlabel('Total Oil Production (bbl/year)');plt.ylabel('Probability')

x_oil_price_2000 = np.linspace(np.min(oil_price_2000), np.max(oil_price_2000))
y_oil_price_2000 = stats.norm.pdf(x_oil_price_2000, np.median(x_oil_price_2000), np.std(x_oil_price_2000))
plt.plot(x_oil_price_2000, y_oil_price_2000);plt.xlim(np.min(oil_price_2000), np.max(oil_price_2000));plt.title('Normal Probability Distribution of Price of Total Oil Production, in 2000 US $');plt.xlabel('Price of Total Oil Production ($ per barrel)');plt.ylabel(' Probability')

x_oil_price_nom = np.linspace(np.min(oil_price_nom), np.max(oil_price_nom))
y_oil_price_nom = stats.norm.pdf(x_oil_price_nom, np.median(x_oil_price_nom), np.std(x_oil_price_nom))
plt.plot(x_oil_price_nom, y_oil_price_nom);plt.xlim(np.min(oil_price_nom), np.max(oil_price_nom));plt.title('Normal Probability Distribution of Nominal Price of Total Oil Production, in US $');plt.xlabel('Nominal Price of Total Oil Production');plt.ylabel(' Probability')

x_oil_value_nom = np.linspace(np.min(oil_value_nom), np.max(oil_value_nom))
y_oil_value_nom = stats.norm.pdf(x_oil_value_nom, np.median(x_oil_value_nom), np.std(x_oil_value_nom))
plt.plot(x_oil_value_nom, y_oil_value_nom);plt.xlim(np.min(oil_value_nom), np.max(oil_value_nom));plt.title('Normal Probability Distribution of Nominal Value of Total Oil Production, in US $');plt.xlabel('Nominal Value of Total Oil Production');plt.ylabel(' Probability')

x_oil_value_2000 = np.linspace(np.min(oil_value_2000), np.max(oil_value_2000))
y_oil_value_2000 = stats.norm.pdf(x_oil_value_2000, np.median(x_oil_value_2000), np.std(x_oil_value_2000))
plt.plot(x_oil_value_2000, y_oil_value_2000);plt.xlim(np.min(oil_value_2000), np.max(oil_value_2000));plt.title('Normal Probability Distribution of Value of Total Oil Production, in 2000 US $');plt.xlabel('Value of Total Oil Production, 2000');plt.ylabel(' Probability')

x_oil_value_2014 = np.linspace(np.min(oil_value_2014), np.max(oil_value_2014))
y_oil_value_2014 = stats.norm.pdf(x_oil_value_2014, np.median(x_oil_value_2014), np.std(x_oil_value_2014))
plt.plot(x_oil_value_2014, y_oil_value_2014);plt.xlim(np.min(oil_value_2014), np.max(oil_value_2014));plt.title('Normal Probability Distribution of Value of Total Oil Production, in 2014 US $');plt.xlabel('Value of Total Oil Production, 2014');plt.ylabel(' Probability')

x_gas_prod55_14 = np.linspace(np.min(gas_prod55_14), np.max(gas_prod55_14))
y_gas_prod55_14 = stats.norm.pdf(x_gas_prod55_14, np.median(x_gas_prod55_14), np.std(x_gas_prod55_14))
plt.plot(x_gas_prod55_14, y_gas_prod55_14);plt.xlim(np.min(gas_prod55_14), np.max(gas_prod55_14));plt.title('Normal Probability Distribution of Total Gas Production from 1955-2014 (Billion Cubic Meters)');plt.xlabel('Total Gas Production (BCM)');plt.ylabel(' Probability')

x_gas_price_2000_mboe = np.linspace(np.min(gas_price_2000_mboe), np.max(gas_price_2000_mboe))
y_gas_price_2000_mboe = stats.norm.pdf(x_gas_price_2000_mboe, np.median(x_gas_price_2000_mboe), np.std(x_gas_price_2000_mboe))
plt.plot(x_gas_price_2000_mboe, y_gas_price_2000_mboe);plt.xlim(np.min(gas_price_2000_mboe), np.max(gas_price_2000_mboe));plt.title('Normal Probability Distribution of Price of Total Gas Production, in 2000 US $');plt.xlabel('Price of Total Gas Production ($ per mboe)');plt.ylabel(' Probability')

x_gas_price_2000 = np.linspace(np.min(gas_price_2000), np.max(gas_price_2000))
y_gas_price_2000 = stats.norm.pdf(x_gas_price_2000, np.median(x_gas_price_2000), np.std(x_gas_price_2000))
plt.plot(x_gas_price_2000, y_gas_price_2000);plt.xlim(np.min(gas_price_2000), np.max(gas_price_2000));plt.title('Normal Probability Distribution of Price of Total Gas Production, in 2000 US $');plt.xlabel('Price of Total Gas Production ($ per gallon)');plt.ylabel(' Probability')

x_gas_price_nom = np.linspace(np.min(gas_price_nom), np.max(gas_price_nom))
y_gas_price_nom = stats.norm.pdf(x_gas_price_nom, np.median(x_gas_price_nom), np.std(x_gas_price_nom))
plt.plot(x_gas_price_nom, y_gas_price_nom);plt.xlim(np.min(gas_price_nom), np.max(gas_price_nom));plt.title('Normal Probability Distribution of Nominal Price of Total Gas Production, in US $');plt.xlabel('Nominal Price of Total Gas Production');plt.ylabel(' Probability')

x_gas_value_nom = np.linspace(np.min(gas_value_nom), np.max(gas_value_nom))
y_gas_value_nom = stats.norm.pdf(x_gas_value_nom, np.median(x_gas_value_nom), np.std(x_gas_value_nom))
plt.plot(x_gas_value_nom, y_gas_value_nom);plt.xlim(np.min(gas_value_nom), np.max(gas_value_nom));plt.title('Normal Probability Distribution of Nominal Value of Total Gas Production, in US $');plt.xlabel('Nominal Value of Total Gas Production');plt.ylabel(' Probability')

x_gas_value_2000 = np.linspace(np.min(gas_value_2000), np.max(gas_value_2000))
y_gas_value_2000 = stats.norm.pdf(x_gas_value_2000, np.median(x_gas_value_2000), np.std(x_gas_value_2000))
plt.plot(x_gas_value_2000, y_gas_value_2000);plt.xlim(np.min(gas_value_2000), np.max(gas_value_2000));plt.title('Normal Probability Distribution of Value of Total Gas Production, in 2000 US $');plt.xlabel('Value of Total Gas Production');plt.ylabel(' Probability')

x_gas_value_2014 = np.linspace(np.min(gas_value_2014), np.max(gas_value_2014))
y_gas_value_2014 = stats.norm.pdf(x_gas_value_2014, np.median(x_gas_value_2014), np.std(x_gas_value_2014))
plt.plot(x_gas_value_2014, y_gas_value_2014);plt.xlim(np.min(gas_value_2014), np.max(gas_value_2014));plt.title('Normal Probability Distribution of Value of Total Gas Production, in 2014 US $');plt.xlabel('Value of Total Gas Production');plt.ylabel(' Probability')

x_oil_gas_value_nom = np.linspace(np.min(oil_gas_value_nom), np.max(oil_gas_value_nom))
y_oil_gas_value_nom = stats.norm.pdf(x_oil_gas_value_nom, np.median(x_oil_gas_value_nom), np.std(x_oil_gas_value_nom))
plt.plot(x_oil_gas_value_nom, y_oil_gas_value_nom);plt.xlim(np.min(oil_gas_value_nom), np.max(oil_gas_value_nom));plt.title('Normal Probability Distribution of Nominal Value of Total Oil and Gas Production, in US $');plt.xlabel('Nominal Value of Total Oil and Gas Production');plt.ylabel(' Probability')

x_oil_gas_value_2000 = np.linspace(np.min(oil_gas_value_2000), np.max(oil_gas_value_2000))
y_oil_gas_value_2000 = stats.norm.pdf(x_oil_gas_value_2000, np.median(x_oil_gas_value_2000), np.std(x_oil_gas_value_2000))
plt.plot(x_oil_gas_value_2000, y_oil_gas_value_2000);plt.xlim(np.min(oil_gas_value_2000), np.max(oil_gas_value_2000));plt.title('Normal Probability Distribution of Value of Total Oil and Gas Production, in 2000 US $');plt.xlabel('Value of Total Oil and Gas Production ($ per mboe)');plt.ylabel(' Probability')

x_oil_gas_value_2014 = np.linspace(np.min(oil_gas_value_2014), np.max(oil_gas_value_2014))
y_oil_gas_value_2014 = stats.norm.pdf(x_oil_gas_value_2014, np.median(x_oil_gas_value_2014), np.std(x_oil_gas_value_2014))
plt.plot(x_oil_gas_value_2014, y_oil_gas_value_2014);plt.xlim(np.min(oil_gas_value_2014), np.max(oil_gas_value_2014));plt.title('Normal Probability Distribution of Value of Total Oil and Gas Production, in 2014 US $');plt.xlabel('Value of Total Oil and Gas Production ($ per mboe)');plt.ylabel(' Probability')

x_oil_gas_valuePOP_nom = np.linspace(np.min(oil_gas_valuePOP_nom), np.max(oil_gas_valuePOP_nom))
y_oil_gas_valuePOP_nom = stats.norm.pdf(x_oil_gas_valuePOP_nom, np.median(x_oil_gas_valuePOP_nom), np.std(x_oil_gas_valuePOP_nom))
plt.plot(x_oil_gas_valuePOP_nom, y_oil_gas_valuePOP_nom);plt.xlim(np.min(oil_gas_valuePOP_nom), np.max(oil_gas_valuePOP_nom));plt.title('Normal Probability Distribution of Nominal Value of Total Oil and Gas POP, in US $');plt.xlabel('Nominal Value of Total Oil and Gas POP');plt.ylabel(' Probability')

x_oil_gas_valuePOP_2000 = np.linspace(np.min(oil_gas_valuePOP_2000), np.max(oil_gas_valuePOP_2000))
y_oil_gas_valuePOP_2000 = stats.norm.pdf(x_oil_gas_valuePOP_2000, np.median(x_oil_gas_valuePOP_2000), np.std(x_oil_gas_valuePOP_2000))
plt.plot(x_oil_gas_valuePOP_2000, y_oil_gas_valuePOP_2000);plt.xlim(np.min(oil_gas_valuePOP_2000), np.max(oil_gas_valuePOP_2000));plt.title('Normal Probability Distribution of Value of Total Oil and Gas POP, in 2000 US $');plt.xlabel('Value of Total Oil and Gas POP ($ per mboe)');plt.ylabel(' Probability')

x_oil_gas_valuePOP_2014 = np.linspace(np.min(oil_gas_valuePOP_2014), np.max(oil_gas_valuePOP_2014))
y_oil_gas_valuePOP_2014 = stats.norm.pdf(x_oil_gas_valuePOP_2014, np.median(x_oil_gas_valuePOP_2014), np.std(x_oil_gas_valuePOP_2014))
plt.plot(x_oil_gas_valuePOP_2014, y_oil_gas_valuePOP_2014);plt.xlim(np.min(oil_gas_valuePOP_2014), np.max(oil_gas_valuePOP_2014));plt.title('Normal Probability Distribution of Value of Total Oil and Gas POP, in 2014 US $');plt.xlabel('Value of Total Oil and Gas POP ($ per mboe)');plt.ylabel(' Probability')

x_oil_exports = np.linspace(np.min(oil_exports), np.max(oil_exports))
y_oil_exports = stats.norm.pdf(x_oil_exports, np.median(x_oil_exports), np.std(x_oil_exports))
plt.plot(x_oil_exports, y_oil_exports);plt.xlim(np.min(oil_exports), np.max(oil_exports));plt.title('Normal Probability Distribution of Total Amount of Oil Exported');plt.xlabel('Amount of Oil Exported (bbl/year)');plt.ylabel(' Probability')

x_net_oil_exports = np.linspace(np.min(net_oil_exports), np.max(net_oil_exports))
y_net_oil_exports = stats.norm.pdf(x_net_oil_exports, np.median(x_net_oil_exports), np.std(x_net_oil_exports))
plt.plot(x_net_oil_exports, y_net_oil_exports);plt.xlim(np.min(net_oil_exports), np.max(net_oil_exports));plt.title('Normal Probability Distribution of Net Amount of Oil Exported');plt.xlabel('Net Amount of Oil Exported');plt.ylabel(' Probability')

x_net_oil_exports_mt = np.linspace(np.min(net_oil_exports_mt), np.max(net_oil_exports_mt))
y_net_oil_exports_mt = stats.norm.pdf(x_net_oil_exports_mt, np.median(x_net_oil_exports_mt), np.std(x_net_oil_exports_mt))
plt.plot(x_net_oil_exports_mt, y_net_oil_exports_mt);plt.xlim(np.min(net_oil_exports_mt), np.max(net_oil_exports_mt));plt.title('Normal Probability Distribution of Net Amount of Oil Exported, in mt');plt.xlabel('Net Amount of Oil Exported (million tonnes)');plt.ylabel(' Probability')

x_net_oil_exports_value = np.linspace(np.min(net_oil_exports_value), np.max(net_oil_exports_value))
y_net_oil_exports_value = stats.norm.pdf(x_net_oil_exports_value, np.median(x_net_oil_exports_value), np.std(x_net_oil_exports_value))
plt.plot(x_net_oil_exports_value, y_net_oil_exports_value);plt.xlim(np.min(net_oil_exports_value), np.max(net_oil_exports_value));plt.title('Normal Probability Distribution of Value of Net Oil Export, in US $');plt.xlabel('Value of Net Oil Export');plt.ylabel(' Probability')

x_net_oil_exports_valuePOP = np.linspace(np.min(net_oil_exports_valuePOP), np.max(net_oil_exports_valuePOP))
y_net_oil_exports_valuePOP = stats.norm.pdf(x_net_oil_exports_valuePOP, np.median(x_net_oil_exports_valuePOP), np.std(x_net_oil_exports_valuePOP))
plt.plot(x_net_oil_exports_valuePOP, y_net_oil_exports_valuePOP);plt.xlim(np.min(net_oil_exports_valuePOP), np.max(net_oil_exports_valuePOP));plt.title('Normal Probability Distribution of Net Oil Export POP, in US $');plt.xlabel('Net Oil Export POP');plt.ylabel(' Probability')

x_gas_exports = np.linspace(np.min(gas_exports), np.max(gas_exports))
y_gas_exports = stats.norm.pdf(x_gas_exports, np.median(x_gas_exports), np.std(x_gas_exports))
plt.plot(x_gas_exports, y_gas_exports);plt.xlim(np.min(gas_exports), np.max(gas_exports));plt.title('Normal Probability Distribution of Total Amount of Gas Exported, in Cubic meters');plt.xlabel('Amount of Gas Exported (cubic meters)');plt.ylabel(' Probability')

x_net_gas_exports_bcf = np.linspace(np.min(net_gas_exports_bcf), np.max(net_gas_exports_bcf))
y_net_gas_exports_bcf = stats.norm.pdf(x_net_gas_exports_bcf, np.median(x_net_gas_exports_bcf), np.std(x_net_gas_exports_bcf))
plt.plot(x_net_gas_exports_bcf, y_net_gas_exports_bcf);plt.xlim(np.min(net_gas_exports_bcf), np.max(net_gas_exports_bcf));plt.title('Normal Probability Distribution of Net Amount of Gas Exports, in Billion cubic feet');plt.xlabel('Amount of Gas Export (bcf)');plt.ylabel(' Probability')

x_net_gas_exports_mboe = np.linspace(np.min(net_gas_exports_mboe), np.max(net_gas_exports_mboe))
y_net_gas_exports_mboe = stats.norm.pdf(x_net_gas_exports_mboe, np.median(x_net_gas_exports_mboe), np.std(x_net_gas_exports_mboe))
plt.plot(x_net_gas_exports_mboe, y_net_gas_exports_mboe);plt.xlim(np.min(net_gas_exports_mboe), np.max(net_gas_exports_mboe));plt.title('Normal Probability Distribution of Net Amount of Gas Exports, in mboe');plt.xlabel('Amount of Gas Export (mboe)');plt.ylabel(' Probability')

x_net_gas_exports_value = np.linspace(np.min(net_gas_exports_value), np.max(net_gas_exports_value))
y_net_gas_exports_value = stats.norm.pdf(x_net_gas_exports_value, np.median(x_net_gas_exports_value), np.std(x_net_gas_exports_value))
plt.plot(x_net_gas_exports_value, y_net_gas_exports_value);plt.xlim(np.min(net_gas_exports_value), np.max(net_gas_exports_value));plt.title('Normal Probability Distribution of Value of Net Gas Exports, in US $');plt.xlabel('Value of Net Gas Export');plt.ylabel(' Probability')

x_net_gas_exports_valuePOP = np.linspace(np.min(net_gas_exports_valuePOP), np.max(net_gas_exports_valuePOP))
y_net_gas_exports_valuePOP = stats.norm.pdf(x_net_gas_exports_valuePOP, np.median(x_net_gas_exports_valuePOP), np.std(x_net_gas_exports_valuePOP))
plt.plot(x_net_gas_exports_valuePOP, y_net_gas_exports_valuePOP);plt.xlim(np.min(net_gas_exports_valuePOP), np.max(net_gas_exports_valuePOP));plt.title('Normal Probability Distribution of Value of Net Gas Export POP, in US $');plt.xlabel('Value of Net Gas Export POP');plt.ylabel(' Probability')

x_net_oil_gas_exports_valuePOP = np.linspace(np.min(net_oil_gas_exports_valuePOP), np.max(net_oil_gas_exports_valuePOP))
y_net_oil_gas_exports_valuePOP = stats.norm.pdf(x_net_oil_gas_exports_valuePOP, np.median(x_net_oil_gas_exports_valuePOP), np.std(x_net_oil_gas_exports_valuePOP))
plt.plot(x_net_oil_gas_exports_valuePOP, y_net_oil_gas_exports_valuePOP);plt.xlim(np.min(net_oil_gas_exports_valuePOP), np.max(net_oil_gas_exports_valuePOP));plt.title('Normal Probability Distribution of Value of Net Oil and Gas Export POP, in US $');plt.xlabel('Value of Net Oil and Gas Export POP');plt.ylabel(' Probability')

x_population = np.linspace(np.min(population), np.max(population))
y_population = stats.norm.pdf(x_population, np.median(x_population), np.std(x_population))
plt.plot(x_population, y_population);plt.xlim(np.min(population), np.max(population));plt.title('Normal Probability Distribution of Population');plt.xlabel('Population');plt.ylabel(' Probability')

x_pop_madd = np.linspace(np.min(pop_maddison), np.max(pop_maddison))
y_pop_madd = stats.norm.pdf(x_pop_madd, np.median(x_pop_madd), np.std(x_pop_madd))
plt.plot(x_pop_madd, y_pop_madd);plt.xlim(np.min(pop_maddison), np.max(pop_maddison));plt.title('Normal Probability Distribution of Maddison Estimated Population');plt.xlabel('Maddison Population');plt.ylabel(' Probability')

# Boxplot

sns.boxplot(dataset['oil_prod32_14'],orient='v').set_title('Boxplot of Total Oil Production from 1932-2014')
sns.boxplot(dataset['oil_price_2000'],orient='v').set_title('Boxplot of Price of Total Oil Production, in 2000 US $')
sns.boxplot(dataset['oil_price_nom'],orient='v').set_title('Boxplot of Nominal Price of Total Oil Production, in US $')
sns.boxplot(dataset['oil_value_nom'],orient='v').set_title('Boxplot of Nominal Value of Total Oil Production, in US $')
sns.boxplot(dataset['oil_value_2000'],orient='v').set_title('Boxplot of Value of Total Oil Production, in 2000 US $')
sns.boxplot(dataset['oil_value_2014'],orient='v').set_title('Boxplot of Value of Total Oil Production, in 2014 US $')
sns.boxplot(dataset['gas_prod55_14'],orient='v').set_title('Boxplot of Total Gas Production from 1955-2014 (Billion Cubic Meters)')
sns.boxplot(dataset['gas_price_2000_mboe'],orient='v').set_title('Boxplot of Price of Total Gas Production, in 2000 US $')
sns.boxplot(dataset['gas_price_2000'],orient='v').set_title('Boxplot of Price of Total Gas Production, in 2000 US $')
sns.boxplot(dataset['gas_price_nom'],orient='v').set_title('Boxplot of Nominal Price of Total Gas Production, in US $')
sns.boxplot(dataset['gas_value_nom'],orient='v').set_title('Boxplot of Nominal Value of Total Gas Production, in US $')
sns.boxplot(dataset['gas_value_2000'],orient='v').set_title('Boxplot of Value of Total Gas Production, in 2000 US $')
sns.boxplot(dataset['gas_value_2014'],orient='v').set_title('Boxplot of Value of Total Gas Production, in 2014 US $')
sns.boxplot(dataset['oil_gas_value_nom'],orient='v').set_title('Boxplot of Nominal Value of Total Oil and Gas Production, in US $')
sns.boxplot(dataset['oil_gas_value_2000'],orient='v').set_title('Boxplot of Value of Total Oil and Gas Production, in 2000 US $')
sns.boxplot(dataset['oil_gas_value_2014'],orient='v').set_title('Boxplot of Value of Total Oil and Gas Production, in 2014 US $')
sns.boxplot(dataset['oil_gas_valuePOP_nom'],orient='v').set_title('Boxplot of Nominal Value of Total Oil and Gas POP, in US $')
sns.boxplot(dataset['oil_gas_valuePOP_2000'],orient='v').set_title('Boxplot of Value of Total Oil and Gas POP, in 2000 US $')
sns.boxplot(dataset['oil_gas_valuePOP_2014'],orient='v').set_title('Boxplot of Value of Total Oil and Gas POP, in 2014 US $')
sns.boxplot(dataset['oil_exports'],orient='v').set_title('Boxplot of Total Amount of Oil Exported')
sns.boxplot(dataset['net_oil_exports'],orient='v').set_title('Boxplot of Net Amount of Oil Exported')
sns.boxplot(dataset['net_oil_exports_mt'],orient='v').set_title('Boxplot of Net Amount of Oil Exported, in mt')
sns.boxplot(dataset['net_oil_exports_value'],orient='v').set_title('Boxplot of Value of Net Oil Export, in US $')
sns.boxplot(dataset['net_oil_exports_valuePOP'],orient='v').set_title('Boxplot of Net Oil Export POP, in US $')
sns.boxplot(dataset['gas_exports'],orient='v').set_title('Boxplot of Total Amount of Gas Exported, in Cubic meters')
sns.boxplot(dataset['net_gas_exports_bcf'],orient='v').set_title('Boxplot of Net Amount of Gas Exports, in Billion cubic feet')
sns.boxplot(dataset['net_gas_exports_mboe'],orient='v').set_title('Boxplot of Net Amount of Gas Exports, in mboe')
sns.boxplot(dataset['net_gas_exports_value'],orient='v').set_title('Boxplot of Value of Net Gas Exports, in US $')
sns.boxplot(dataset['net_gas_exports_valuePOP'],orient='v').set_title('Boxplot of Value of Net Gas Export POP, in US $')
sns.boxplot(dataset['net_oil_gas_exports_valuePOP'],orient='v').set_title('Boxplot of Value of Net Oil and Gas Export POP, in US $')
sns.boxplot(dataset['population'],orient='v').set_title('Boxplot of Population')
sns.boxplot(dataset['pop_maddison'],orient='v').set_title('Boxplot of Maddison Estimated Population')
sns.boxplot(dataset['mult_nom_2000'])
sns.boxplot(dataset['mult_nom_2014'])

# Scatter Plot

plt.scatter(x='oil_prod32_14',y='oil_price_2000',data=dataset)
plt.scatter(x='oil_prod32_14',y='oil_price_nom',data=dataset)
plt.scatter(x='oil_prod32_14',y='oil_value_nom',data=dataset)
plt.scatter(x='oil_prod32_14',y='oil_value_2000',data=dataset)
plt.scatter(x='oil_prod32_14',y='oil_value_2014',data=dataset)
plt.scatter(x='oil_prod32_14',y='gas_prod55_14',data=dataset)
plt.scatter(x='oil_prod32_14',y='gas_price_2000_mboe',data=dataset)
plt.scatter(x='oil_prod32_14',y='gas_price_2000',data=dataset)
plt.scatter(x='oil_prod32_14',y='gas_price_nom',data=dataset)
plt.scatter(x='oil_prod32_14',y='gas_value_nom',data=dataset)
plt.scatter(x='oil_prod32_14',y='gas_value_2000',data=dataset)
plt.scatter(x='oil_prod32_14',y='gas_value_2014',data=dataset)
plt.scatter(x='oil_prod32_14',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='oil_prod32_14',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='oil_prod32_14',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='oil_prod32_14',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='oil_prod32_14',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='oil_prod32_14',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='oil_prod32_14',y='oil_exports',data=dataset)
plt.scatter(x='oil_prod32_14',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_prod32_14',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_prod32_14',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_prod32_14',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_prod32_14',y='gas_exports',data=dataset)
plt.scatter(x='oil_prod32_14',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_prod32_14',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_prod32_14',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_prod32_14',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_prod32_14',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_prod32_14',y='population',data=dataset)
plt.scatter(x='oil_prod32_14',y='pop_maddison',data=dataset)

plt.scatter(x='oil_price_2000',y='oil_price_nom',data=dataset)
plt.scatter(x='oil_price_2000',y='oil_value_nom',data=dataset)
plt.scatter(x='oil_price_2000',y='oil_value_2000',data=dataset)
plt.scatter(x='oil_price_2000',y='oil_value_2014',data=dataset)
plt.scatter(x='oil_price_2000',y='gas_prod55_14',data=dataset)
plt.scatter(x='oil_price_2000',y='gas_price_2000_mboe',data=dataset)
plt.scatter(x='oil_price_2000',y='gas_price_2000',data=dataset)
plt.scatter(x='oil_price_2000',y='gas_price_nom',data=dataset)
plt.scatter(x='oil_price_2000',y='gas_value_nom',data=dataset)
plt.scatter(x='oil_price_2000',y='gas_value_2000',data=dataset)
plt.scatter(x='oil_price_2000',y='gas_value_2014',data=dataset)
plt.scatter(x='oil_price_2000',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='oil_price_2000',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='oil_price_2000',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='oil_price_2000',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='oil_price_2000',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='oil_price_2000',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='oil_price_2000',y='oil_exports',data=dataset)
plt.scatter(x='oil_price_2000',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_price_2000',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_price_2000',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_price_2000',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_price_2000',y='gas_exports',data=dataset)
plt.scatter(x='oil_price_2000',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_price_2000',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_price_2000',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_price_2000',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_price_2000',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_price_2000',y='population',data=dataset)
plt.scatter(x='oil_price_2000',y='pop_maddison',data=dataset)

plt.scatter(x='oil_price_nom',y='oil_value_nom',data=dataset)
plt.scatter(x='oil_price_nom',y='oil_value_2000',data=dataset)
plt.scatter(x='oil_price_nom',y='oil_value_2014',data=dataset)
plt.scatter(x='oil_price_nom',y='gas_prod55_14',data=dataset)
plt.scatter(x='oil_price_nom',y='gas_price_2000_mboe',data=dataset)
plt.scatter(x='oil_price_nom',y='gas_price_2000',data=dataset)
plt.scatter(x='oil_price_nom',y='gas_price_nom',data=dataset)
plt.scatter(x='oil_price_nom',y='gas_value_nom',data=dataset)
plt.scatter(x='oil_price_nom',y='gas_value_2000',data=dataset)
plt.scatter(x='oil_price_nom',y='gas_value_2014',data=dataset)
plt.scatter(x='oil_price_nom',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='oil_price_nom',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='oil_price_nom',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='oil_price_nom',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='oil_price_nom',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='oil_price_nom',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='oil_price_nom',y='oil_exports',data=dataset)
plt.scatter(x='oil_price_nom',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_price_nom',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_price_nom',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_price_nom',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_price_nom',y='gas_exports',data=dataset)
plt.scatter(x='oil_price_nom',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_price_nom',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_price_nom',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_price_nom',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_price_nom',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_price_nom',y='population',data=dataset)
plt.scatter(x='oil_price_nom',y='pop_maddison',data=dataset)

plt.scatter(x='oil_value_nom',y='oil_value_2000',data=dataset)
plt.scatter(x='oil_value_nom',y='oil_value_2014',data=dataset)
plt.scatter(x='oil_value_nom',y='gas_prod55_14',data=dataset)
plt.scatter(x='oil_value_nom',y='gas_price_2000_mboe',data=dataset)
plt.scatter(x='oil_value_nom',y='gas_price_2000',data=dataset)
plt.scatter(x='oil_value_nom',y='gas_price_nom',data=dataset)
plt.scatter(x='oil_value_nom',y='gas_value_nom',data=dataset)
plt.scatter(x='oil_value_nom',y='gas_value_2000',data=dataset)
plt.scatter(x='oil_value_nom',y='gas_value_2014',data=dataset)
plt.scatter(x='oil_value_nom',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='oil_value_nom',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='oil_value_nom',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='oil_value_nom',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='oil_value_nom',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='oil_value_nom',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='oil_value_nom',y='oil_exports',data=dataset)
plt.scatter(x='oil_value_nom',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_value_nom',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_value_nom',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_value_nom',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_value_nom',y='gas_exports',data=dataset)
plt.scatter(x='oil_value_nom',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_value_nom',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_value_nom',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_value_nom',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_value_nom',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_value_nom',y='population',data=dataset)
plt.scatter(x='oil_value_nom',y='pop_maddison',data=dataset)

plt.scatter(x='oil_value_2000',y='oil_value_2014',data=dataset)
plt.scatter(x='oil_value_2000',y='gas_prod55_14',data=dataset)
plt.scatter(x='oil_value_2000',y='gas_price_2000_mboe',data=dataset)
plt.scatter(x='oil_value_2000',y='gas_price_2000',data=dataset)
plt.scatter(x='oil_value_2000',y='gas_price_nom',data=dataset)
plt.scatter(x='oil_value_2000',y='gas_value_nom',data=dataset)
plt.scatter(x='oil_value_2000',y='gas_value_2000',data=dataset)
plt.scatter(x='oil_value_2000',y='gas_value_2014',data=dataset)
plt.scatter(x='oil_value_2000',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='oil_value_2000',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='oil_value_2000',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='oil_value_2000',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='oil_value_2000',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='oil_value_2000',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='oil_value_2000',y='oil_exports',data=dataset)
plt.scatter(x='oil_value_2000',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_value_2000',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_value_2000',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_value_2000',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_value_2000',y='gas_exports',data=dataset)
plt.scatter(x='oil_value_2000',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_value_2000',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_value_2000',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_value_2000',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_value_2000',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_value_2000',y='population',data=dataset)
plt.scatter(x='oil_value_2000',y='pop_maddison',data=dataset)

plt.scatter(x='oil_value_2014',y='gas_prod55_14',data=dataset)
plt.scatter(x='oil_value_2014',y='gas_price_2000_mboe',data=dataset)
plt.scatter(x='oil_value_2014',y='gas_price_2000',data=dataset)
plt.scatter(x='oil_value_2014',y='gas_price_nom',data=dataset)
plt.scatter(x='oil_value_2014',y='gas_value_nom',data=dataset)
plt.scatter(x='oil_value_2014',y='gas_value_2000',data=dataset)
plt.scatter(x='oil_value_2014',y='gas_value_2014',data=dataset)
plt.scatter(x='oil_value_2014',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='oil_value_2014',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='oil_value_2014',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='oil_value_2014',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='oil_value_2014',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='oil_value_2014',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='oil_value_2014',y='oil_exports',data=dataset)
plt.scatter(x='oil_value_2014',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_value_2014',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_value_2014',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_value_2014',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_value_2014',y='gas_exports',data=dataset)
plt.scatter(x='oil_value_2014',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_value_2014',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_value_2014',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_value_2014',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_value_2014',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_value_2014',y='population',data=dataset)
plt.scatter(x='oil_value_2014',y='pop_maddison',data=dataset)

plt.scatter(x='gas_prod55_14',y='gas_price_2000_mboe',data=dataset)
plt.scatter(x='gas_prod55_14',y='gas_price_2000',data=dataset)
plt.scatter(x='gas_prod55_14',y='gas_price_nom',data=dataset)
plt.scatter(x='gas_prod55_14',y='gas_value_nom',data=dataset)
plt.scatter(x='gas_prod55_14',y='gas_value_2000',data=dataset)
plt.scatter(x='gas_prod55_14',y='gas_value_2014',data=dataset)
plt.scatter(x='gas_prod55_14',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='gas_prod55_14',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='gas_prod55_14',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='gas_prod55_14',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='gas_prod55_14',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='gas_prod55_14',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='gas_prod55_14',y='oil_exports',data=dataset)
plt.scatter(x='gas_prod55_14',y='net_oil_exports',data=dataset)
plt.scatter(x='gas_prod55_14',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='gas_prod55_14',y='net_oil_exports_value',data=dataset)
plt.scatter(x='gas_prod55_14',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='gas_prod55_14',y='gas_exports',data=dataset)
plt.scatter(x='gas_prod55_14',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='gas_prod55_14',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='gas_prod55_14',y='net_gas_exports_value',data=dataset)
plt.scatter(x='gas_prod55_14',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_prod55_14',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_prod55_14',y='population',data=dataset)
plt.scatter(x='gas_prod55_14',y='pop_maddison',data=dataset)

plt.scatter(x='gas_price_2000_mboe',y='gas_price_2000',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='gas_price_nom',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='gas_value_nom',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='gas_value_2000',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='gas_value_2014',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='oil_exports',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='net_oil_exports',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='net_oil_exports_value',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='gas_exports',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='net_gas_exports_value',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='population',data=dataset)
plt.scatter(x='gas_price_2000_mboe',y='pop_maddison',data=dataset)

plt.scatter(x='gas_price_2000',y='gas_price_nom',data=dataset)
plt.scatter(x='gas_price_2000',y='gas_value_nom',data=dataset)
plt.scatter(x='gas_price_2000',y='gas_value_2000',data=dataset)
plt.scatter(x='gas_price_2000',y='gas_value_2014',data=dataset)
plt.scatter(x='gas_price_2000',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='gas_price_2000',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='gas_price_2000',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='gas_price_2000',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='gas_price_2000',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='gas_price_2000',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='gas_price_2000',y='oil_exports',data=dataset)
plt.scatter(x='gas_price_2000',y='net_oil_exports',data=dataset)
plt.scatter(x='gas_price_2000',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='gas_price_2000',y='net_oil_exports_value',data=dataset)
plt.scatter(x='gas_price_2000',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='gas_price_2000',y='gas_exports',data=dataset)
plt.scatter(x='gas_price_2000',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='gas_price_2000',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='gas_price_2000',y='net_gas_exports_value',data=dataset)
plt.scatter(x='gas_price_2000',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_price_2000',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_price_2000',y='population',data=dataset)
plt.scatter(x='gas_price_2000',y='pop_maddison',data=dataset)

plt.scatter(x='gas_price_nom',y='gas_value_nom',data=dataset)
plt.scatter(x='gas_price_nom',y='gas_value_2000',data=dataset)
plt.scatter(x='gas_price_nom',y='gas_value_2014',data=dataset)
plt.scatter(x='gas_price_nom',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='gas_price_nom',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='gas_price_nom',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='gas_price_nom',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='gas_price_nom',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='gas_price_nom',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='gas_price_nom',y='oil_exports',data=dataset)
plt.scatter(x='gas_price_nom',y='net_oil_exports',data=dataset)
plt.scatter(x='gas_price_nom',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='gas_price_nom',y='net_oil_exports_value',data=dataset)
plt.scatter(x='gas_price_nom',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='gas_price_nom',y='gas_exports',data=dataset)
plt.scatter(x='gas_price_nom',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='gas_price_nom',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='gas_price_nom',y='net_gas_exports_value',data=dataset)
plt.scatter(x='gas_price_nom',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_price_nom',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_price_nom',y='population',data=dataset)
plt.scatter(x='gas_price_nom',y='pop_maddison',data=dataset)

plt.scatter(x='gas_value_nom',y='gas_value_2000',data=dataset)
plt.scatter(x='gas_value_nom',y='gas_value_2014',data=dataset)
plt.scatter(x='gas_value_nom',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='gas_value_nom',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='gas_value_nom',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='gas_value_nom',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='gas_value_nom',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='gas_value_nom',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='gas_value_nom',y='oil_exports',data=dataset)
plt.scatter(x='gas_value_nom',y='net_oil_exports',data=dataset)
plt.scatter(x='gas_value_nom',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='gas_value_nom',y='net_oil_exports_value',data=dataset)
plt.scatter(x='gas_value_nom',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='gas_value_nom',y='gas_exports',data=dataset)
plt.scatter(x='gas_value_nom',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='gas_value_nom',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='gas_value_nom',y='net_gas_exports_value',data=dataset)
plt.scatter(x='gas_value_nom',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_value_nom',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_value_nom',y='population',data=dataset)
plt.scatter(x='gas_value_nom',y='pop_maddison',data=dataset)

plt.scatter(x='gas_value_2000',y='gas_value_2014',data=dataset)
plt.scatter(x='gas_value_2000',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='gas_value_2000',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='gas_value_2000',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='gas_value_2000',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='gas_value_2000',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='gas_value_2000',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='gas_value_2000',y='oil_exports',data=dataset)
plt.scatter(x='gas_value_2000',y='net_oil_exports',data=dataset)
plt.scatter(x='gas_value_2000',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='gas_value_2000',y='net_oil_exports_value',data=dataset)
plt.scatter(x='gas_value_2000',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='gas_value_2000',y='gas_exports',data=dataset)
plt.scatter(x='gas_value_2000',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='gas_value_2000',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='gas_value_2000',y='net_gas_exports_value',data=dataset)
plt.scatter(x='gas_value_2000',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_value_2000',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_value_2000',y='population',data=dataset)
plt.scatter(x='gas_value_2000',y='pop_maddison',data=dataset)

plt.scatter(x='gas_value_2014',y='oil_gas_value_nom',data=dataset)
plt.scatter(x='gas_value_2014',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='gas_value_2014',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='gas_value_2014',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='gas_value_2014',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='gas_value_2014',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='gas_value_2014',y='oil_exports',data=dataset)
plt.scatter(x='gas_value_2014',y='net_oil_exports',data=dataset)
plt.scatter(x='gas_value_2014',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='gas_value_2014',y='net_oil_exports_value',data=dataset)
plt.scatter(x='gas_value_2014',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='gas_value_2014',y='gas_exports',data=dataset)
plt.scatter(x='gas_value_2014',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='gas_value_2014',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='gas_value_2014',y='net_gas_exports_value',data=dataset)
plt.scatter(x='gas_value_2014',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_value_2014',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_value_2014',y='population',data=dataset)
plt.scatter(x='gas_value_2014',y='pop_maddison',data=dataset)

plt.scatter(x='oil_gas_value_nom',y='oil_gas_value_2000',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='oil_exports',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='gas_exports',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='population',data=dataset)
plt.scatter(x='oil_gas_value_nom',y='pop_maddison',data=dataset)

plt.scatter(x='oil_gas_value_2000',y='oil_gas_value_2014',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='oil_exports',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='gas_exports',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='population',data=dataset)
plt.scatter(x='oil_gas_value_2000',y='pop_maddison',data=dataset)

plt.scatter(x='oil_gas_value_2014',y='oil_gas_valuePOP_nom',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='oil_exports',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='gas_exports',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='population',data=dataset)
plt.scatter(x='oil_gas_value_2014',y='pop_maddison',data=dataset)

plt.scatter(x='oil_gas_valuePOP_nom',y='oil_gas_valuePOP_2000',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='oil_exports',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='gas_exports',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='population',data=dataset)
plt.scatter(x='oil_gas_valuePOP_nom',y='pop_maddison',data=dataset)

plt.scatter(x='oil_gas_valuePOP_2000',y='oil_gas_valuePOP_2014',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='oil_exports',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='gas_exports',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='population',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2000',y='pop_maddison',data=dataset)

plt.scatter(x='oil_gas_valuePOP_2014',y='oil_exports',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='gas_exports',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='population',data=dataset)
plt.scatter(x='oil_gas_valuePOP_2014',y='pop_maddison',data=dataset)

plt.scatter(x='oil_exports',y='net_oil_exports',data=dataset)
plt.scatter(x='oil_exports',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='oil_exports',y='net_oil_exports_value',data=dataset)
plt.scatter(x='oil_exports',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='oil_exports',y='gas_exports',data=dataset)
plt.scatter(x='oil_exports',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='oil_exports',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='oil_exports',y='net_gas_exports_value',data=dataset)
plt.scatter(x='oil_exports',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_exports',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='oil_exports',y='population',data=dataset)
plt.scatter(x='oil_exports',y='pop_maddison',data=dataset)

plt.scatter(x='net_oil_exports',y='net_oil_exports_mt',data=dataset)
plt.scatter(x='net_oil_exports',y='net_oil_exports_value',data=dataset)
plt.scatter(x='net_oil_exports',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='net_oil_exports',y='gas_exports',data=dataset)
plt.scatter(x='net_oil_exports',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='net_oil_exports',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='net_oil_exports',y='net_gas_exports_value',data=dataset)
plt.scatter(x='net_oil_exports',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_oil_exports',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_oil_exports',y='population',data=dataset)
plt.scatter(x='net_oil_exports',y='pop_maddison',data=dataset)

plt.scatter(x='net_oil_exports_mt',y='net_oil_exports_value',data=dataset)
plt.scatter(x='net_oil_exports_mt',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='net_oil_exports_mt',y='gas_exports',data=dataset)
plt.scatter(x='net_oil_exports_mt',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='net_oil_exports_mt',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='net_oil_exports_mt',y='net_gas_exports_value',data=dataset)
plt.scatter(x='net_oil_exports_mt',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_oil_exports_mt',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_oil_exports_mt',y='population',data=dataset)
plt.scatter(x='net_oil_exports_mt',y='pop_maddison',data=dataset)

plt.scatter(x='net_oil_exports_value',y='net_oil_exports_valuePOP',data=dataset)
plt.scatter(x='net_oil_exports_value',y='gas_exports',data=dataset)
plt.scatter(x='net_oil_exports_value',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='net_oil_exports_value',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='net_oil_exports_value',y='net_gas_exports_value',data=dataset)
plt.scatter(x='net_oil_exports_value',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_oil_exports_value',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_oil_exports_value',y='population',data=dataset)
plt.scatter(x='net_oil_exports_value',y='pop_maddison',data=dataset)

plt.scatter(x='net_oil_exports_valuePOP',y='gas_exports',data=dataset)
plt.scatter(x='net_oil_exports_valuePOP',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='net_oil_exports_valuePOP',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='net_oil_exports_valuePOP',y='net_gas_exports_value',data=dataset)
plt.scatter(x='net_oil_exports_valuePOP',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_oil_exports_valuePOP',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_oil_exports_valuePOP',y='population',data=dataset)
plt.scatter(x='net_oil_exports_valuePOP',y='pop_maddison',data=dataset)

plt.scatter(x='gas_exports',y='net_gas_exports_bcf',data=dataset)
plt.scatter(x='gas_exports',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='gas_exports',y='net_gas_exports_value',data=dataset)
plt.scatter(x='gas_exports',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_exports',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='gas_exports',y='population',data=dataset)
plt.scatter(x='gas_exports',y='pop_maddison',data=dataset)

plt.scatter(x='net_gas_exports_bcf',y='net_gas_exports_mboe',data=dataset)
plt.scatter(x='net_gas_exports_bcf',y='net_gas_exports_value',data=dataset)
plt.scatter(x='net_gas_exports_bcf',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_gas_exports_bcf',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_gas_exports_bcf',y='population',data=dataset)
plt.scatter(x='net_gas_exports_bcf',y='pop_maddison',data=dataset)

plt.scatter(x='net_gas_exports_mboe',y='net_gas_exports_value',data=dataset)
plt.scatter(x='net_gas_exports_mboe',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_gas_exports_mboe',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_gas_exports_mboe',y='population',data=dataset)
plt.scatter(x='net_gas_exports_mboe',y='pop_maddison',data=dataset)

plt.scatter(x='net_gas_exports_value',y='net_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_gas_exports_value',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_gas_exports_value',y='population',data=dataset)
plt.scatter(x='net_gas_exports_value',y='pop_maddison',data=dataset)

plt.scatter(x='net_gas_exports_valuePOP',y='net_oil_gas_exports_valuePOP',data=dataset)
plt.scatter(x='net_gas_exports_valuePOP',y='population',data=dataset)
plt.scatter(x='net_gas_exports_valuePOP',y='pop_maddison',data=dataset)

plt.scatter(x='net_oil_gas_exports_valuePOP',y='population',data=dataset)
plt.scatter(x='net_oil_gas_exports_valuePOP',y='pop_maddison',data=dataset)

plt.scatter(x='population',y='pop_maddison',data=dataset)

sns.pairplot(dataset)
sns.pairplot(dataset, diag_kind='kde')
sns.pairplot(dataset, hue='y')

# Heatmap
dataset.corr()
sns.heatmap(dataset.corr(), annot=True)

# Outliers
# Z score
from scipy import stats
z = np.abs(stats.zscore(imputed_knn))
threshold=3
print(np.where(z>3))
imputed_knn.shape
df_out = imputed_knn[(z<3).all(axis=1)] 
df_out.shape # 2101 rows are removed


# Metrics of features
x = df_out.drop('oil_price_2000',axis=1).values
y = df_out.iloc[:,1].values

# Check Zero inflation
from matplotlib import pyplot as plt
import numpy as np
import math as math
import pandas as pd
from scipy import stats
import statsmodels.discrete.count_model as reg_models
#help(reg_models)

# Generate model object itself
out = reg_models.ZeroInflatedPoisson(y,x,x,inflation='logit')
# Fit it
fit_regularized = out.fit_regularized(maxiter = 35) # Essentially forces convergence by penalizing Biases estimates
fit_regularized.params # notice that these are regularized values, not the true values. The Ordinal scale of the variables.

# Lets try a real fit, no regularization, no biasing the estimates
fit = out.fit(method='nm',maxiter=1000) # may need more than the default 35 iterations, very small number!
fit.params

# Splitting the dataset into train set and test set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=454)

# Check Zero inflation.
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

print("Model: Zero Inflated Poisson")
zip_mod = sm.ZeroInflatedPoisson(y_train, x_train).fit(method="nm", maxiter=50)
zip_mean_pred = zip_mod.predict(x_test,exog_infl=np.ones((len(x_test), 1)))
zip_ppf_obs = stats.poisson.ppf(q=0.95, mu=zip_mean_pred)
zip_rmse = np.sqrt(mean_squared_error(y_test, zip_ppf_obs))

print("Model: Zero Inflated Neg. Binomial")
zinb_mod = sm.ZeroInflatedNegativeBinomialP(y_train, x_train).fit(method="nm", maxiter=50)
zinb_pred = zinb_mod.predict(x_test,exog_infl=np.ones((len(x_test),1)))
zinb_rmse = np.sqrt(mean_squared_error(y_test, zinb_pred))

print("RMSE ZIP", zip_rmse)
print("RMSE ZINB: ", zinb_rmse)


'''# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) '''

# Fitting Multiple Linear Regression on train set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression(normalize=True)
regressor.fit(x_train,y_train)

# Prediction on train set
reg_train_pred = regressor.predict(x_train)
# Train set residuals
reg_train_resid = reg_train_pred - y_train
# RMSE value of train set
reg_train_rmse = np.sqrt(np.mean(reg_train_resid**2))
print(reg_train_rmse) # 5.52

# Prediction on test set
reg_test_pred = regressor.predict(x_test)
# Test set residuals
reg_test_resid = reg_test_pred - y_test
# RMSE value of test set
reg_test_rmse = np.sqrt(np.mean(reg_test_resid**2))
print(reg_test_rmse) # 5.46


# Building the Optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((13005,1)).astype(int), values=x, axis=1) # Adding X metrics to 1's Column metrix

# With all the independent variables
X_opt = X[:,:] # Metrix containing all the independent variables
model1 = sm.OLS(endog=y, exog=X_opt).fit()
model1.summary() # R^2 = 0.899 & Adj.R^2 = 0.899
model1.params
# Confidence value 99%
print(model1.conf_int(0.01))

# Remove X4,X5,X14,X15,X16,X21,X23,X28,X29
X_opt = np.delete(X,[4,5,14,15,16,21,23,28,29], axis=1)
#X_opt = X_opt[:,[0,1,2,3,6,7,8,9,10,11,12,13,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35]]
model2= sm.OLS(endog=y, exog=X_opt).fit()
model2.summary() # R^2 = 0.896 & Adj.R^2 = 0.896
model2.params
# Confidence value 99%
print(model2.conf_int(0.01))

# Remove X9,X10,X14,X15,X16
X_opt = np.delete(X_opt,[9,10,14,15,16],axis=1)
#X_opt = X_opt[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,19,20,21,22,23,24,25,26,27,28,29,30]]
model3= sm.OLS(endog=y, exog=X_opt).fit()
model3.summary() # R^2 = 0.896 & Adj.R^2 = 0.896
model3.params
# Confidence value 99%
print(model3.conf_int(0.01))

# Remove X3,X9
X_opt = np.delete(X_opt,[3,9],axis=1)
#X_opt = X_opt[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,17,18,19,20,21,22,23,24,25,26,27,28,29]]
model4= sm.OLS(endog=y, exog=X_opt).fit()
model4.summary() # R^2 = 0.896 & Adj.R^2 = 0.896
model4.params
# Confidence value 99%
print(model4.conf_int(0.01))

# Checking whether data has any influentail values
# Influence index plots 
# Model1
import statsmodels.api as sm
sm.graphics.influence_plot(model1) # index 9925,9920,9927,9929,9921,5537,9914,9910,3859 showing high influence so we can exclude that entire row
# Studentized Residuals = Residual/standard deviation of residuals

new_X = np.delete(X, [9925,9927,9920,9929,9921,5537,9914,9910,3859], axis=0)
new_X_opt = new_X[:,:]
new_Y = np.delete(y, [9925,9927,9920,9929,9921,5537,9914,9910,3859], axis=0)

# Preparing a model
model1_new = sm.OLS(endog=new_Y, exog=new_X_opt).fit()
model1_new.summary() # R^2 = 0.896 and Adj.R^2 = 0.896
model1_new.params
# Confidence value 99%
print(model1_new.conf_int(0.01))

# Model 4
sm.graphics.influence_plot(model4)
# For model4 we get 9925,5537,9927 index showing high influence
# For model4 we get 2946,12325,5520,12224,8729,12533,8442,2121,4381,8918,9924,9017,8849,
# 5624,8729,793,11865,8437,2116,12335,12336,9016,12493,2115,8870,12698,2114,2115,884,5987,2112 index showing high influence

new_X = np.delete(X_opt,[2946,12325,5520,12224,8729,12533,8442,2121,4381,8918,9924,9017,8849,5624,8729,793,11865,8437,2116,
                         12335,12336,9016,12493,2115,8870,12698,2114,2115,884,5987,2112],axis=0)
new_X_opt = new_X[:,:]
new_Y = np.delete(y,[2946,12325,5520,12224,8729,12533,8442,2121,4381,8918,9924,9017,8849,5624,8729,793,11865,8437,2116,
                         12335,12336,9016,12493,2115,8870,12698,2114,2115,884,5987,2112],axis=0)

# Preapring new model
model2_new = sm.OLS(endog=new_Y, exog=new_X_opt).fit()
model2_new.summary() # R^2 = 0.897 and Adj.R^2 = 0.897
model2_new.params
# Confidence value 99%
print(model2_new.conf_int(0.01))

# Exclude only 12335,12336,9016,9017,884,5537 index only
new_X = np.delete(X_opt, [12335,12336,9016,9017,884,5537], axis=0)
new_X_opt = new_X[:,:]
new_Y = np.delete(y, [12335,12336,9016,9017,884,5537], axis=0)

# Preparing new model
model3_new = sm.OLS(endog=new_Y, exog=new_X_opt).fit()
model3_new.summary() # R^2 = 0.896 and Adj.R^2 = 0.896
model3_new.params
# Confidence value 99%
print(model3_new.conf_int(0.01))


# Calculating VIF's values of independent variables
rsq_oil_prod32_14 = sm.OLS(endog=X[:,1], exog=np.delete(X,[1],axis=1)).fit().rsquared
vif_oil_prod32_14 = 1/(1-rsq_oil_prod32_14)
print(vif_oil_prod32_14) # 3.1958

rsq_oil_price_nom = sm.OLS(endog=X[:,2], exog=np.delete(X,[2],axis=1)).fit().rsquared
vif_oil_price_nom = 1/(1-rsq_oil_price_nom)
print(vif_oil_price_nom) # 4.5625

rsq_oil_value_nom = sm.OLS(endog=X[:,3], exog=np.delete(X,[3],axis=1)).fit().rsquared
vif_oil_value_nom = 1/(1-rsq_oil_value_nom)
print(vif_oil_value_nom) # 481.6928

rsq_oil_value_2000 = sm.OLS(endog=X[:,4], exog=np.delete(X,[4],axis=1)).fit().rsquared
vif_oil_value_2000 = 1/(1-rsq_oil_value_2000)
print(vif_oil_value_2000) # 7522139.0434

rsq_oil_value_2014 = sm.OLS(endog=X[:,5], exog=np.delete(X,[5],axis=1)).fit().rsquared
vif_oil_value_2014 = 1/(1-rsq_oil_value_2014)
print(vif_oil_value_2014) # 7574513.6065

rsq_gas_prod55_14 = sm.OLS(endog=X[:,6], exog=np.delete(X,[6],axis=1)).fit().rsquared
vif_gas_prod55_14 = 1/(1-rsq_gas_prod55_14)
print(vif_gas_prod55_14) # 14.8204

rsq_gas_price_2000_mboe = sm.OLS(endog=X[:,7], exog=np.delete(X,[7],axis=1)).fit().rsquared
vif_gas_price_2000_mboe = 1/(1-rsq_gas_price_2000_mboe)
print(vif_gas_price_2000_mboe) # 15.4386

rsq_gas_price_2000 = sm.OLS(endog=X[:,8], exog=np.delete(X,[8],axis=1)).fit().rsquared
vif_gas_price_2000 = 1/(1-rsq_gas_price_2000)
print(vif_gas_price_2000) # 10.7389

rsq_gas_price_nom = sm.OLS(endog=X[:,9], exog=np.delete(X,[9],axis=1)).fit().rsquared
vif_gas_price_nom = 1/(1-rsq_gas_price_nom)
print(vif_gas_price_nom) # 6.2611

rsq_gas_value_nom = sm.OLS(endog=X[:,10], exog=np.delete(X,[10],axis=1)).fit().rsquared
vif_gas_value_nom = 1/(1-rsq_gas_value_nom)
print(vif_gas_value_nom) # 125.7108

rsq_gas_value_2000 = sm.OLS(endog=X[:,11], exog=np.delete(X,[11],axis=1)).fit().rsquared
vif_gas_value_2000 = 1/(1-rsq_gas_value_2000)
print(vif_gas_value_2000) # 12855360.4737

rsq_gas_value_2014 = sm.OLS(endog=X[:,12], exog=np.delete(X,[12],axis=1)).fit().rsquared
vif_gas_value_2014 = 1/(1-rsq_gas_value_2014)
print(vif_gas_value_2014) # 12762911.5862

rsq_oil_gas_value_nom = sm.OLS(endog=X[:,13], exog=np.delete(X,[13],axis=1)).fit().rsquared
vif_oil_gas_value_nom = 1/(1-rsq_oil_gas_value_nom)
print(vif_oil_gas_value_nom) # 759.0077

rsq_oil_gas_value_2000 = sm.OLS(endog=X[:,14], exog=np.delete(X,[14],axis=1)).fit().rsquared
vif_oil_gas_value_2000 = 1/(1-rsq_oil_gas_value_2000)
print(vif_oil_gas_value_2000) # 20522213.2100

rsq_oil_gas_value_2014 = sm.OLS(endog=X[:,15], exog=np.delete(X,[15],axis=1)).fit().rsquared
vif_oil_gas_value_2014 = 1/(1-rsq_oil_gas_value_2014)
print(vif_oil_gas_value_2014) # 20505938.0639

rsq_oil_gas_valuePOP_nom = sm.OLS(endog=X[:,16], exog=np.delete(X,[16],axis=1)).fit().rsquared
vif_oil_gas_valuePOP_nom = 1/(1-rsq_oil_gas_valuePOP_nom)
print(vif_oil_gas_valuePOP_nom) # 3.4337

rsq_oil_gas_valuePOP_2000 = sm.OLS(endog=X[:,17], exog=np.delete(X,[17],axis=1)).fit().rsquared
vif_oil_gas_valuePOP_2000 = 1/(1-rsq_oil_gas_valuePOP_2000)
print(vif_oil_gas_valuePOP_2000) # divide by zero = inf

rsq_oil_gas_valuePOP_2014 = sm.OLS(endog=X[:,18], exog=np.delete(X,[18],axis=1)).fit().rsquared
vif_oil_gas_valuePOP_2014 = 1/(1-rsq_oil_gas_valuePOP_2014)
print(vif_oil_gas_valuePOP_2014) # divide by zero = inf

rsq_oil_exports = sm.OLS(endog=X[:,19], exog=np.delete(X,[19],axis=1)).fit().rsquared
vif_oil_exports = 1/(1-rsq_oil_exports)
print(vif_oil_exports) # 3.3782

rsq_net_oil_exports = sm.OLS(endog=X[:,20], exog=np.delete(X,[20],axis=1)).fit().rsquared
vif_net_oil_exports = 1/(1-rsq_net_oil_exports)
print(vif_net_oil_exports) # divide by zero = inf

rsq_net_oil_exports_mt = sm.OLS(endog=X[:,21], exog=np.delete(X,[21],axis=1)).fit().rsquared
vif_net_oil_exports_mt = 1/(1-rsq_net_oil_exports_mt)
print(vif_net_oil_exports_mt) # divide by zero = inf

rsq_net_oil_exports_value = sm.OLS(endog=X[:,22], exog=np.delete(X,[22],axis=1)).fit().rsquared
vif_net_oil_exports_value = 1/(1-rsq_net_oil_exports_value)
print(vif_net_oil_exports_value) # 3.7530

rsq_net_oil_exports_valuePOP = sm.OLS(endog=X[:,23], exog=np.delete(X,[23],axis=1)).fit().rsquared
vif_net_oil_exports_valuePOP = 1/(1-rsq_net_oil_exports_valuePOP)
print(vif_net_oil_exports_valuePOP) # 8.9565

rsq_gas_exports = sm.OLS(endog=X[:,24], exog=np.delete(X,[24],axis=1)).fit().rsquared
vif_gas_exports = 1/(1-rsq_gas_exports)
print(vif_gas_exports) # 3.5649

rsq_net_gas_exports_bcf = sm.OLS(endog=X[:,25], exog=np.delete(X,[25],axis=1)).fit().rsquared
vif_net_gas_exports_bcf = 1/(1-rsq_net_gas_exports_bcf)
print(vif_net_gas_exports_bcf) # divide by zero = inf

rsq_net_gas_exports_mboe = sm.OLS(endog=X[:,26], exog=np.delete(X,[26],axis=1)).fit().rsquared
vif_net_gas_exports_mboe = 1/(1-rsq_net_gas_exports_mboe)
print(vif_net_gas_exports_mboe) # divide by zero = inf

rsq_net_gas_exports_value = sm.OLS(endog=X[:,27], exog=np.delete(X,[27],axis=1)).fit().rsquared
vif_net_gas_exports_value = 1/(1-rsq_net_gas_exports_value)
print(vif_net_gas_exports_value) # 3.9772

rsq_net_gas_exports_valuePOP = sm.OLS(endog=X[:,28], exog=np.delete(X,[28],axis=1)).fit().rsquared
vif_net_gas_exports_valuePOP = 1/(1-rsq_net_gas_exports_valuePOP)
print(vif_net_gas_exports_valuePOP) # 2.0013

rsq_net_oil_gas_exports_valuePOP = sm.OLS(endog=X[:,29], exog=np.delete(X,[29],axis=1)).fit().rsquared
vif_net_oil_gas_exports_valuePOP = 1/(1-rsq_net_oil_gas_exports_valuePOP)
print(vif_net_oil_gas_exports_valuePOP) # 10.0335

rsq_population = sm.OLS(endog=X[:,30], exog=np.delete(X,[30],axis=1)).fit().rsquared
vif_population = 1/(1-rsq_population)
print(vif_population) # 22.4059

rsq_pop_maddison = sm.OLS(endog=X[:,31], exog=np.delete(X,[31],axis=1)).fit().rsquared
vif_pop_maddison = 1/(1-rsq_pop_maddison)
print(vif_pop_maddison) # 22.3892

rsq_sovereign = sm.OLS(endog=X[:,32], exog=np.delete(X,[32],axis=1)).fit().rsquared
vif_sovereign = 1/(1-rsq_sovereign)
print(vif_sovereign) # 1.3931

rsq_mult_nom_2000 = sm.OLS(endog=X[:,33], exog=np.delete(X,[33],axis=1)).fit().rsquared
vif_mult_nom_2000 = 1/(1-rsq_mult_nom_2000)
print(vif_mult_nom_2000) # 9007199254740992.0

rsq_mult_nom_2014 = sm.OLS(endog=X[:,34], exog=np.delete(X,[34],axis=1)).fit().rsquared
vif_mult_nom_2014 = 1/(1-rsq_mult_nom_2014)
print(vif_mult_nom_2014) # divide by zero = inf

rsq_mult_2000_2014 = sm.OLS(endog=X[:,35], exog=np.delete(X,[35],axis=1)).fit().rsquared
vif_mult_2000_2014 = 1/(1-rsq_mult_2000_2014)
print(vif_mult_2000_2014) # 4.0557

# Sorting VIF's values in dataframe
d1 = {'variables':['oil_prod32_14','oil_price_nom','oil_value_nom','oil_value_2000','oil_value_2014','gas_prod55_14',
                   'gas_price_2000_mboe', 'gas_price_2000', 'gas_price_nom','gas_value_nom', 'gas_value_2000', 'gas_value_2014',
                   'oil_gas_value_nom', 'oil_gas_value_2000', 'oil_gas_value_2014','oil_gas_valuePOP_nom', 'oil_gas_valuePOP_2000',
                   'oil_gas_valuePOP_2014', 'oil_exports', 'net_oil_exports','net_oil_exports_mt', 'net_oil_exports_value',
                   'net_oil_exports_valuePOP', 'gas_exports', 'net_gas_exports_bcf','net_gas_exports_mboe', 'net_gas_exports_value',
                   'net_gas_exports_valuePOP', 'net_oil_gas_exports_valuePOP','population', 'pop_maddison', 'sovereign', 'mult_nom_2000','mult_nom_2014', 'mult_2000_2014'],
'VIF':[vif_oil_prod32_14,vif_oil_price_nom,vif_oil_value_nom,vif_oil_value_2000,vif_oil_value_2014,vif_gas_prod55_14,
       vif_gas_price_2000_mboe,vif_gas_price_2000,vif_gas_price_nom,vif_gas_value_nom,vif_gas_value_2000,vif_gas_value_2014,vif_oil_gas_value_nom,vif_oil_gas_value_2000,
       vif_oil_gas_value_2014,vif_oil_gas_valuePOP_nom,vif_oil_gas_valuePOP_2000,vif_oil_gas_valuePOP_2014,vif_oil_exports,vif_net_oil_exports,vif_net_oil_exports_mt,
       vif_net_oil_exports_value,vif_net_oil_exports_valuePOP,vif_gas_exports,vif_net_gas_exports_bcf,vif_net_gas_exports_mboe,vif_net_gas_exports_value,
       vif_net_gas_exports_valuePOP,vif_net_oil_gas_exports_valuePOP,vif_population,vif_pop_maddison,vif_sovereign,vif_mult_nom_2000,vif_mult_nom_2014,vif_mult_2000_2014]}

vif_frame = pd.DataFrame(d1)
vif_frame
# Some of  variables satisfies the VIF <10 and some variables exceeds the threshold value.

# Added Variable plot
fig = plt.figure(figsize=(20,12))
fig=sm.graphics.plot_partregress_grid(model1,fig=fig)

fig = plt.figure(figsize=(20,12))
fig=sm.graphics.plot_partregress_grid(model2_new,fig=fig)

fig = plt.figure(figsize=(20,12))
fig=sm.graphics.plot_partregress_grid(model3_new,fig=fig)
# All variables are changing with target variable

# model2_new is the final model and R^2 = 0.896 & Adj.R^2 = 0.896
# Exclude only 9925
new_Y_pred = model2_new.predict(new_X_opt)
new_Y_pred
# Added Variable Plot for final model

# Linearity
# Observed values VS fitted values
plt.scatter(new_Y, new_Y_pred, c='r');plt.xlabel('Observed_Values');plt.ylabel('Fitted_Values')

# Residuals VS fitted values
#plt.scatter(new_Y_pred, model2_new.resid_pearson, c='b'),plt.axhline(y=0, color='red');plt.xlabel('Fitted_Values');plt.ylabel('Residuals')

# Normality plot for Residuals
# Histogram
plt.hist(model2_new.resid_pearson) # Checking the standardized residuals are normallly distributed

# Q-Q plot 
stats.probplot(model2_new.resid_pearson, dist='norm', plot=plt)

# Homoscedasticity
# Residuals VS fitted values
plt.scatter(new_Y_pred, model2_new.resid_pearson, c='r'),plt.axhline(y=0, color='blue');plt.xlabel('Fitted_Values');plt.ylabel('Residuals')

# Splitting the new data into train set and test set
from sklearn.model_selection import train_test_split
newX_train, newX_test, newY_train, newY_test = train_test_split(new_X_opt, new_Y, test_size=0.20, random_state=0)

'''# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
newX_train = sc_X.fit_transform(newX_train)
newX_test = sc_X.transform(newX_test) '''

# Prepare the model on the train set data
final_model = sm.OLS(endog=newY_train, exog=newX_train).fit()

# Train data prediction
train_pred = final_model.predict(newX_train)

# train residual values
train_resid = train_pred - newY_train

# RMSE value of train data
train_rmse = np.sqrt(np.mean(train_resid**2))
print(train_rmse) # 5.68 | 5.58(standardized)

# prediction on test set data
test_pred = final_model.predict(newX_test)

# Test set residual values
test_resid = test_pred - newY_test

# RMSE value for test data
test_rmse = np.sqrt(np.mean(test_resid**2))
print(test_rmse) # 5.69 | 5.57(standardied)
