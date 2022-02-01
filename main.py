#%%
from multiprocessing import pool
import pandas as pd
import numpy as np


#%%
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

plt.rcParams['figure.figsize'] = [12,6]
sns.set(style="darkgrid")

pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))

#%%

def unique(list1):
     
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))

    if(all(isinstance(n, not str)) for n in unique_list):
        
        unique_list = [item for item in unique_list if not(pd.isnull(item)) == True]
    
    return unique_list, len(unique_list)

def counterElements(lista):

    auxLista, _ = unique(lista)

    dicio = dict.fromkeys(auxLista, 0)
    dicio['NaN'] = 0

    for i, value in enumerate(auxLista):
        
        counter = 0

        for j in range(len(lista)):

            if lista[j] == auxLista[i]:

                counter += 1
    
        dicio[value] = counter

    dicio['NaN'] = lista.isna().sum()

    return dicio

#%%
X = pd.read_csv('train.csv')

# %%

train_ID = X['Id']

X.drop('Id', axis = 1, inplace = True)

# %%

print(X['SalePrice'].describe()['mean'])

#%%

sns.histplot(X['SalePrice'], kde = True)

print('Assimetria: {}'.format(X['SalePrice'].skew()))

print('Curtose: {}'.format(X['SalePrice'].kurt()))

# %%

correlation = X.corr()

mask = np.zeros_like(correlation)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style('white'):

    f, ax = plt.subplots()

    sns.heatmap(correlation, mask = mask, ax = ax, cbar_kws = {'shrink' : .82},
                vmax = .9, cmap = 'coolwarm', square = True)

# %%

feature = list(correlation.columns)

values = correlation.values

corrList = []

for i, val in enumerate(values[36]):

    if(val > 0.4 and feature[i] != 'SalePrice'):

        corrList.append(feature[i])
        print('SalePrice and {} correlates with value {}.'.format(feature[i], "{:.2f}".format(val)))

print('Total features related to Saleprice: {}'.format(len(corrList)))

# %%

i = 0
totalSum = 0
columnsWNaN = []

for column in corrList:
    
    nullElements = X[column].isnull().sum()

    if(nullElements):
        columnsWNaN.append(column)
        totalSum+=1


print('there are {} relevant columns in with one or more "NaN" as values: {}'.format(totalSum, columnsWNaN))

# %%

for i in columnsWNaN:

    unique_list, size = unique(X[i])

    unique_list.append(np.nan)
    size += 1

    print('{}: {}, {}'.format(i, size, unique_list))

# %%
def createPercentageOfMissingValues(lista = columnsWNaN):
    percentColumnWNaN = {}
    totalColumnWNan = {}

    for column in lista:

        percentColumnWNaN[column] = (X[column].isnull().sum() / len(X[column])) * 100
        totalColumnWNan[column] = X[column].isnull().sum()

    dfPercent = pd.DataFrame(list(percentColumnWNaN.items()), columns = ['Feature', 'Faltantes %'])

    dfPercent['Faltantes Total'] = dfPercent['Feature'].map(totalColumnWNan)

    dfPercent = dfPercent.sort_values(by = ['Faltantes %'], ascending = False)

    return dfPercent

# %%
dfPercent = createPercentageOfMissingValues(lista = X.columns)

sns.barplot(x = dfPercent['Feature'], y = dfPercent['Faltantes %'])
plt.xticks(rotation = '90')
plt.xlabel('Features', fontsize = 15)
plt.ylabel('% de valores faltantes', fontsize = 15)
plt.title('Porcentagem de valores faltantes por Feature', fontsize = 15)


#%%
corrList.append('SalePrice')

#%%

sns.set()

sns.pairplot(X[corrList], height = 2.5, corner = True)

plt.show

# %% removendo os elementos faltantes da feature GarageYrBlt

#X = X[X.GarageYrBlt.notnull()]

#print(unique(X['GarageYrBlt']))
