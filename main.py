#%%
from multiprocessing import pool
import pandas as pd
import numpy as np


#%%
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

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
            
            if lista[j] == auxLista[i] or (value == 'NaN' and np.isnan(lista[j])):

                counter += 1
    
        dicio[value] = counter

    dicio['NaN'] = lista.isna().sum()

    return dicio

#%%
X = pd.read_csv('test.csv')

# %%

i = 0
totalSum = 0
columnsWNaN = []

for column in X:

    nullElements = X[column].isnull().sum()

    if(nullElements):
        columnsWNaN.append(column)
        #print('column {} ({}): {}'.format(i, column, nullElements))
        totalSum+=1

    i+=1

print('there are {} columns with one or more "NaN" as values: {}'.format(totalSum, columnsWNaN))

# %%

for i in columnsWNaN:

    unique_list, size = unique(X[i])

    unique_list.append(np.nan)
    size += 1

    print('{}: {}, {}'.format(i, size, unique_list))

# %%
percentColumnWNaN = {}

for column in columnsWNaN:

    percentColumnWNaN[column] = (X[column].isnull().sum() / len(X[column])) * 100

dfPercent = pd.DataFrame(list(percentColumnWNaN.items()), columns = ['Feature', 'Faltantes %'])

dfPercent = dfPercent.sort_values(by = ['Faltantes %'], ascending = False)

print(dfPercent)

# %%

sns.barplot(x = dfPercent['Feature'], y = dfPercent['Faltantes %'])
plt.xticks(rotation = '90')
plt.xlabel('Features', fontsize = 15)
plt.ylabel('% de valores faltantes', fontsize = 15)
plt.title('Porcentagem de valores faltantes por Feature', fontsize = 15)

# %% tratando os faltantes da feature PoolQC
print(unique(X['PoolQC']))

X['PoolQC'].fillna('NA', inplace = True)

# %% tratando os faltantes da feature MiscFeature

print(counterElements(X['MiscFeature']))
print(counterElements(X['MiscVal']))

X['MiscFeature'].fillna('NA', inplace = True)

# %% tratando os faltantes da feature Alley
print(unique(X['Alley']))

X['Alley'].fillna('NA', inplace = True)

# %% tratando os faltantes da feature Fence
print(unique(X['Fence']))

X['Fence'].fillna('NA', inplace = True)

# %%
print(counterElements(X['Fireplaces']))
print(counterElements(X['FireplaceQu']))

X['FireplaceQu'].fillna('NA', inplace = True)

# %%
