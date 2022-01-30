#%%
import pandas as pd
import numpy as np


#%%
import seaborn as sns

import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))

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

def unique(list1):
     
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))

    if(all(isinstance(n, not str)) for n in unique_list):
        
        unique_list = [item for item in unique_list if not(pd.isnull(item)) == True]
        unique_list.append(np.nan) 
    
    return unique_list, len(unique_list)

for i in columnsWNaN:

    unique_list, size = unique(X[i])

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
# %%
