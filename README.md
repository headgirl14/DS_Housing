# DS_Housing
# Correlation matrix for set of data
import pandas as pd # contains function which allows you to calculate correlation matrices
import numpy as np
import matplotlib.pyplot as plt # used to create a visualisation of the correlation matrix

data = pd.read_csv('airbnb_dataset.csv')
corr_matrix = data.corr().abs()
num_rows = data.shape[1]

total_cols = len(data.axes[1])
print(f"The total number of features represented in this data set are {total_cols}")
data.info()

# correlation matrix required to be sorted first before deducing most and least correlated variables
# retain upper triangular values of the correlation matrix and makes lower triangle values null
sorted_corr_matrix = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool)).stack().sort_values(ascending=False))

most_corr_vars = sorted_corr_matrix[0:1]
print("Most correlated two variables are:")
print(most_corr_vars)

least_corr_vars = sorted_corr_matrix[(num_rows - 1):num_rows]
print("Least correlated two variables are:")
print(least_corr_vars)

print("Correlation matrix is:")
print(corr_matrix)

f = plt.figure(figsize=(19, 15))
plt.matshow(corr_matrix, fignum=f.number)
plt.xticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns, fontsize=8, rotation=45)
plt.yticks(range(corr_matrix.select_dtypes(['number']).shape[1]), corr_matrix.select_dtypes(['number']).columns, fontsize=8)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=8)
plt.title('Correlation Matrix', fontsize=16);
plt.show()

q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr = q3-q1
print(iqr)
print(data['price'].skew())
data["price"] = np.where(data["price"] <49.0, 49.0,data['price'])
data["price"] = np.where(data["price"] >350.0, 350.0,data['price'])
print(data['price'].skew())
data.boxplot(column='price', by='State', return_type='both', color='m', showfliers=False)
title_boxplot = 'Distribution of price by state'
plt.title(title_boxplot)
plt.suptitle('')
plt.show()

data.boxplot(column='price', by='room_type', return_type='both', color='m', showfliers=False)
title_boxplot = 'Distribution of price by room type'
plt.title(title_boxplot)
plt.suptitle('')
plt.show()
