import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def getAndIndexAndGetChange(loc, ticker):
    def p2f(x):
        return float(x.strip('%'))/100
    A = pd.read_csv(loc, converters={'Change %':p2f})[::-1]
    A['Date'] = pd.DatetimeIndex(pd.to_datetime(A['Date'], format='%b %d, %Y'))
    A = A.set_index('Date')
    A = pd.DataFrame(A['Change %'])
    A.columns = [ticker]
    A[ticker] = StandardScaler().fit_transform(A[[ticker]]) # IMPORTANT
    return A

A = getAndIndexAndGetChange('Data/ABGJ Historical Data.csv', 'ABSA') # stock
B = getAndIndexAndGetChange('Data/FTSE_JSE All Share Historical Data.csv', 'JSE ASI') # index
C = getAndIndexAndGetChange('Data/Gold Futures Historical Data.csv', 'GOLD') # market indicator
D = getAndIndexAndGetChange('Data/KIOJ Historical Data.csv', 'KUMBA') # stock
E = getAndIndexAndGetChange('Data/BTIJ Historical Data.csv', 'BAT') # stock
F = getAndIndexAndGetChange('Data/USD_ZAR Historical Data.csv', 'USD/ZAR') # market indicator

merged = pd.merge(A, B, how='inner', left_index=True, right_index=True)
merged = pd.merge(merged, C, how='inner', left_index=True, right_index=True)
merged = pd.merge(merged, D, how='inner', left_index=True, right_index=True)
merged = pd.merge(merged, E, how='inner', left_index=True, right_index=True)
merged = pd.merge(merged, F, how='inner', left_index=True, right_index=True)
X = merged.values

pca = PCA()
pca.fit(X)

components = pd.DataFrame(pca.components_, columns=list(merged.columns))
components.index.names = ['PC']
print(components.to_latex())

explainedVariance = pd.DataFrame(np.array([pca.explained_variance_, pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)]).T,  columns = ['EIGENVALUES', 'EXPLAINED VARIANCE RATIO (%)', 'CUMULATIVE (%)'])
explainedVariance.index.names = ['PC']
print(explainedVariance.to_latex())

print('Explained Variance: \n%s' %pca.explained_variance_ratio_)

featureContributions = pd.DataFrame(abs(pca.transform(np.identity(X.shape[1]))), columns=merged.columns)
featureContributions.iloc[0].plot(kind='bar', title='Contribution to 1st principal component')
featureContributions.iloc[1].plot(kind='bar', title='Contribution to 2nd principal component')
featureContributions.iloc[2].plot(kind='bar', title='Contribution to 3rd principal component')
featureContributions.iloc[3].plot(kind='bar', title='Contribution to 4th principal component')
featureContributions.iloc[4].plot(kind='bar', title='Contribution to 5th principal component')
featureContributions.iloc[5].plot(kind='bar', title='Contribution to 6th principal component')

plt.plot([1, 2, 3, 4, 5, 6], pca.explained_variance_ratio_ * 100, '-o')
plt.ylim(0,35)
plt.xlabel('Number of principal components')
plt.ylabel('Explained variance (%)')

plt.plot([1, 2, 3, 4, 5, 6], np.cumsum(pca.explained_variance_ratio_) * 100, '-o')
plt.ylim(0,105)
plt.xlabel('Number of principal components')
plt.ylabel('Cumulative explained variance (%)')