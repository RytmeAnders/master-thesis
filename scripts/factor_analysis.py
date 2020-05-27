import pandas as pd
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn import preprocessing
from factor_analyzer import FactorAnalyzer
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib


# TODO: Try non-flipped dimensions

matplotlib.rcParams['figure.figsize'] = (10.0, 6.0)

plt.style.use('ggplot')

####### Setup the dataset
df = pd.read_csv("dims_new.csv",delimiter=',',header=0)
#df.drop(['Unnamed: 0','gender', 'education', 'age'],axis=1,inplace=True)
# Dropping missing values rows
#df.dropna(inplace=True)
df.head()


####### Find eigenvalues
fa = FactorAnalyzer(rotation=None, n_factors=17)
fa = fa.fit(df) # used instead of "fa.analyze(df, 17, rotation=None)"
ev, v = fa.get_eigenvalues() # Check Eigenvalues

####### Plot eigenvalues
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.axhline(y=1,c='k')


##### Eigenvalues suggest that 6 dimensions is a good fit
def loadThem(rotation, factors):
    fa = FactorAnalyzer(rotation=rotation, n_factors=factors)
    fa = fa.fit(df.values)
    loadings = fa.loadings_
    
    # Visualize factor loadings
    import numpy as np
    Z=np.abs(fa.loadings_)
    fig, ax = plt.subplots()
    c = ax.pcolor(Z)
    fig.colorbar(c, ax=ax)
    ax.set_yticks(np.arange(fa.loadings_.shape[0])+0.5, minor=False)
    ax.set_xticks(np.arange(fa.loadings_.shape[1])+0.5, minor=False)
    ax.set_title(rotation)
    plt.show()

    vari = fa.get_factor_variance()
    
    return loadings, vari

j = loadThem('promax',6) #6 = 68% variance explained (4 = 48%)
k = loadThem('varimax',6) # 6 = ~63% variance explained (4 = 48%)
l = loadThem(None,6)