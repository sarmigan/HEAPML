### GENERAL IMPORTS ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

### SKLEARN IMPORTS ###
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold

### SKOMPTOMIZE IMPORTS ###
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

alloys = pd.read_csv('./data/featurized_alloys.csv')

feature_selection = ['0-norm', 'MagpieData avg_dev AtomicWeight', 'MagpieData avg_dev Column', 'MagpieData avg_dev CovalentRadius', 'MagpieData avg_dev Electronegativity', 'MagpieData avg_dev GSvolume_pa', 'MagpieData avg_dev NdUnfilled', 'MagpieData avg_dev NpUnfilled', 'MagpieData avg_dev NpValence', 'MagpieData avg_dev Number', 'MagpieData avg_dev Row', 'MagpieData avg_dev SpaceGroupNumber', 'MagpieData maximum MeltingT', 'MagpieData mean Column', 'MagpieData mean CovalentRadius', 'MagpieData mean Electronegativity', 'MagpieData mean GSmagmom', 'MagpieData mean GSvolume_pa', 'MagpieData mean MendeleevNumber', 'MagpieData mean NUnfilled', 'MagpieData mean NdValence', 'MagpieData mean NsUnfilled', 'MagpieData mean NsValence', 'MagpieData mode CovalentRadius', 'MagpieData range CovalentRadius', 'MagpieData range MeltingT']

n_splits=10

x_cols = [c for c in alloys.columns if c in feature_selection]

y = alloys['phase'].values
X = alloys[x_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Parameter space to use in search
space = [Integer(1, 1000, 'uniform', name='n_estimators'),
         Real(10**-3, 1, 'log-uniform', name='max_features'),
         Integer(2, 10**5, 'log-uniform', name='max_leaf_nodes'),
         Real(10**-6, 10**0, 'log-uniform', name='ccp_alpha'),
         Integer(1, int(X_train.shape[0]-(X_train.shape[0]/10)), 'uniform', name='max_samples')]

# Objective function to minimise using search
@use_named_args(space)
def objective(**params):
    rf = RandomForestClassifier(random_state=0, n_jobs=-1)
    rf.set_params(**params)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=3, random_state=0)
    accuracy = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    accuracy = np.mean(accuracy)
    return 1.0 - accuracy

# Find parameters that minimise objective function
result = gp_minimize(objective,
                     space,
                     n_calls=250,
                     random_state=0)

plot_convergence(result)
plt.savefig('plot.png')

print(result.x)
print(1-result.fun)
print(result)