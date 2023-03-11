### GENERAL IMPORTS ###
import numpy as np
import pandas as pd

### SKLEARN IMPORTS ###
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, RepeatedStratifiedKFold

### SKOMPTOMIZE IMPORTS ###
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize

alloys = pd.read_csv('../data/featurized_alloys.csv')

feature_selection = ['MagpieData avg_dev CovalentRadius' 'MagpieData mean NdValence'
 'MagpieData mean Electronegativity' 'MagpieData mean NsUnfilled' '0-norm'
 'MagpieData avg_dev NdUnfilled' 'MagpieData avg_dev SpaceGroupNumber'
 'MagpieData avg_dev Electronegativity' 'MagpieData mean GSvolume_pa'
 'MagpieData avg_dev GSvolume_pa' 'MagpieData mean GSmagmom'
 'MagpieData avg_dev Number' 'MagpieData mean MendeleevNumber'
 'MagpieData avg_dev AtomicWeight' 'MagpieData maximum MeltingT'
 'MagpieData avg_dev Column' 'MagpieData range CovalentRadius'
 'MagpieData mode CovalentRadius' 'MagpieData mean NsValence'
 'MagpieData avg_dev NpValence' 'MagpieData avg_dev NpUnfilled'
 'MagpieData mean NUnfilled' 'MagpieData range MeltingT'
 'MagpieData mean Column' 'MagpieData avg_dev Row'
 'MagpieData mean CovalentRadius']

# Choose training columns from dataframe
x_cols = [c for c in alloys.columns if c in feature_selection]

y = alloys['phase'].values
X = alloys[x_cols]

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

space = [Integer(10, 10**3, 'uniform', name='n_estimators'), 
         Integer(1, 10**3, 'uniform', name='max_depth'), 
         Integer(2, 10**3, 'uniform', name='min_samples_split'), 
         Integer(1, 10**3, 'uniform', name='min_samples_leaf'), 
         Real(10**-5, 0.5, 'uniform', name='min_weight_fraction_leaf'), 
         Integer(1, 138, 'uniform', name='max_features'), 
         Integer(2, 10**3, 'uniform', name='max_leaf_nodes'), 
         Real(10**-5, 10**-1, 'log-uniform', name='ccp_alpha'), 
         Integer(1, 138, 'uniform', name='max_samples')]

@use_named_args(space)
def objective(**params):
    rf = RandomForestClassifier(random_state=0, n_jobs=-1)
    rf.set_params(**params)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=0)
    accuracy = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
    accuracy = np.mean(accuracy)
    return 1.0 - accuracy

n_calls = 250
result = gp_minimize(objective,
                     space,
                     n_calls=n_calls,
                     random_state=0)

plot_convergence(result)
plt.savefig('plot.png')

print(result.x)
print(1-result.fun)
print(result)