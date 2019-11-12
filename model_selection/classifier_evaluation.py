import pandas as pd
import math
import numpy as np


from sklearn.model_selection import cross_validate, GridSearchCV, StratifiedKFold
from  sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from  sklearn.naive_bayes import BernoulliNB, GaussianNB
from  sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from  sklearn.neural_network import MLPClassifier
from  sklearn.svm import LinearSVC, NuSVC, SVC
from  sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from  sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from  sklearn.gaussian_process import GaussianProcessClassifier
from  sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from imblearn.ensemble import EasyEnsembleClassifier, BalancedBaggingClassifier, BalancedRandomForestClassifier

def model_evaluation(X, y, cv=StratifiedKFold(n=3), classifiers='all', scorer=None,
                     transforms=None, sampler=None, grids=None):
    
    '''
    Args:
        X (pd.DataFrame): independent variables
        y (pd.Series): response variable
        cv : cross validation object, Default to StratifiedKFold(n=3)
        classifiers (list or str): list of classifier objects to evaluate. if 'all' it will default to 
            an assembled list of all(or most) of sklearn's classifiers
        scorer (list or str): list or one of sklearn.metrics.SCORERS. If None will default to sklearn's default
            if using grid search scorer must be str object
        transforms (sklearn.pipeline.Pipeline): Pipeline of transformations.
        sampler (imblearn sampler): Used to balance data sets
        grids
    '''

    if grids is not None and isinstance(scorer, list) and len(scorer) > 1:
        raise ValueError('Grids only accept the use of one metric')

    if transforms is not None:
        X = transforms.fit_transform(X, y)

    if sampler is not None:
        X, y = sampler.fit_resample(X, y)

    if classifiers=='all':
        classifiers = [
            DummyClassifier(), LogisticRegression(), PassiveAggressiveClassifier(), RidgeClassifier(), SGDClassifier(), \
            KNeighborsClassifier(), MLPClassifier(), LinearSVC(), \
            NuSVC(), SVC(), DecisionTreeClassifier(), ExtraTreeClassifier(), AdaBoostClassifier(), \
            BaggingClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(), \
            RandomForestClassifier(), GaussianProcessClassifier(), \
            EasyEnsembleClassifier(), BalancedBaggingClassifier(), BalancedRandomForestClassifier(),
            XGBClassifier()
    ]

    results = []
    for clf in classifiers:
        name = clf.__class__.__name__
        try:
            grid = grids[name]
        except:
            grid = None

        if grid is not None:
            gridcv = GridSearchCV(clf, grid, cv =cv, scoring=scorer, refit=scorer).fit(X, y)
            cross_val = {
                'model':name, 'params':gridcv.best_params_ ,
                'test_score':gridcv.best_score_
            }                
        else:
            cross_val = cross_validate(clf, X, y, cv=cv, scoring=scorer)
            
            for key in cross_val:
                if isinstance(cross_val[key], np.ndarray) and len(cross_val[key])==1:
                    cross_val[key] = float(cross_val[key])
            cross_val.pop('fit_time', None)
            cross_val.pop('score_time', None)
            cross_val['model'] = name
            cross_val['params'] = 'default'
    
        results.append(cross_val)    

    return pd.DataFrame(results)


