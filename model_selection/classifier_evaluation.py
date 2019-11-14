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
from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline

class ClassifierEvaluator():
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
        grids :'default', None, or dictionary containing grids with keys corresponding to clf.__class__.__name__
            grid dict key names should be prefaced by "clf__" 
    '''
    



    def __init__(self, classifiers='all', scorer=None, transforms=None, sampler=None, grids=None, cv=StratifiedKFold(3), verbose=0, n_jobs=1):
       
        if grids is not None and isinstance(scorer, list) and len(scorer) > 1:
            raise ValueError('Grids only accept the use of one metric')
        
        self.classifiers_ = classifiers if classifiers != 'all' else get_classifiers()
        self.scorer_ = scorer
        self.transforms_ = transforms
        self.sampler_ = sampler
        self.grids_ = grids if grids != 'default' else get_grids()
        self.cv_ = cv
        self.verbose = verbose
        self.n_jobs = n_jobs

    def evaluate(self, X, y):
        results = []
        for clf in self.classifiers_:
            pipe = self.transforms_.copy()
            pipe.append(('clf', clf))
            pipe = Pipeline(pipe)
            name = clf.__class__.__name__
            try:
                grid = self.grids_[name]
            except:
                grid = None
            if grid is not None:
                gridcv = GridSearchCV(pipe, grid, cv=self.cv_, scoring=self.scorer_, refit=self.scorer_, verbose=self.verbose, n_jobs=self.n_jobs).fit(X, y)
                cross_val = {
                    'model':name, 'params':gridcv.best_params_ ,
                    'test_score':gridcv.best_score_
                }                
            else:
                cross_val = cross_validate(pipe, X, y, cv=self.cv_, scoring=self.scorer_)
                
                for key in cross_val:
                    if isinstance(cross_val[key], np.ndarray) and len(cross_val[key])==1:
                        cross_val[key] = float(cross_val[key])
                cross_val.pop('fit_time', None)
                cross_val.pop('score_time', None)
                cross_val['model'] = name
                cross_val['params'] = 'default'
        
            results.append(cross_val)    
            self.results = pd.DataFrame(results)
        return self.results

def get_grids():
    grids = {
        'LogisticRegression': [{
            'clf__penalty':['elasticnet'],
            'clf__solver':['saga'],
            'clf__C':[.001, .01, .1 , 1, 10],
            'clf__max_iter':[100, 500, 1000],
            'clf__l1_ratio':[0, .5, 1]
        }],
        'PassiveAggressiveClassifier':[{
            'clf__C':[.001, .01, .1, 1, 10],
            'clf__max_iter':[500, 1000, 2000],
            'clf__class_weight':[None, 'balanced']
        }],
        'RidgeClassifier':[{
            'clf__alpha':[1000, 100, 10, 1],
            'clf__class_weight':[None, 'balanced']
        }],
        'SGDClassifier':[{
            'clf__loss':['hinge', 'modified_huber', 'squared_hinge', 'perceptron'],
            'clf__l1_ratio':[0, .5, 1],
            'clf__learning_rate':['optimal'],
            'clf__max_iter':[1000, 5000],
            'clf__class_weight':[None, 'balanced']
        }],
        'KNeighborsClassifier':[{
            'clf__n_neighbors':[5, 10, 20],
            'clf__weights':['uniform', 'distance'],
            'clf__leaf_size':[15, 30, 60]
        }],
        'RandomForestClassifier':[{
            'clf__max_features':['auto', None],
            'clf__criterion':['gini'],
            'clf__n_estimators':[10, 100, 500, 1000],
            'clf__class_weight':[None, 'balanced']
        }],
        'XGBClassifier':[{
            'clf__max_depth':[3, 10, 30],
            'clf__n_estimators':[100, 500, 1000],
            'clf__learning_rate':[.01, .1, 1]
        },{
            'clf__grow_policy':['lossguide'],
            'clf__max_depth':[0],
            'clf__max_leaves':[1000],
            'clf__tree_method':['hist']
        }],
        'BaggingClassifier':[{
            'clf__n_estimators':[10, 100, 1000],
            'clf__max_samples':[1, 3, 9],
            'clf__max_features':[.25, .5, 1]
        }]
    }
    return grids

def get_classifiers():
    classifiers = [
        DummyClassifier(), LogisticRegression(), PassiveAggressiveClassifier(), RidgeClassifier(), SGDClassifier(), \
        KNeighborsClassifier(), MLPClassifier(), LinearSVC(), \
        NuSVC(), SVC(), DecisionTreeClassifier(), ExtraTreeClassifier(), AdaBoostClassifier(), \
        BaggingClassifier(), ExtraTreesClassifier(), GradientBoostingClassifier(), \
        RandomForestClassifier(), GaussianProcessClassifier(), \
        EasyEnsembleClassifier(), BalancedBaggingClassifier(), BalancedRandomForestClassifier(),
        XGBClassifier()
    ]
    return classifiers
    