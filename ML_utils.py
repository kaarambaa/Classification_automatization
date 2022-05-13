# MIT License
#
# Copyright (c) 2022 Konstantin Kovalev
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from sklearn import ensemble
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn import model_selection
from sklearn import metrics
from Container_utils import *
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
import numpy as np
import pandas as pd
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uni
class models ():

    def __init__(self, algList_ = [], tune_list_ = [], container_ = {}):
        self.algList = algList_
        self.tuneList = tune_list_
        self.container = container_
        self.init = {'RF': ensemble.RandomForestClassifier(), 'GB': ensemble.GradientBoostingClassifier(),
                     'LGBM': LGBMClassifier(),
                     'XGB': XGBClassifier(), 'AB': ensemble.AdaBoostClassifier()}
        self.trainedModels = []
        self.predictions = []
        self.metrics = {}

    def fit(self):
        self.trainedModels = []
        x_train = getFile(self.container, 'x_train')
        y_train = getFile(self.container, 'y_train')
        algs = []
        for a in self.algList:
            if a in self.init.keys():
                algs.append(self.init[a])

        for alg in algs:
            alg.fit(x_train, y_train)
            self.trainedModels.append(alg)
            alg_name = alg.__class__.__name__
            addFile(self.container,f'{alg_name}/model',alg)
        return self.trainedModels

    def predict(self):
        self.predictions = []
        self.pred_probas = []
        x_test = getFile(self.container,'x_test')
        self.fit()
        for alg in self.trainedModels:
            pred = alg.predict(x_test)
            probas = alg.predict_proba(x_test)
            self.predictions.append(pred)
            self.pred_probas.append(probas)
            alg_name = alg.__class__.__name__
            addFile(self.container, f'{alg_name}/preds',pred)
            addFile(self.container, f'{alg_name}/probas', probas)


    def eval(self):
        y_test = getFile(self.container, 'y_test')
        self.metrics = {}

        for i in range(len(self.predictions)):
            qual = {"Accuracy" : metrics.accuracy_score(y_test,self.predictions[i]),
                    "Precision" : metrics.precision_score(y_test,self.predictions[i]),
                    'Recall' : metrics.recall_score(y_test,self.predictions[i]),
                    'F1' : metrics.f1_score(y_test,self.predictions[i]),
                    'ROC_AUC score': metrics.roc_auc_score(y_test,self.pred_probas[i].T[1]),
                    'ROC_AUC': metrics.roc_curve(y_test,self.pred_probas[i].T[1]),
                    'Conf': metrics.confusion_matrix(y_test,self.predictions[i])}
            self.metrics[self.algList[i]] = qual

    def pipeline(self):
        self.fit()
        self.predict()
        self.eval()
        return self.metrics

    def crossVal(self, resemp, n, mode):
        tune = []
        X = getFile(self.container, 'X')
        Y = getFile(self.container, 'Y')

        if resemp == "Oversampling":
            over = RandomOverSampler(sampling_strategy='minority', random_state=101)
            step = [('over', over)]
        elif resemp == "Oversampling":
            under = RandomOverSampler(sampling_strategy='majority', random_state=101)
            step = [('under', under)]
        elif resemp == "Over + Under":
            over = RandomOverSampler(sampling_strategy=0.5, random_state=101)
            under = RandomUnderSampler(sampling_strategy=1, random_state=101)
            step = [('over', over), ('under', under)]
        elif resemp == "SMOTE":
            over = SMOTE(sampling_strategy='minority', random_state=101)
            step = [('over', over)]
        elif resemp == "NearMiss":
            under = NearMiss(sampling_strategy='majority', random_state=101)
            step = [('under', under)]
        else:
            step = []
        algs = []
        for a in self.algList:
            print(a)
            if a in self.init.keys():
                algs.append(self.init[a])
        strcv = model_selection.StratifiedKFold(n_splits=n)

        if len(step) > 0:
            pip = Pipeline(step)
            X_res, y_res = pip.fit_resample(X, Y)

        else:
            X_res = X
            y_res = Y
        if mode == 1:
            row_index = 0
            MLA_columns = ['Model', 'Train F1 score', 'Test F1 score', 'Time consumed']
            MLA_compare = pd.DataFrame(columns=MLA_columns)
            for alg in algs:
                # s = step.copy()
                # s.append(('model', alg))
                # pipeline = Pipeline(s)
                MLA_name = alg.__class__.__name__
                MLA_compare.loc[row_index, 'Model'] = MLA_name
                cv_results = model_selection.cross_validate(alg, X_res, y_res, cv=strcv, scoring='f1',
                                                            return_train_score=True)
                MLA_compare.loc[row_index, 'Time consumed'] = np.round(cv_results['fit_time'].mean(), 4)
                MLA_compare.loc[row_index, 'Train F1 score'] = np.round(cv_results['train_score'].mean(), 4)
                MLA_compare.loc[row_index, 'Test F1 score'] = np.round(cv_results['test_score'].mean(), 4)
                row_index += 1
        if mode == 2:

            tune = []
            row_index = 0
            for a in self.tuneList:
                if a in self.init.keys():
                    tune.append(self.init[a])
            param_grids = [self.paramsRF(),self.paramsGB(),self.paramsAB(),self.paramsXGB(),self.paramsLGB()]
            MLA_columns = ['Model', 'Best F1 score', 'Best params']
            MLA_compare = pd.DataFrame(columns=MLA_columns)
            for alg in tune:
                # s = step.copy()
                # s.append(('model', alg))
                # pipeline = Pipeline(s)
                MLA_name = alg.__class__.__name__
                MLA_compare.loc[row_index, 'Model'] = MLA_name
                randomgrid_res = model_selection.RandomizedSearchCV(estimator=alg, param_distributions=param_grids[row_index], cv=strcv, scoring='f1')
                randomgrid_res.fit(X_res,y_res)
                MLA_compare.loc[row_index, 'Best F1 score'] = np.round(randomgrid_res.best_score_, 4)
                MLA_compare.loc[row_index, 'Best params'] = str(randomgrid_res.best_params_)
                row_index += 1
        return MLA_compare



    def paramsRF(self):
        n_estimators = [int(x) for x in np.linspace(start=50, stop=500, num=10)]
        max_features = ['auto', 'sqrt']
        max_depth = [2, 4]
        min_samples_split = [2, 5]
        min_samples_leaf = [1, 2]
        bootstrap = [True, False]
        param_grid = {'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'bootstrap': bootstrap}

        return param_grid

    def paramsGB(self):
        learning_rate = np.logspace(-5, 1)
        n_estimate_dist = sp_randint(50, 500)
        max_depth_dist = sp_randint(1, 10)
        param_grid = {'learning_rate':learning_rate,
                          'n_estimators':n_estimate_dist,
                          'max_depth':max_depth_dist}

        return param_grid

    def paramsAB(self):
        n_estimators = sp_randint(50, 500)
        learning_rate = np.logspace(-5, 1)

        param_grid = {'n_estimators': n_estimators ,
                            'learning_rate':learning_rate}

        return param_grid

    def paramsXGB(self):
        param_grid = {'n_estimators': sp_randint(50,500),
                      'learning_rate': np.logspace(-5, 1),
                      'subsample': sp_uni(0.3, 0.6),
                      'max_depth': [3, 4, 5, 6, 7, 8, 9],
                      'colsample_bytree': sp_uni(0.5, 0.4),
                      'min_child_weight': [1, 2, 3, 4],
                      'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                      'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]
                      }

        return param_grid

    def paramsLGB(self):
        param_grid = {'num_leaves': sp_randint(6, 50),
                      'min_child_samples': sp_randint(100, 500),
                      'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],
                      'subsample': sp_uni(loc=0.2, scale=0.8),
                      'colsample_bytree': sp_uni(loc=0.4, scale=0.6),
                      'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],
                      'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

        return param_grid








