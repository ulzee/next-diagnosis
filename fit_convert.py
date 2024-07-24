#%%
import os, sys
sys.path.append('.')
sys.path.append('..')
from configs import project_root, mimic_root
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pickle as pk
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool, metrics, cv
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import pickle as pk
from tqdm import tqdm
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import os, sys
from time import time
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.linear_model import LogisticRegression
#%%
class TTE:

    class diagnoses:
        pass

    def __init__(self):
        with open(f'{project_root}/saved/blob.pk', 'rb') as fl:
            blob = pk.load(fl)
        self.first_diagnosis = blob['first_diagnosis']
        self.visits = blob['visits']
        self.next_visits = blob['next_visits']

        self.diagnoses = TTE.diagnoses()
        self.diagnoses.visit = blob['diagnoses']['visit']

tte = TTE()
#%%
chapter = 'F'
early_diag = 1
time_window = 'month'

censor_future = True
use_history = True
# use_emb = 'kane/biogpt100'
use_emb = None
nearly = 0
nconsidered = 0
nhadm = 0
nmonth = 0

embdict = None
if use_emb:
    with open(f'{project_root}/../data/icd10/embeddings/{use_emb}.pk', 'rb') as fl:
        embdict = pk.load(fl)

visit_has_code = lambda v, match: any([c[:len(match)] == match for c in tte.diagnoses.visit[v]])

hsamples = []
for patient in tqdm(tte.visits.keys()):

    early_case = False
    for code, h in tte.first_diagnosis[patient].items():
        if chapter == code[0] and h == tte.visits[patient][0]:
            early_case = True
    if early_case:
        nearly += 1
        continue

    nconsidered += 1

    running_hist = []
    for h in tte.visits[patient][early_diag:-1]:

        nhadm += 1
        nmonth += len(tte.next_visits[h][time_window])

        if censor_future:
            # NOTE: current visit should not have the target diagnosis
            #  therefore, after a control converts to a case, don't keep scanning
            if visit_has_code(h, chapter):
                break

        iscase = False
        for hnxt in tte.next_visits[h][time_window]:
            if visit_has_code(hnxt, chapter):
                iscase = True

        running_hist += [h]
        hsamples += [(list.copy(running_hist) if use_history else [h], iscase)]

len(tte.visits), nearly, nconsidered, nhadm, nmonth, len(hsamples), len([s for s in hsamples if s[1]])
# %%
format_code = lambda c: c #[:4]
vocab = dict()
for codes in tte.diagnoses.visit.values():
    for c in codes:
        if c == 'NoDx': continue
        c = format_code(c)
        if c not in vocab:
            vocab[c] = len(vocab)
len(vocab)
#%%
# len([v for v in vocab if v not in embdict])
#%%
Xrows = []
yrows = []
embrows = []
for hls, iscase in tqdm(hsamples):
    sample_embs = []

    row = np.zeros(len(vocab))
    for h in hls:
        for c in tte.diagnoses.visit[h]:
            match_c = format_code(c)
            if match_c in vocab:
                row[vocab[match_c]] = 1
                if use_emb:
                    if c in embdict:
                        sample_embs += [embdict[c]]
    Xrows += [row]
    yrows += [iscase]

    if use_emb:
        embavg = np.mean(sample_embs, 0) if len(sample_embs) else np.zeros(len(embavg))
        embrows += [embavg]

X = np.array(Xrows)
if embdict:
    embrows = np.array(embrows)
    X = np.concatenate([X, embrows], 1)
y = np.array(yrows)
X.shape, y.shape
#%%
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.75, random_state=0)
#%%
# LR
lreg = LogisticRegression(random_state=0, penalty=None).fit(X_train, y_train)
ypred = lreg.predict_proba(X_val)[:, 1]
#%%
ap = average_precision_score(y_val, ypred)
roc = roc_auc_score(y_val, ypred)
print(ap, roc)
#%%
pr, re, _ = precision_recall_curve(y_val, ypred)
tpr, fpr, _ = roc_curve(y_val, ypred)
saved[('lr', censor_future, use_history, use_emb)] = (pr, re, tpr, fpr)
#%%
# XGB
# space = {
#     'max_depth': hp.quniform("max_depth", 3, 18, 1),
#     'gamma': hp.uniform ('gamma', 1,9),
#     'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
#     'reg_lambda' : hp.uniform('reg_lambda', 0,1),
#     'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
#     'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
#     'n_estimators': 180,
#     'seed': 0
# }

# def objective(space):
#     clf = XGBClassifier(
#         n_estimators = space['n_estimators'],
#         max_depth = int(space['max_depth']),
#         gamma = space['gamma'],
#         reg_alpha = int(space['reg_alpha']),
#         min_child_weight=int(space['min_child_weight']),
#         colsample_bytree=int(space['colsample_bytree']),
#         early_stopping_rounds=10
#     )

#     evaluation = [( X_val, y_val)]
#     clf.fit(
#         X_train, y_train,
#         eval_set=evaluation,

#         verbose=False)

#     # print(clf.evals_result())
#     # return None
#     return { 'loss': clf.evals_result()['validation_0']['logloss'][-1], 'status': STATUS_OK }

# trials = Trials()

# best_hyperparams = fmin(
#     fn = objective,
#     space = space,
#     algo = tpe.suggest,
#     max_evals = 100,
#     trials = trials)
#%%
# best_hyperparams
xgb = XGBClassifier(
    max_depth=10,
    n_estimators=100,
    # learning_rate=0.1,
    # early_stopping_rounds=10
)

# evaluation = [( X_val, y_val)]
xgb.fit(
    X_train, y_train,
    # eval_set=evaluation,
    verbose=True)
#%%
ypred = xgb.predict_proba(X_val)[:, 1]

ap = average_precision_score(y_val, ypred)
roc = roc_auc_score(y_val, ypred)
print(ap, roc)

pr, re, _ = precision_recall_curve(y_val, ypred)
tpr, fpr, _ = roc_curve(y_val, ypred)
saved[('xgb', censor_future, use_history, use_emb)] = (pr, re, tpr, fpr)
# %%
#%%
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
for (mdl, _future, __history, __emb), (pr, re, tpr, fpr) in saved.items():
    plt.plot(re, pr, label=f'({not _future}) futr, ({__history}) hist ({__emb}) emb')
plt.axhline(y.sum()/len(y), color='lightgray')
plt.xlim(0, 1)
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
for (mdl, _future, __history, __emb), (pr, re, tpr, fpr) in saved.items():
    plt.plot(tpr, fpr, label=f'({not _future}) futr, ({__history}) hist ({__emb}) emb')
plt.axline((0,0), slope=1, color='lightgray')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.legend()
plt.show()
# %%
