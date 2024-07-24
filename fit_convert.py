#%%
%load_ext autoreload
%autoreload 2
#%%
import os, sys
import matplotlib.pyplot as plt
import helpers
#%%
tte = helpers.TTE()
#%%
use_emb = None
use_medbert = True
embdict = None
if use_emb:
    with open(f'{project_root}/../data/icd10/embeddings/{use_emb}.pk', 'rb') as fl:
        embdict = pk.load(fl)
#%%
poshist = [len(h) for h, iscase in hsamples if iscase]
neghist = [len(h) for h, iscase in hsamples if not iscase]
plt.figure()
plt.hist([poshist, neghist], bins=20)
plt.ylim(0, 3000)
plt.show()
len(poshist)/len(neghist)
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

if use_medbert:
    emb_model = helpers.Inference.MedBERT()

    for hls, iscase in tqdm(hsamples):
        batch = dict(concept=[], age=[], abspos=[], segment=[], los=[])
        [batch[prop].append(list()) for prop in batch]
        for hi, h in enumerate(hls):
            diags_in_visit = tte.diagnoses.visit[h]
            batch['concept'][-1] += list(diags_in_visit.keys())
            batch['age'][-1] += [60]*len(diags_in_visit)
            batch['abspos'][-1] += [0]*len(diags_in_visit)
            batch['segment'][-1] += [hi]*len(diags_in_visit)
            batch['los'][-1] += [1]*len(diags_in_visit)

        emb_model(**batch)

        break
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
