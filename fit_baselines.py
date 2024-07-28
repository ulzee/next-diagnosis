#%%
import os, sys
import matplotlib.pyplot as plt
import helpers
from sklearn.linear_model import LogisticRegression
import argparse
import numpy as np
import pickle as pk
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, roc_curve, precision_recall_curve
from xgboost import XGBClassifier
from hyperopt import hp, Trials, fmin, tpe, STATUS_OK
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='linear')
parser.add_argument('--code', type=str, required=True)
parser.add_argument('--penalty', type=str, default=None)
parser.add_argument('--predict_split', type=str, default='test')
parser.add_argument('--bootstrap', type=int, default=10)
parser.add_argument('--max_depth', type=int, default=None)
parser.add_argument('--n_estimators', type=int, default=None)
parser.add_argument('--use_embedding', type=str, default=None)
parser.add_argument('--embedding_agg', type=str, default='sum')
parser.add_argument('--covariates_only', action='store_true', default=False)
parser.add_argument('--no_covariates', action='store_true', default=False)
args = parser.parse_args()
#%%
if args.model == 'linear':
    baseline_tag = f'linear_p{args.penalty}'
elif args.model == 'xgb':
    baseline_tag = f'xgb'
    if args.max_depth is not None:
        baseline_tag += f'_d{args.max_depth}'
    if args.n_estimators is not None:
        baseline_tag += f'_e{args.n_estimators}'
if args.use_embedding is not None:
    embname = args.use_embedding.split('/')[-1].split('.')[0]
    baseline_tag += f'_emb-{embname}'
    baseline_tag += f'_agg{args.embedding_agg}'
print(baseline_tag)

if not os.path.exists(f'saved/scores'):
    os.mkdir(f'saved/scores')
if not os.path.exists(f'saved/scores/{args.code}'):
    os.mkdir(f'saved/scores/{args.code}')
if not os.path.exists(f'saved/scores/{args.code}/{baseline_tag}'):
    os.mkdir(f'saved/scores/{args.code}/{baseline_tag}')
#%% #[:4]
tte = helpers.TTE()
#%%
vocab = dict()
for codes in tte.diagnoses.visit.values():
    for c in codes:
        if c == 'NoDx': continue
        if c not in vocab:
            vocab[c] = len(vocab)
len(vocab)
#%%
all_samples, _ = tte.gather_samples(target_code=args.code, time_window='year', censor_future=True, use_history=True)
len(all_samples)
#%%
print('# cases   :', len([h for h in all_samples if h[-1]]))
print('# controls:', len([h for h in all_samples if not h[-1]]))
#%%
datamats = helpers.generate_data_splits(
    tte, all_samples, vocab,
    covariates=[] if args.no_covariates else ['age', 'sex'],
    covariates_only=args.covariates_only
)
#%%
if args.use_embedding:
    with open(args.use_embedding, 'rb') as fl:
        embdict = pk.load(fl)
    for k, v in list(embdict.items()):
        while len(k) > 3:
            k = k[:len(k)-1]
            if k not in embdict:
                embdict[k] = v

    agg_fn = dict(
        sum=lambda ls: np.sum(ls, 0),
        mean=lambda ls: np.mean(ls, 0)
    )[args.embedding_agg]
    embmats = helpers.get_visit_history_embedding(
        tte, all_samples, vocab, embdict,
        aggregation=agg_fn)

    for phase, (X, y) in datamats.items():
        datamats[phase][0] = np.concatenate([X, embmats[phase]], axis=1)
#%%
print('# of features:', datamats['train'][0].shape[1])
#%%
np.random.seed(0)
if args.model == 'linear':
    mdl = LogisticRegression(random_state=0, penalty=args.penalty).fit(*datamats['train'])
elif args.model == 'xgb':
    if args.max_depth is not None and args.n_estimators is not None:
        print('Running XGB with given settings')
        mdl = XGBClassifier(
            max_depth=args.max_depth,
            n_estimators=args.n_estimators
        ).fit(*datamats['train'])
    else:
        print('Tuning hyperparams...')

        space = {
            'max_depth': hp.quniform("max_depth", 2, 10, 1),
            'n_estimators': hp.quniform("n_estimators", 10, 100, 1),
        }

        def objective(space):
            clf = XGBClassifier(
                n_estimators = int(space['n_estimators']),
                max_depth = int(space['max_depth']),
            )
            clf.fit(
                *datamats['train'],
                eval_set=[datamats['val']],
                verbose=False)

            ypred = clf.predict_proba(datamats['val'][0])[:, 1]
            ytarg = datamats['val'][1]
            ap = average_precision_score(ytarg, ypred)
            roc = roc_auc_score(ytarg, ypred)
            final_loss = clf.evals_result()['validation_0']['logloss'][-1]

            print(f'{final_loss:.4f} {ap:.4f} {roc:.4}')

            return { 'loss': final_loss, 'status': STATUS_OK }

        trials = Trials()

        best_hyperparams = fmin(
            fn = objective,
            space = space,
            algo = tpe.suggest,
            max_evals = 10,
            trials = trials,
            rstate=np.random.default_rng(0))

        print('Best:', best_hyperparams)
        with open(f'saved/scores/{args.code}/{baseline_tag}/best_hp.pk', 'wb') as fl:
            pk.dump(best_hyperparams, fl)

        mdl = XGBClassifier(
            max_depth=int(best_hyperparams['max_depth']),
            n_estimators=int(best_hyperparams['n_estimators'])
        ).fit(*datamats['train'])
#%%
with open(f'saved/scores/{args.code}/{baseline_tag}/model.pk', 'wb') as fl:
    pk.dump(mdl, fl)
#%%
ypred = mdl.predict_proba(datamats[args.predict_split][0])[:, 1]
ytarg = datamats[args.predict_split][1]
#%%
np.random.seed(0)
bixs = [np.random.choice(len(ypred), size=len(ypred), replace=True) for _ in range(args.bootstrap)]
metrics = [
    average_precision_score,
    roc_auc_score,
    f1_score,
]

results = []
for bi in range(args.bootstrap):
    results += [[m(ytarg[bixs[bi]], ypred[bixs[bi]] if m != f1_score else ypred[bixs[bi]] > 0.5) for m in metrics]]
ap_est, roc_est, f1_est = [np.mean(ls) for ls in zip(*results)]
ap_std, roc_std, f1_std = [np.std(ls) for ls in zip(*results)]
print(f'PR: {ap_est:.3f} ({1.95*ap_std:.3f}), ROC: {roc_est:.3f} ({1.95*roc_std:.3f}), F1: {f1_est:.3f} ({1.95*f1_std:.3f})')
#%%
pd.DataFrame(dict(
    metrics=['pr', 'roc', 'f1'],
    ests=[ap_est, roc_est, f1_est],
    stds=[ap_std, roc_std, f1_std]
)).to_csv(f'saved/scores/{args.code}/{baseline_tag}/scores_{args.predict_split}_boot{args.bootstrap}.csv', index=False)
# %%
tpr, fpr, _ = roc_curve(ytarg, ypred)
pr, re, _ = precision_recall_curve(ytarg, ypred)
with open(f'saved/scores/{args.code}/{baseline_tag}/curves.pk', 'wb') as fl:
    pk.dump([tpr, fpr, pr, re], fl)