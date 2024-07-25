#%%
import os, sys
import matplotlib.pyplot as plt
import helpers
from sklearn.linear_model import LogisticRegression
import argparse
import numpy as np
import pickle as pk
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
#%%
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='linear')
parser.add_argument('--penalty', type=str, default=None)
parser.add_argument('--predict_split', type=str, default='test')
parser.add_argument('--bootstrap', type=int, default=10)
args = parser.parse_args()
#%%
baseline_tag = f'linear_p{args.penalty}'
if not os.path.exists(f'saved/{baseline_tag}'):
    os.mkdir(f'saved/{baseline_tag}')
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
target_code = 'F329'
all_samples, _ = tte.gather_samples(target_code=target_code, time_window='year', censor_future=True, use_history=True)
len(all_samples)
#%%
print('# cases   :', len([h for h in all_samples if h[-1]]))
print('# controls:', len([h for h in all_samples if not h[-1]]))
#%%
datamats = helpers.generate_data_splits(tte, all_samples, vocab)
#%%
# LR
lreg = LogisticRegression(random_state=0, penalty=args.penalty).fit(*datamats['train'])
#%%
with open(f'saved/{baseline_tag}/model.pk', 'wb') as fl:
    pk.dump(lreg, fl)
#%%
ypred = lreg.predict_proba(datamats['test'][0])[:, 1]
ytarg = datamats['test'][1]
#%%
bixs = np.load('artifacts/splits/boots.npy')
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
)).to_csv(f'saved/{baseline_tag}/scores_{args.predict_split}_boot{args.bootstrap}.csv', index=False)
# %%
