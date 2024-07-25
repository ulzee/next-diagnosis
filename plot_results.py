#%%
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
from sklearn.metrics import auc
#%%
models = [
    'linear_pNone',
    'xgb_d10_e100'
]
tcode = 'F329'
# %%
roc = []
aps = []
for mdl in models:
    with open(f'saved/scores/{tcode}/{mdl}/curves.pk', 'rb') as fl:
        tpr, fpr, pr, re = pk.load(fl)
        roc += [(tpr, fpr)]
        aps += [(pr, re)]

plt.figure(figsize=(8, 4))
plt.suptitle(tcode)
plt.subplot(1, 2, 1)
for mdl, (pr, re) in zip(models, aps):
    plt.plot(re, pr, label=f'{mdl} ({auc(re, pr):.2f})')
plt.legend()

plt.subplot(1, 2, 2)
for mdl, (tpr, fpr) in zip(models, roc):
    plt.plot(tpr, fpr, label=f'{mdl} ({auc(tpr, fpr):.2f})', zorder=1)
plt.axline((0,0), slope=1, color='lightgray', zorder=0)
plt.legend()

plt.tight_layout()
plt.savefig('images/bench.png', dpi=150)
plt.show()
# %%
