#%%
import os, sys
sys.path.append('.')
sys.path.append('..')
from configs import project_root, mimic_root
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle as pk
#%%
pdf = pd.read_csv(f'{mimic_root}/hosp/patients.csv')
pdf
# %%
icddf = pd.read_csv(f'{project_root}/saved/hosp-diagnoses_icd.csv')
icddf
#%%
hadmdf = pd.read_csv(f'{mimic_root}/hosp/admissions.csv')
hadmdf
# %%
hicd10df = hadmdf.set_index('hadm_id')[['admittime', 'dischtime']].join(icddf.set_index('hadm_id'), how='right')
hicd10df
#%%
phicd10df = hicd10df.reset_index().set_index('subject_id').join(pdf.set_index('subject_id'), on='subject_id', how='left')
phicd10df = phicd10df[~phicd10df['icd10'].isna()]
phicd10df
#%%
phicd10df['admittime_dt'] = phicd10df['admittime'].map(lambda y: datetime.strptime(str(y), "%Y-%m-%d %H:%M:%S"))
phicd10df['anchor_year_dt'] = phicd10df['anchor_year'].map(lambda y: datetime.strptime(str(y), "%Y"))
phicd10df
#%%
phicd10df['tsince'] = (phicd10df['admittime_dt'] - phicd10df['anchor_year_dt']).dt.total_seconds()
# %%
plt.figure()
plt.hist([
    phicd10df[phicd10df['icd_version'] == 9]['tsince'],
    phicd10df[phicd10df['icd_version'] == 10]['tsince'],
])
plt.show()
# %%
# histdf = phicd10df[phicd10df['icd_version'] == 10].reset_index().sort_values('admittime_dt')[['subject_id', 'icd10', 'admittime_dt', 'hadm_id', 'icd_version']].groupby('subject_id').agg(list)
histdf = phicd10df.reset_index().sort_values('admittime_dt')[['subject_id', 'icd10', 'admittime_dt', 'hadm_id', 'icd_version']].groupby('subject_id').agg(list)
histdf
#%%
lsd = { d: histdf[d].values for d in ['icd10', 'admittime_dt', 'hadm_id', 'icd_version'] }
# %%
# early_diag = 1

ntoo_few = 0
nvalid = 0
nearly = 0
nhadms = 0
nmonth = 0
nyear = 0
next_visits = dict()
first_diagnosis = dict()
visits = dict()
diagnoses = dict(visit=dict())
when = dict(admit=dict())
for i in tqdm(range(len(lsd['hadm_id']))):
    patient = histdf.index[i]
    icds, adts, adids, vers = [np.array(lsd[k][i]) for k in ['icd10', 'admittime_dt', 'hadm_id', 'icd_version']]

    patient_first_diagnosis = dict()
    for c, hadm in zip(icds, adids):
        if c not in patient_first_diagnosis:
            patient_first_diagnosis[c] = hadm
    first_diagnosis[patient] = patient_first_diagnosis # important to determine when people convert

    for c, hadm, dt in zip(icds, adids, adts):
        when['admit'][hadm] = dt
        if hadm not in diagnoses['visit']: diagnoses['visit'][hadm] = dict()
        diagnoses['visit'][hadm][c] = True

    u_adids = list()
    for adid in adids:
        if adid not in u_adids: u_adids.append(adid)
    nhadms += len(u_adids)
    u_adts = np.array([adts[adids == i][0] for i in u_adids])
    u_adids = np.array(u_adids)
    visits[patient] = u_adids #list(zip(u_adids, u_adts))

    # too few obs for next visit bench
    if len(u_adids) < 2:
        ntoo_few += 1
        continue

    nvalid += 1

    for hadmi in range(len(u_adids) - 1):
        future_times = u_adts[hadmi+1:]
        future_hs = u_adids[hadmi+1:]
        current_hadm = u_adids[hadmi]
        t0 = adts[adids == current_hadm][0]
        next_visits[current_hadm] = dict(date=t0, patient=patient, month=[], year=[])

        # gathering all future visits in this range
        for hi in future_hs[[(t - t0).total_seconds() / 60 / 60 / 24 / 30 < 1 for t in future_times]]:
            next_visits[current_hadm]['month'] += [hi]
            nmonth += 1
        for hi in future_hs[[(t - t0).total_seconds() / 60 / 60 / 24 / 365 < 1 for t in future_times]]:
            next_visits[current_hadm]['year'] += [hi]
            nyear += 1

nvalid, nearly, nhadms, nmonth, nyear
# %%
with open(f'{project_root}/saved/blob.pk', 'wb') as fl:
    pk.dump(dict(
        next_visits=next_visits,
        first_diagnosis=first_diagnosis,
        visits=visits,
        diagnoses=diagnoses,
        when=when,
    ), fl)
# %%
