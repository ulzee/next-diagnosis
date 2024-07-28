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
from time import time
#%%
pdf = pd.read_csv(f'{mimic_root}/hosp/patients.csv')
pdf['anchor_year_dt'] = pdf['anchor_year'].map(lambda y: datetime.strptime(str(y), "%Y"))
pdf
# %%
icddf = pd.read_csv(f'{project_root}/saved/hosp-diagnoses_icd.csv')
icddf
#%%
hadmdf = pd.read_csv(f'{mimic_root}/hosp/admissions.csv')
hadmdf['admittime_dt'] = hadmdf['admittime'].map(lambda y: datetime.strptime(str(y), "%Y-%m-%d %H:%M:%S"))
hadmdf
#%%
hpdf = hadmdf.set_index('subject_id').join(pdf.set_index('subject_id'), how='left')
hpdf
# %%
hicd10df = hadmdf.set_index('hadm_id')[['admittime', 'admittime_dt', 'dischtime']].join(icddf.set_index('hadm_id'), how='right')
hicd10df
#%%
phicd10df = hicd10df.reset_index().set_index('subject_id').join(pdf.set_index('subject_id'), on='subject_id', how='left')
phicd10df = phicd10df[~phicd10df['icd10'].isna()]
phicd10df
#%%
phicd10df['tsince'] = (phicd10df['admittime_dt'] - phicd10df['anchor_year_dt']).dt.total_seconds()
#%%
patients_info = dict()
for pid, vid, gender in zip(hpdf.index, hpdf['hadm_id'], hpdf['gender']):
    if pid not in patients_info:
        patients_info[pid] = dict()
        patients_info[pid]['sex'] = gender
        patients_info[pid]['visits'] = list()

    # if vid not in patients_info[pid]['visits']:
    #     patients_info[pid]['visits'] += [vid]
#%%
visits_info = dict()
for pid, vid, admt, ancht, anchage in tqdm(zip(
    *([hpdf.index] + [hpdf[c] for c in [
        'hadm_id', 'admittime_dt', 'anchor_year_dt', 'anchor_age']]))):

    if vid in visits_info: continue

    since_born = (admt - (ancht - pd.DateOffset(years=anchage))).total_seconds()
    approx_age = since_born/60/60/24/365.26
    visits_info[vid] = dict(
        patient=pid,
        admittime=admt.to_datetime64(),
        age=approx_age,
    )
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
ntoo_few = 0
nvalid = 0
nearly = 0
nhadms = 0
nmonth = 0
nyear = 0
visits = dict() # Deprecated
next_visits = dict()
first_diagnosis = dict()
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
    # for id in u_adids:
    #     visits_info[id]['benchmark'] = True

    # too few obs for next visit bench
    if len(u_adids) < 2:
        ntoo_few += 1
        continue

    visits[patient] = u_adids[:-1]
    nvalid += 1

    for hadmi in range(len(u_adids) - 1):
        future_times = u_adts[hadmi+1:]
        future_hs = u_adids[hadmi+1:]
        current_hadm = u_adids[hadmi]
        t0 = adts[adids == current_hadm][0]
        next_visits[current_hadm] = dict(date=t0, patient=patient, month=[], year=[])

        # gathering all future visits in predetermined ranges
        for hi in future_hs[[(t - t0).total_seconds() / 60 / 60 / 24 / 30 < 1 for t in future_times]]:
            next_visits[current_hadm]['month'] += [hi]
            nmonth += 1
        for hi in future_hs[[(t - t0).total_seconds() / 60 / 60 / 24 / 365 < 1 for t in future_times]]:
            next_visits[current_hadm]['year'] += [hi]
            nyear += 1

        patients_info[patient]['visits'] += [current_hadm]

nvalid, nearly, nhadms, nmonth, nyear
# %%
for pid in visits:
    assert pid in patients_info
    # for vid in patients_info[pid]['visits']:
    assert len(patients_info[pid]['visits']) == len(visits[pid])
# %%
t0 = time()
with open(f'{project_root}/saved/blob.pk', 'wb') as fl:
    pk.dump(dict(
        next_visits=next_visits,
        first_diagnosis=first_diagnosis,
        diagnoses=diagnoses,
        # when=when,

        patients=patients_info,
        visits=visits_info,
        # visits=visits,
    ), fl)
print(time() - t0)
