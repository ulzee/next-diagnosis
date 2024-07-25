#%%
%load_ext autoreload
%autoreload 2
#%%
import numpy as np
import helpers
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# %%
tte = helpers.TTE()
#%%
target_icd = 'F'
hist_samples, stats = tte.gather_samples(target_code=target_icd, censor_future=True)
#%%
len([h for h in hist_samples if h[2]])
# %%
mimic_total_patients = 299713
n_no_icd = mimic_total_patients - stats['patient']['total']
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      label = [
            'Total\n' + str(mimic_total_patients),
            'No ICD\n' + str(mimic_total_patients - stats['patient']['total']),
            'Has ICD\n' + str(stats['patient']['total']),
            f'One visit only\n' + str(stats['patient']['onevisit']),
            f'1+ visits\n' + str(stats['patient']['early'] + stats['patient']['considered']),
            f'Early ({target_icd})\n' + str(stats['patient']['early']),
            f'Considered ({target_icd})\n' + str(stats['patient']['considered']),
            f'Cases ({target_icd})\n' + str(stats['patient']['cases']),
            f'Controls ({target_icd})\n' + str(stats['patient']['controls']),
        ],
    ),
    link = dict(
      source = [0, 0, 2, 2, 4, 4, 6, 6],
      target = [1, 2, 3, 4, 5, 6, 7, 8],
      value = [
        n_no_icd,
        stats['patient']['total'],
        stats['patient']['onevisit'],
        stats['patient']['considered'] + stats['patient']['early'],
        stats['patient']['early'],
        stats['patient']['considered'],
        stats['patient']['cases'],
        stats['patient']['controls'],
    ]
  ))])

fig.update_layout(title_text="Patients", font_size=10)
fig.show()
#%%
target_icd = 'F329'
hist_samples_anytime, _ = tte.gather_samples(target_code=target_icd, censor_future=True, time_window=None, per_patient='first')
hist_samples_inyear_all, _ = tte.gather_samples(target_code=target_icd, censor_future=True, time_window='year', per_patient=None)
hist_samples_inyear_all_wrepeats, _ = tte.gather_samples(target_code=target_icd, censor_future=False, time_window='year', per_patient=None)
print('# of patients with diagnosis after first visit', len([h for h in hist_samples_anytime if h[2]]))
print('# visits where target diagnosis occurs within 1 year', len([h for h in hist_samples_inyear_all if h[2]]))
print('# visits where target diagnosis occurs within 1 year (& repeats)', len([h for h in hist_samples_inyear_all_wrepeats if h[2]]))
#%%
visit_times = []
for patient in tte.visits:
    iscase = None
    for h in tte.visits[patient]:
        if 'F329' in tte.diagnoses.visit[h]:
            iscase = h
            break
    if h == tte.visits[patient][0]: continue

    if iscase:
        t0 = tte.when.admit[iscase].timestamp()
        visit_times += [[(tte.when.admit[h].timestamp()-t0, 'F329' in tte.diagnoses.visit[h]) for h in tte.visits[patient]]]
len(visit_times)
#%%
visit_times_inorder = sorted(visit_times, key=lambda ls: ls[0][0])
plt.figure(figsize=(10, 4))
all_times = []
all_pixs = []
all_before_times = []
all_before_pixs = []
all_case_times = []
all_case_pixs = []
for pi, times in enumerate(visit_times_inorder):
    # if pi % 10 != 0: continue

    tafter = [t[0] for t in times if t[0] >= 0 and not t[1]]
    all_times += tafter
    all_pixs += [pi]*len(tafter)

    tbefore = [t[0] for t in times if t[0] < 0]
    all_before_times += tbefore
    all_before_pixs += [pi] * len(tbefore)

    tcase = [t[0] for t in times if t[1]]
    all_case_times += tcase
    all_case_pixs += [pi] * len(tcase)

plt.scatter(all_before_times, all_before_pixs, color='lightgray', s=3, label='visit (before)')
plt.scatter(all_times, all_pixs, color='black', s=3, label='visit (after)')
plt.scatter(all_case_times, all_case_pixs, color='C0', s=3, label='repeat diagnosis')
plt.axvline(0, color='red', label='first diagnosis')
xs = range(-10, 11, 2)
ts = [365*24*60*60* i for i in xs]
plt.xticks(ts, xs)
plt.yticks([], [])
plt.ylabel('Patients')
plt.xlabel('years')
plt.legend()
plt.savefig('images/convert.png', dpi=150)
plt.show()
#%%
