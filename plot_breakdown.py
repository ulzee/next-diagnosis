#%%
%load_ext autoreload
%autoreload 2
#%%
import helpers
import plotly.graph_objects as go
import matplotlib.pyplot as plt
# %%
tte = helpers.TTE()
#%%
target_icd = 'F'
hist_samples, stats = tte.gather_samples(chapter=target_icd, censor_future=True)
#%%
len([h for h in hist_samples if h[1]])
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
# %%
hist_samples_repeat, _ = tte.gather_samples(chapter=target_icd, censor_future=False)
# %%
len([h for h in hist_samples_repeat if h[1]])
#%%
currid = hist_samples[0][0][0]
ccount = 0
max_id = (None, 0, None)
prev_hist = None
for hist, iscase in hist_samples:
    if currid != hist[0]:
        currid = hist[0]
        if ccount > max_id[1]:
            max_id = (currid, ccount, prev_hist)
        if ccount >= 10:
            break
        ccount = 0
    else:
        if iscase:
            ccount += 1
    prev_hist = hist
max_id
#%%
bycode = dict()
all_visit_times = []
for hid in max_id[2]:
    visit_info = tte.next_visits[hid]
    t = visit_info['date'].timestamp()
    for code in tte.diagnoses.visit[hid]:
        if 'F' != code[0]: continue
        if code not in bycode: bycode[code] = list()
        bycode[code] += [t]
    all_visit_times += [visit_info['date']]

plt.figure(figsize=(10, 4))
seen_codes = sorted(bycode.keys(), key=lambda s: int(s[1:]))
for ci, code in enumerate(seen_codes):
    if len(bycode[code]):
        ts = bycode[code]
        plt.scatter(ts, [ci]*len(ts), zorder=1)
plt.yticks(range(len(seen_codes)), seen_codes)
for t in all_visit_times:
    plt.axvline(t.timestamp(), color='lightgray', zorder=0, alpha=0.5)
for ci in range(len(seen_codes)):
    plt.axhline(ci, color='lightgray', zorder=0, alpha=0.5)
plt.xticks(
    [all_visit_times[0].timestamp(), all_visit_times[-1].timestamp()],
    [all_visit_times[0].date(), all_visit_times[-1].date()]
)
plt.show()
# %%
