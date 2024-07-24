#%%
%load_ext autoreload
%autoreload 2
#%%
import helpers
import plotly.graph_objects as go
# %%
tte = helpers.TTE()
#%%
target_icd = 'F'
hist_samples, stats = tte.gather_samples(chapter=target_icd)
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
