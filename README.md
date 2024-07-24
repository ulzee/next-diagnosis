# Next visit diagnosis benchmark

## Getting started

Create a python file `config.py` with the following variables to tell the benchmark where to find the EHR datasets.
```python
mimic_root = '/media/ulzee/data/ukbb/gpt/data/mimic4/mimiciv/2.2'
project_root = '/home/ulzee/gpt/next_visit'
```

### Basic dependancies

```bash
numpy
pandas
pickle
```

### Advanced dependancies (WIP)

- [ ] TODO: script to download and process these automatically

#### MedBERT (WIP)

Navigate to `saved/medbert` and clone the codebase found at:
[https://github.com/kirilklein/Med-BERT.git](https://github.com/kirilklein/Med-BERT.git) to run medbert-based benchmarks such that the code is found in `saved/medbert/Med-BERT`.

After fitting Med-BERT, it will save the weights and the model config in a log folder. Copy or symlink these files such that they can be found in:

```bash
ls saved/medbert
Med-BERT
checkpoint_epoch99_end.pt
config.py
pretrain_mimmiciv.yaml
```

- [ ] TODO: copying or symlink is not necessary, our code can access them directly

## Breakdown of patients who would be considered a case or a control

The data processing will identify patients who have been diagnosed with ICD codes,
and those who have multiple visits.
Among this group, patients who already have the prediction target diagnosis will be ignored as the next diagnosis benchmark is not possible for them.
Therefore, cases and controls are guaranteed to have at least 2 visits.

![patients](https://github.com/ulzee/next-diagnosis/blob/master/images/patients.png?raw=true)