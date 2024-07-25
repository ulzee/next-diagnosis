# Next visit first-time diagnosis benchmark

## A lightweight benchmark for predicting future assignments of diagnosis codes based on previous diagnoses

### Patients who would be considered in the benchmark

The benchmark will identify patients who have been diagnosed with ICD codes in the MIMIC-IV dataset.
Among this group, patients who already have the target diagnosis on their first visit will be ignored (the presumption is that they have likely had the condition for a while).
Patients who also only have 1 visit cannot be assessed for the next-visit setting.
Patients in cases and controls therefore have at least 2 visits.
As shown below, the actual number of patients considered in the benchmark can be a small fraction of all the data available.
This figure can be reproduced using `plot_breakdown.py`.

![patients](https://github.com/ulzee/next-diagnosis/blob/master/images/patients.png?raw=true)

<!-- ### How visits are selected for the benchmark
Among patients who are considered, -->

### First time diagnoses and repeat diagnoses

Successfully identifying patients before they convert to a disease group for the first time may open up more avenues of treatment in real world medical settings.
Therefore, the proposed benchmark by default focuses on first-time diagnoses, as opposed to considering diagnoses at any point in a patient's health records history.
Intuitively, the medical condition of a patient may notably change after they receive the diagnosis of interest, especially for life-altering conditions.
One way this can be observed is the frequency of similar disease codes a patient recieves after they receive the first one.
For some disease, repeat diagnoses of the same patient may consist the majority of all such diagnoses across all visits.

![patients](https://github.com/ulzee/next-diagnosis/blob/master/images/convert.png?raw=true)

It follows that predicting when a patient will receive a diagnosis for the first time in their history is much harder than predicting repeat diagnoses.
There are also fewer such cases (a patient can only be diagnosed for the first time once in their medical history) as opposed to considering all visits ever.

The benchmark settings can be changed easily such that it considers either __first or first+repeat__ diagnoses to explore both options.

## Getting started

The benchmark has been developed with the __MIMIC-IV (2.2)__ dataset in mind. The dataset can be downloaded from `physionet.org` at no cost after an approval process.

Create a python file `config.py` with the following variables to tell the benchmark where to find the EHR datasets.
```python
mimic_root = '/media/ulzee/data/ukbb/gpt/data/mimic4/mimiciv/2.2'
project_root = '/home/ulzee/gpt/next_visit'
```

### Basic dependancies

Python >=3.10 is required.

```bash
numpy
pandas
pickle
icd-mappings
xgboost
hyperopt
plotly
matplotlib
```

### Advanced dependancies (WIP)

- [ ] This part is not ready, don't try to run it right now
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

## Running the benchmark

- [ ] Warning, you may run into errors, lmk if you do

### 1. Create a local copy of diagnoses where all ICD9 are converted to ICD10

```bash
python scripts/convert_icd10.py
```

### 2. Check all visits and cache any subsequent visits that are within the benchmark time frame

- [ ] NOTE: currently next visit ranges of 1 month and 1 year are supported
```bash
python scripts/create_next_visit_dataset.py
```

### 3. Run baselines

At this point, it is possible to run a few baseline methods for the next visit diagnosis task.

Some examples:

```bash
python fit_baselines.py --model linear
python fit_baselines.py --model xgb --max_depth 10 --n_estimators 100
```

![patients](https://github.com/ulzee/next-diagnosis/blob/master/images/bench.png?raw=true)