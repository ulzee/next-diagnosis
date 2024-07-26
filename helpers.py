
from os.path import join
import sys
from configs import project_root
import pickle as pk
from tqdm import tqdm
import numpy as np

class DotDict(dict):
    def __getattr__(self, attr):
        return self[attr]

class TTE:

    class diagnoses:
        pass

    class when:
        pass

    def __init__(self):
        with open(f'{project_root}/saved/blob.pk', 'rb') as fl:
            blob = pk.load(fl)
        self.first_diagnosis = blob['first_diagnosis']
        self.visits = blob['visits']
        self.next_visits = blob['next_visits']

        self.diagnoses = TTE.diagnoses()
        self.diagnoses.visit = blob['diagnoses']['visit']
        self.when.admit = blob['when']['admit']

    def gather_samples(self,
        target_code=None,
        time_window='year', # set None to set indefinite time window
        censor_future=True,
        use_history=True,
        per_patient=None # 'first' to only count each patient as a case once
    ):

        nearly = 0
        nconsidered = 0
        nhadm = 0
        n_onevisit = 0
        n_cases = 0
        n_controls = 0

        prefix_is_same = lambda match, ref: match == ref[:len(match)]
        visit_has_code = lambda v, match: any([prefix_is_same(match, c) for c in self.diagnoses.visit[v]])
        hsamples = []
        for patient in tqdm(self.visits.keys()):

            if len(self.visits[patient]) < 2:
                n_onevisit += 1
                continue

            early_case = False
            for code, h in self.first_diagnosis[patient].items():
                if prefix_is_same(target_code, code) and h == self.visits[patient][0]:
                    early_case = True

            if early_case:
                nearly += 1
                continue

            nconsidered += 1

            running_hist = []
            for vi, h in enumerate(self.visits[patient][:-1]):

                nhadm += 1

                if censor_future:
                    # NOTE: current visit should not have the target diagnosis
                    #  therefore, after a control converts to a case, don't keep scanning
                    if visit_has_code(h, target_code):
                        break

                # NOTE: It is possible for the same patient to generate 1+ next diagnosis cases
                #  e.g. there are two visits close together for which both have the targ diag within the next month
                iscase = False
                future_visits_inrange = self.next_visits[h][time_window] \
                    if time_window is not None else self.visits[patient][vi+1:]
                for hnxt in future_visits_inrange:
                    if visit_has_code(hnxt, target_code):
                        iscase = True

                running_hist += [h]
                hsamples += [(patient, list.copy(running_hist) if use_history else [h], iscase)]

                if iscase and per_patient == 'first':
                    break

            if iscase: n_cases += 1
            else: n_controls += 1

        return hsamples, dict(
            patient=dict(
                total=len(self.visits),
                early=nearly,
                considered=nconsidered,
                onevisit=n_onevisit,
                cases=n_cases,
                controls=n_controls,
            )
        )

def load_splits(split_dir='artifacts/splits'):
    return { phase: np.genfromtxt(f'{split_dir}/{phase}_ids.txt') for phase in ['train', 'val', 'test'] }

def generate_data_splits(tte, hsamples, vocab, split_dir='artifacts/splits', format_code=lambda c: c):

    splits = load_splits(split_dir)

    datamats = dict()
    for phase, pids in splits.items():
        pids = { i: True for i in pids }

        Xrows = []
        yrows = []
        for pid, hls, iscase in tqdm(hsamples):
            if pid not in pids: continue
            row = np.zeros(len(vocab))
            for h in hls:
                for c in tte.diagnoses.visit[h]:
                    match_c = format_code(c)
                    if match_c in vocab:
                        row[vocab[match_c]] += 1
            Xrows += [row]
            yrows += [iscase]

        X = np.array(Xrows)
        y = np.array(yrows)

        datamats[phase] = [X, y]
    return datamats

def get_visit_history_embedding(
    tte, hsamples, vocab, embdict,
    aggregation=lambda els: np.sum(els, 0),
    split_dir='artifacts/splits', format_code=lambda c: c):

    splits = load_splits(split_dir)

    datamats = dict()
    for phase, pids in splits.items():
        pids = { i: True for i in pids }

        Xrows = []
        yrows = []
        for pid, hls, iscase in tqdm(hsamples):
            if pid not in pids: continue
            embs = []
            for h in hls:
                for c in tte.diagnoses.visit[h]:
                    match_c = format_code(c)
                    if match_c not in vocab: continue
                    if match_c in embdict:
                        embs += [embdict[match_c]]
                    else:
                        pass # NOTE: when no embed can be matched, code will be skipped
            if len(embs) == 0:
                # due to matching issues, sometimes no embs are collected
                embs = [np.zeros(len(next(iter(embdict.values()))))]
            Xrows += [aggregation(embs)]

        X = np.array(Xrows)

        datamats[phase] = X
    return datamats

class Inference:
    class MedBERT:
        # Compatible with MedBERT trained using
        #  https://github.com/kirilklein/Med-BERT.git

        def __init__(self):
            sys.path.append('saved/medbert/Med-BERT/medbert')
            from features.dataset import MLM_PLOS_Dataset
            from models.model import EHRBertForPretraining, EHRBertForMaskedLM
            from trainer.trainer import EHRTrainer
            from transformers import BertConfig
            from hydra import compose, initialize
            from features.tokenizer import EHRTokenizer
            import torch

            with initialize(config_path='saved/medbert'):
                cfg: compose(config_name='pretrain_mimiciv.yaml')

            bertconfig = BertConfig.from_json_file('saved/medbert/config.json')
            self.model = EHRBertForMaskedLM(bertconfig)
            self.model.load_state_dict(torch.load('saved/medbert/checkpoint_epoch99_end.pt')['model_state_dict'])
            tokenizer_config = DotDict({
                'sep_tokens': False, # should we add [SEP] tokens?
                'cls_token': False, # should we add a [CLS] token?
                'padding': True, # should we pad the sequences?
                'truncation': 100}) # how long should the longest sequence be
            self.tokenizer = EHRTokenizer(config=tokenizer_config)

        def __call__(self, **batch):
            # 'concept', 'age', 'abspos', 'segment', 'los'

            train_tokenized = self.tokenizer(batch)
            print(train_tokenized['concept'][:3])
