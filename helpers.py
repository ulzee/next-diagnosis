import torch
from os.path import join
import sys
from configs import project_root
import pickle as pk
from tqdm import tqdm

class DotDict(dict):
    def __getattr__(self, attr):
        return self[attr]

class TTE:

    class diagnoses:
        pass

    def __init__(self):
        with open(f'{project_root}/saved/blob.pk', 'rb') as fl:
            blob = pk.load(fl)
        self.first_diagnosis = blob['first_diagnosis']
        self.visits = blob['visits']
        self.next_visits = blob['next_visits']

        self.diagnoses = TTE.diagnoses()
        self.diagnoses.visit = blob['diagnoses']['visit']

    def gather_samples(self,
        chapter=None, time_window='month',
        censor_future=True,
        use_history=True
    ):

        nearly = 0
        nconsidered = 0
        nhadm = 0
        nmonth = 0
        n_onevisit = 0
        n_cases = 0
        n_controls = 0

        visit_has_code = lambda v, match: any([c[:len(match)] == match for c in self.diagnoses.visit[v]])

        hsamples = []
        for patient in tqdm(self.visits.keys()):

            early_case = False
            for code, h in self.first_diagnosis[patient].items():
                if chapter == code[0] and h == self.visits[patient][0]:
                    early_case = True

            if len(self.visits[patient]) < 2:
                n_onevisit += 1
                continue

            if early_case:
                nearly += 1
                continue

            nconsidered += 1

            running_hist = []
            for h in self.visits[patient][:-1]:

                nhadm += 1
                nmonth += len(self.next_visits[h][time_window])

                if censor_future:
                    # NOTE: current visit should not have the target diagnosis
                    #  therefore, after a control converts to a case, don't keep scanning
                    if visit_has_code(h, chapter):
                        break

                # NOTE: It is possible for the same patient to generate 1+ next diagnosis cases
                #  e.g. there are two visits close together for which both have the targ diag within the next month
                iscase = False
                for hnxt in self.next_visits[h][time_window]:
                    if visit_has_code(hnxt, chapter):
                        iscase = True

                running_hist += [h]
                hsamples += [(list.copy(running_hist) if use_history else [h], iscase)]

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
