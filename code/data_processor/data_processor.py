import pandas as pd
import torch
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np


class LiarDataset(Dataset):

    def __init__(self, statements, labels, metadata, states, affiliations, credit_count, tokenizer, max_len):
        self.statements = statements
        self.labels = labels
        self.metadata = metadata
        self.states = np.array(states)
        self.affiliations = np.array(affiliations)
        self.credit_count = np.array(credit_count)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.categorical_numerical_data = []

        for credit, state, affiliation in zip(credit_count, states, affiliations):
            self.categorical_numerical_data.append(credit + state + affiliation)

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, item):
        statements = str(self.statements[item])
        labels = self.labels[item]

        metadata = str(self.metadata[item])
        categorical_numerical_data = self.categorical_numerical_data[item]

        encoding = self.tokenizer.encode_plus(
            statements,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        meta_encoding = self.tokenizer.encode_plus(
            metadata,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask1': encoding['attention_mask'].flatten(),
            'metadata': meta_encoding['input_ids'].flatten(),
            'attention_mask2': meta_encoding['attention_mask'].flatten(),
            'categorical_numerical_data': torch.tensor(categorical_numerical_data),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


class DataProcessor:
    TRAIN_PATH = Path(__file__).parent / "../data/train.tsv"
    TEST_PATH = Path(__file__).parent / "../data/test.tsv"
    VAL_PATH = Path(__file__).parent / "../data/valid.tsv"

    state_dict = {}
    party_dict = {}

    # metadata version

    @staticmethod
    def create_dataloader(statements, labels, metadata, states, affiliations, credit_count, tokenizer, max_len,
                          batch_size):
        dataset = LiarDataset(
            statements=statements,
            labels=labels,
            metadata=metadata,
            states=states,
            affiliations=affiliations,
            credit_count=credit_count,
            tokenizer=tokenizer,
            max_len=max_len
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4
        )

    def load_dataset(self):
        train_df0 = pd.read_csv(DataProcessor.TRAIN_PATH, sep="\t", header=None)
        train_df = train_df0[:100]
        test_df0 = pd.read_csv(DataProcessor.TEST_PATH, sep="\t", header=None)
        test_df = test_df0[:10]
        val_df0 = pd.read_csv(DataProcessor.VAL_PATH, sep="\t", header=None)
        val_df = val_df0[:10]

        # Fill nan (empty boxes) with -1
        train_df = train_df.fillna(-1)
        test_df = test_df.fillna(-1)
        val_df = val_df.fillna(-1)

        train_df = train_df.to_numpy()
        test_df = test_df.to_numpy()
        val_df = val_df.to_numpy()

        labels = {'train': [train_df[i][1] for i in range(len(train_df))],
                  'test': [test_df[i][1] for i in range(len(test_df))],
                  'validation': [val_df[i][1] for i in range(len(val_df))]}
        statements = {'train': [train_df[i][2] for i in range(len(train_df))],
                      'test': [test_df[i][2] for i in range(len(test_df))],
                      'validation': [val_df[i][2] for i in range(len(val_df))]}

        subjects = {'train': [train_df[i][3] for i in range(len(train_df))],
                    'test': [test_df[i][3] for i in range(len(test_df))],
                    'validation': [val_df[i][3] for i in range(len(val_df))]}

        speakers = {'train': [train_df[i][4] for i in range(len(train_df))],
                    'test': [test_df[i][4] for i in range(len(test_df))],
                    'validation': [val_df[i][4] for i in range(len(val_df))]}
        jobs = {'train': [train_df[i][5] for i in range(len(train_df))],
                'test': [test_df[i][5] for i in range(len(test_df))],
                'validation': [val_df[i][5] for i in range(len(val_df))]}

        states = {'train': [self.convert_state_to_num(train_df[i][6]) for i in range(len(train_df))],
                  'test': [self.convert_state_to_num(test_df[i][6]) for i in range(len(test_df))],
                  'validation': [self.convert_state_to_num(val_df[i][6]) for i in range(len(val_df))]}
        affiliations = {'train': [self.convert_party(train_df[i][7]) for i in range(len(train_df))],
                        'test': [self.convert_party(test_df[i][7]) for i in range(len(test_df))],
                        'validation': [self.convert_party(val_df[i][7]) for i in range(len(val_df))]}
        credit_count = {'train': [DataProcessor.convert_credit_history(labels['train'][i], train_df[i][8:13]) for i in
                                  range(len(train_df))],
                        'test': [DataProcessor.convert_credit_history(labels['test'][i], test_df[i][8:13]) for i in
                                 range(len(test_df))],
                        'validation': [DataProcessor.convert_credit_history(labels['validation'][i], val_df[i][8:13])
                                       for i in range(len(val_df))]}
        contexts = {'train': [train_df[i][13] for i in range(len(train_df))],
                    'test': [test_df[i][13] for i in range(len(test_df))],
                    'validation': [val_df[i][13] for i in range(len(val_df))]}

        # see the label distribution
        print('Label distribution in training dataset')
        print(Counter(labels['train']))
        print(Counter(labels['test']))
        print(Counter(labels['validation']))

        metadata = {'train': [0] * len(train_df), 'validation': [0] * len(val_df), 'test': [0] * len(test_df)}
        metadata = {'train': DataProcessor.convert_none_value(metadata['train'], subjects['train'], speakers['train'],
                                                              jobs['train'], contexts['train']),
                    'test': DataProcessor.convert_none_value(metadata['test'], subjects['test'], speakers['test'],
                                                             jobs['test'], contexts['test']),
                    'validation': DataProcessor.convert_none_value(metadata['validation'], subjects['validation'],
                                                                   speakers['validation'], jobs['validation'],
                                                                   contexts['validation'])}

        return labels, statements, metadata, states, affiliations, credit_count

    @staticmethod
    def convert_none_value(metadata, subjects, speakers, jobs, contexts):
        for i in range(len(metadata)):
            subject = subjects[i] if subjects[i] != -1 else 'None'
            speaker = speakers[i] if speakers[i] != -1 else 'None'
            job = jobs[i] if jobs[i] != -1 else 'None'
            context = contexts[i] if contexts[i] != -1 else 'None'
            meta = subject + ' ' + speaker + ' ' + job + ' ' + context
            metadata[i] = meta
        return metadata

    def convert_party(self, party):
        if party not in self.party_dict:
            self.party_dict[party] = [len(self.party_dict)]
        return self.party_dict[party]

    def convert_state_to_num(self, state):
        if state not in self.state_dict:
            self.state_dict[state] = [len(self.state_dict)]
        return self.state_dict[state]

    @staticmethod
    def convert_labels(num_labels, labels):
        if not num_labels == 6 and not num_labels == 2:
            print('Invalid number of labels. The number of labels should be either 2 or 6')
        # only consider binary case now
        encoded_labels = [0] * len(labels)
        if num_labels == 2:
            for i in range(len(labels)):
                if labels[i] in ['true', 'mostly-true', 'half-true']:
                    encoded_labels[i] = 0
                elif labels[i] in ['barely-true', 'false', 'pants-fire']:
                    encoded_labels[i] = 1
                else:
                    print('Incorrect label')
        else:
            for i in range(len(labels)):
                if labels[i] == 'true':
                    encoded_labels[i] = 0
                elif labels[i] == 'mostly-true':
                    encoded_labels[i] = 1
                elif labels[i] == 'half-true':
                    encoded_labels[i] = 2
                elif labels[i] == 'barely-true':
                    encoded_labels[i] = 3
                elif labels[i] == 'false':
                    encoded_labels[i] = 4
                elif labels[i] == 'pants-fire':
                    encoded_labels[i] = 5
                else:
                    print('Incorrect label')
        return encoded_labels

    @staticmethod
    def convert_credit_history(cur_label, history_credit):
        # column 8: 13 are the credit history. according to the paper:
        # history vector h = {19, 32, 34, 58, 33}, which corresponds to his counts of “pants on fire”, “false”,
        # “barely true”, “half true”, “mostly true” for historical statements, Since this vector also includes the
        # count for the current statement, it is important to subtract the current label from the credit history
        # when using this meta data vector in prediction experiments
        credit_and_sum = []
        for credit in history_credit:
            credit_and_sum.append(int(credit))
        if cur_label == 'pants-fire':
            credit_and_sum[0] -= 1
        elif cur_label == 'false':
            credit_and_sum[1] -= 1
        elif cur_label == 'barely-true':
            credit_and_sum[2] -= 1
        elif cur_label == 'half-true':
            credit_and_sum[3] -= 1
        elif cur_label == 'mostly-true':
            credit_and_sum[4] -= 1
        credit_and_sum.append(sum(credit_and_sum))
        return credit_and_sum
