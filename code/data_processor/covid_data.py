import pandas as pd
from collections import Counter
from pathlib import Path
import torch
import tagme
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

KG_EMBED_PATH = str(Path(__file__).parent / "../kg_embed")
ERNIE_BASE_PATH = str(Path(__file__).parent / "../ernie_base")

# dataset = 'covid'
dataset = 'liar'


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, input_ent, ent_mask, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.input_ent = input_ent
        self.ent_mask = ent_mask


class CovidDataProcessor:
    if dataset == 'covid':
        TRAIN_PATH = Path(__file__).parent / "../data/covid/covid_train.tsv"
        TEST_PATH = Path(__file__).parent / "../data/covid/covid_test_with_label.tsv"
        VAL_PATH = Path(__file__).parent / "../data/covid/covid_valid.tsv"
    else:
        # use same file to load liar dataset
        TRAIN_PATH = Path(__file__).parent / "../data/liar/train.tsv"
        TEST_PATH = Path(__file__).parent / "../data/liar/test.tsv"
        VAL_PATH = Path(__file__).parent / "../data/liar/valid.tsv"

    @staticmethod
    def load_dataset():
        train_df = pd.read_csv(CovidDataProcessor.TRAIN_PATH, sep="\t", header=None)[:]
        test_df = pd.read_csv(CovidDataProcessor.TEST_PATH, sep="\t", header=None)[:]
        val_df = pd.read_csv(CovidDataProcessor.VAL_PATH, sep="\t", header=None)[:]

        # Fill nan (empty boxes) with -1
        train_df = train_df.fillna(-1)
        test_df = test_df.fillna(-1)
        val_df = val_df.fillna(-1)

        train_df = train_df.to_numpy()
        test_df = test_df.to_numpy()
        val_df = val_df.to_numpy()

        # for covid
        if dataset == 'covid':
            statements = {'train': [train_df[i][1] for i in range(len(train_df))],
                          'test': [test_df[i][1] for i in range(len(test_df))],
                          'validation': [val_df[i][1] for i in range(len(val_df))]}
            labels = {'train': [train_df[i][2] for i in range(len(train_df))],
                      'test': [test_df[i][2] for i in range(len(test_df))],
                      'validation': [val_df[i][2] for i in range(len(val_df))]}
        else:
            # liar has different position
            labels = {'train': [train_df[i][1] for i in range(len(train_df))],
                      'test': [test_df[i][1] for i in range(len(test_df))],
                      'validation': [val_df[i][1] for i in range(len(val_df))]}
            statements = {'train': [train_df[i][2] for i in range(len(train_df))],
                          'test': [test_df[i][2] for i in range(len(test_df))],
                          'validation': [val_df[i][2] for i in range(len(val_df))]}

        # see the label distribution
        print('Label distribution in training dataset')
        print(f"train: {Counter(labels['train'])}")
        print(f"test: {Counter(labels['test'])}")
        print(f"validation: {Counter(labels['validation'])}")

        return labels, statements

    @staticmethod
    def convert_labels(num_labels, labels):
        if dataset == 'covid':
            encoded_labels = [0] * len(labels)
            if num_labels == 2:
                for i in range(len(labels)):
                    if labels[i] == 'real':
                        encoded_labels[i] = 0
                    elif labels[i] == 'fake':
                        encoded_labels[i] = 1
                    else:
                        print('Incorrect label')
            return encoded_labels
        else:
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
    def get_ernie_dataloader(statements, labels, max_len, tokenizer, bath_size, entity2id, ent_map):
        features = CovidDataProcessor.convert_to_ernie_features(statements, labels, max_len, tokenizer, entity2id,
                                                                ent_map)

        all_input_ids = torch.tensor([feature.input_ids for feature in features], dtype=torch.long)
        all_input_mask = torch.tensor([feature.input_mask for feature in features], dtype=torch.long)
        all_segment_ids = torch.tensor([feature.segment_ids for feature in features], dtype=torch.long)
        all_label_ids = torch.tensor([feature.label_id for feature in features], dtype=torch.long)
        all_ent = torch.tensor([feature.input_ent for feature in features], dtype=torch.long)
        all_ent_masks = torch.tensor([feature.ent_mask for feature in features], dtype=torch.long)
        data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_ent, all_ent_masks, all_label_ids)

        return DataLoader(data, sampler=RandomSampler(data), batch_size=bath_size, num_workers=4)

    @staticmethod
    def convert_to_ernie_features(statements, labels, max_seq_length, tokenizer, entity2id, ent_map):
        # Set the authorization token for subsequent calls.
        tagme.GCUBE_TOKEN = "c8623405-ea8c-4c06-8394-ef7550483f75-843339462"

        def get_ents(ann):
            ents = []
            # Keep annotations with a score higher than 0.3
            for a in ann.get_annotations(0.3):
                if a.entity_title not in ent_map:
                    continue
                ents.append([ent_map[a.entity_title], a.begin, a.end, a.score])
            return ents

        features = []
        a = 1
        for statement, label in zip(statements, labels):
            print(a)

            statement_ann = tagme.annotate(statement)
            ents_statement = get_ents(statement_ann)

            # Tokenize
            tokens_statement, entities_statement = tokenizer.tokenize(statement, ents_statement)

            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_statement) > max_seq_length - 2:
                tokens_statement = tokens_statement[:(max_seq_length - 2)]
                entities_statement = entities_statement[:(max_seq_length - 2)]

            tokens = ["[CLS]"] + tokens_statement + ["[SEP]"]
            ents = ["UNK"] + entities_statement + ["UNK"]
            segment_ids = [0] * len(tokens)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ent = []
            ent_mask = []

            for ent in ents:
                if ent != "UNK" and ent in entity2id:
                    input_ent.append(entity2id[ent])
                    ent_mask.append(1)
                else:
                    input_ent.append(-1)
                    ent_mask.append(0)
            ent_mask[0] = 1

            # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            padding_ = [-1] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            input_ent += padding_
            ent_mask += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(input_ent) == max_seq_length
            assert len(ent_mask) == max_seq_length

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              input_ent=input_ent,
                              ent_mask=ent_mask,
                              label_id=label))
            a += 1

        return features
