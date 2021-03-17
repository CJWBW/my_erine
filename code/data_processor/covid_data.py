import pandas as pd
from collections import Counter
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch
import tagme
import pickle

KG_EMBED_PATH = str(Path(__file__).parent / "../kg_embed")
ERNIE_BASE_PATH = str(Path(__file__).parent / "../ernie_base")

with open('embed.txt', 'rb') as f:
    embed = pickle.load(f)


# class CovidDataset(Dataset):
#
#     def __init__(self, statements, labels, tokenizer, max_len):
#         self.statements = statements
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#
#     def __len__(self):
#         return len(self.statements)
#
#     def __getitem__(self, item):
#         statements = str(self.statements[item])
#         labels = self.labels[item]
#
#         encoding = self.tokenizer.encode_plus(
#             statements,
#             add_special_tokens=True,
#             truncation=True,
#             max_length=self.max_len,
#             return_token_type_ids=False,
#             padding='max_length',
#             return_attention_mask=True,
#             return_tensors='pt',
#         )
#
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(labels, dtype=torch.long)
#         }


class ErnieCovidDataset(Dataset):

    def __init__(self, statements, labels, tokenizer, max_len):
        self.statements = statements
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.statements)

    def __getitem__(self, item):
        statements = str(self.statements[item])
        labels = self.labels[item]
        return CovidDataProcessor.convert_to_ernie_features(statements, labels, self.max_len, self.tokenizer)


class CovidDataProcessor:
    TRAIN_PATH = Path(__file__).parent / "../data/covid/covid_train.tsv"
    TEST_PATH = Path(__file__).parent / "../data/covid/covid_test_with_label.tsv"
    VAL_PATH = Path(__file__).parent / "../data/covid/covid_valid.tsv"

    state_dict = {}

    @staticmethod
    def load_dataset():
        train_df = pd.read_csv(CovidDataProcessor.TRAIN_PATH, sep="\t", header=None)[:100]
        test_df = pd.read_csv(CovidDataProcessor.TEST_PATH, sep="\t", header=None)[:10]
        val_df = pd.read_csv(CovidDataProcessor.VAL_PATH, sep="\t", header=None)[:10]

        # Fill nan (empty boxes) with -1
        train_df = train_df.fillna(-1)
        test_df = test_df.fillna(-1)
        val_df = val_df.fillna(-1)

        train_df = train_df.to_numpy()
        test_df = test_df.to_numpy()
        val_df = val_df.to_numpy()

        statements = {'train': [train_df[i][1] for i in range(len(train_df))],
                      'test': [test_df[i][1] for i in range(len(test_df))],
                      'validation': [val_df[i][1] for i in range(len(val_df))]}
        labels = {'train': [train_df[i][2] for i in range(len(train_df))],
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

    # @staticmethod
    # def create_dataloader(statements, labels, tokenizer, max_len, batch_size):
    #     dataset = CovidDataset(
    #         statements=statements,
    #         labels=labels,
    #         tokenizer=tokenizer,
    #         max_len=max_len
    #     )
    #
    #     return DataLoader(
    #         dataset,
    #         batch_size=batch_size,
    #         num_workers=4
    #     )

    @staticmethod
    def create_ernie_dataloader(statements, labels, tokenizer, max_len, batch_size):
        dataset = ErnieCovidDataset(
            statements=statements,
            labels=labels,
            tokenizer=tokenizer,
            max_len=max_len
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4
        )

    @staticmethod
    def convert_to_ernie_features(statement, label, max_seq_length, tokenizer):
        # Set the authorization token for subsequent calls.
        tagme.GCUBE_TOKEN = "c8623405-ea8c-4c06-8394-ef7550483f75-843339462"
        statement_ann = tagme.annotate(statement)

        # Read entity map
        ent_map = {}
        with open(KG_EMBED_PATH + "/entity_map.txt") as fin:
            for line in fin:
                name, qid = line.strip().split("\t")
                ent_map[name] = qid

        def get_ents(ann):
            ents = []
            # Keep annotations with a score higher than 0.3
            for a in ann.get_annotations(0.3):
                if a.entity_title not in ent_map:
                    continue
                ents.append([ent_map[a.entity_title], a.begin, a.end, a.score])
            return ents

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

        # Convert ents
        entity2id = {}
        with open(KG_EMBED_PATH + "/entity2id.txt") as fin:
            fin.readline()
            for line in fin:
                qid, eid = line.strip().split('\t')
                entity2id[qid] = int(eid)

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

        input_ent_tensor = torch.tensor([input_ent]) + 1
        input_ent = embed(input_ent_tensor)

        features = {'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'input_mask': torch.tensor(input_mask, dtype=torch.long),
                    'segment_ids': torch.tensor(segment_ids, dtype=torch.long),
                    'input_ent': input_ent.half().type(torch.LongTensor),
                    'ent_mask': torch.tensor(ent_mask, dtype=torch.long),
                    'labels': torch.tensor(label, dtype=torch.long)}
        return features
