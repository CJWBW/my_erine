import torch
import torch.nn as nn
import pickle
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
from data_processor.covid_data import CovidDataProcessor
from bert.model import ErnieModel
from knowledge_bert import BertTokenizer
from knowledge_bert.modeling import BertForSequenceClassification, BertModel, PreTrainedBertModel

from pathlib import Path

KG_EMBED_PATH = str(Path(__file__).parent / "kg_embed")
ERNIE_BASE_PATH = str(Path(__file__).parent / "ernie_base")

NUM_LABELS = 2
BATCH_SIZE = 2
EPOCHS = 5
MAX_LEN = 128


def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():
        # Tell PyTorch to use the GPU.
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
        return device

    else:
        # print('No GPU available, using the CPU instead.')
        # device = torch.device("cpu")
        print('no GPU')
        exit()

# class MyClassifier(PreTrainedBertModel):
#
#     def __init__(self):
#         super(MyClassifier, self).__init__()
#         self.bert = BertModel(ERNIE_BASE_PATH)
#         self.dropout = nn.Dropout(p=0.3)
#         self.classifier = nn.Linear(768, 2)
#         # self.apply(self.init_bert_weights)
#
#
#     def __init__(self, config, num_labels=2):
#         super(BertForSequenceClassification, self).__init__(config)
#         self.num_labels = num_labels
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, num_labels)
#         self.apply(self.init_bert_weights)
#
#     def forward(self, input_ids, token_type_ids=None, attention_mask=None, input_ent=None, ent_mask=None, labels=None):
#         _, pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, input_ent=input_ent, ent_mask=ent_mask)
#         pooled_output = self.dropout(pooled_output)
#         return self.classifier(pooled_output)


def main():
    class_names = ['True', 'Fake']
    data_processor = CovidDataProcessor()
    labels, statements = data_processor.load_dataset()
    # convert text labels to 0-5
    labels = {'train': CovidDataProcessor.convert_labels(NUM_LABELS, labels['train']),
              'test': CovidDataProcessor.convert_labels(NUM_LABELS, labels['test']),
              'validation': CovidDataProcessor.convert_labels(NUM_LABELS, labels['validation'])}

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(ERNIE_BASE_PATH, do_lower_case=False)
    train_dataloader = CovidDataProcessor.create_ernie_dataloader(statements['train'], labels['train'], tokenizer,
                                                                  MAX_LEN, BATCH_SIZE)

    test_dataloader = CovidDataProcessor.create_ernie_dataloader(statements['test'], labels['test'], tokenizer, MAX_LEN,
                                                                 BATCH_SIZE)

    validation_dataloader = CovidDataProcessor.create_ernie_dataloader(statements['validation'], labels['validation'],
                                                                       tokenizer, MAX_LEN, BATCH_SIZE)

    device = get_device()
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Train model
    # model, _ = BertModel.from_pretrained(ERNIE_BASE_PATH)
    # model = MyClassifier()
    model, _ = BertForSequenceClassification.from_pretrained(ERNIE_BASE_PATH, num_labels=NUM_LABELS)

    # model = BERTClassifier(model_option="bert-base-uncased", n_classes=NUM_LABELS)

    model.to(device)
    train_history = ErnieModel.train_model(model, train_dataloader, validation_dataloader, len(statements['train']),
                                           len(statements['validation']), EPOCHS, device, loss_fn)

    # evaluate model on test dataset
    test_acc, _ = ErnieModel.eval_model(model, test_dataloader, len(statements['test']), device, loss_fn)
    print('test accuracy: ', test_acc.item())

    # predictions
    pred, test_labels = ErnieModel.get_predictions(model, test_dataloader, device)

    print(classification_report(test_labels, pred, target_names=class_names, digits=4))
    with open('record_.txt', 'wb') as f:
        pickle.dump(pred, f)
        pickle.dump(test_labels, f)


if __name__ == "__main__":
    main()
