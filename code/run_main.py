import torch
import pickle
from sklearn.metrics import classification_report
from data_processor.covid_data import CovidDataProcessor
from bert.model import ErnieModel
from knowledge_bert import BertTokenizer
from knowledge_bert.modeling import BertForSequenceClassification
from pathlib import Path

KG_EMBED_PATH = str(Path(__file__).parent / "kg_embed")
ERNIE_BASE_PATH = str(Path(__file__).parent / "ernie_base")

NUM_LABELS = 2
BATCH_SIZE = 4
EPOCHS = 10
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


def main():
    device = get_device()
    class_names = ['True', 'Fake']
    data_processor = CovidDataProcessor()
    labels, statements = data_processor.load_dataset()
    # convert text labels to 0-5
    labels = {'train': CovidDataProcessor.convert_labels(NUM_LABELS, labels['train']),
              'test': CovidDataProcessor.convert_labels(NUM_LABELS, labels['test']),
              'validation': CovidDataProcessor.convert_labels(NUM_LABELS, labels['validation'])}

    # Load pre-trained model tokenizer
    tokenizer = BertTokenizer.from_pretrained(ERNIE_BASE_PATH, do_lower_case=False)

    with open('embed.txt', 'rb') as f:
        embed = pickle.load(f)
    with open('entity2id.txt', 'rb') as f:
        entity2id = pickle.load(f)
    with open('ent_map.txt', 'rb') as f:
        ent_map = pickle.load(f)

    # currently all saved dataloader is generated with batch_size = 4, if change to other batch_size, need to regenerate
    if Path('covid_train_dataloader.txt').is_file():
        with open('covid_train_dataloader.txt', 'rb') as ff:
            train_dataloader = pickle.load(ff)
    else:
        print('generating train_dataloader')
        train_dataloader = CovidDataProcessor.get_ernie_dataloader(statements['train'], labels['train'], MAX_LEN,
                                                                   tokenizer, BATCH_SIZE, entity2id, ent_map)
        with open('covid_train_dataloader.txt', 'wb') as ff:
            pickle.dump(train_dataloader, ff)

    if Path('covid_test_dataloader.txt').is_file():
        with open('covid_test_dataloader.txt', 'rb') as ff:
            test_dataloader = pickle.load(ff)
    else:
        print('generating test_dataloader')
        test_dataloader = CovidDataProcessor.get_ernie_dataloader(statements['test'], labels['test'], MAX_LEN,
                                                                  tokenizer, BATCH_SIZE, entity2id, ent_map)
        with open('covid_test_dataloader.txt', 'wb') as ff:
            pickle.dump(test_dataloader, ff)

    if Path('covid_val_dataloader.txt').is_file():
        with open('covid_val_dataloader.txt', 'rb') as ff:
            validation_dataloader = pickle.load(ff)
    else:
        print('generating validation_dataloader')
        validation_dataloader = CovidDataProcessor.get_ernie_dataloader(statements['validation'], labels['validation'],
                                                                        MAX_LEN, tokenizer, BATCH_SIZE, entity2id,
                                                                        ent_map)
        with open('covid_val_dataloader.txt', 'wb') as ff:
            pickle.dump(validation_dataloader, ff)

    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    # Train model
    model, _ = BertForSequenceClassification.from_pretrained(ERNIE_BASE_PATH, num_labels=NUM_LABELS)
    model.to(device)

    train_history = ErnieModel.train_model(model, train_dataloader, validation_dataloader, len(statements['train']),
                                           len(statements['validation']), EPOCHS, device, loss_fn, embed)

    # evaluate model on test dataset
    test_acc, _ = ErnieModel.eval_model(model, test_dataloader, len(statements['test']), device, loss_fn, embed)
    print('test accuracy: ', test_acc.item())

    # predictions
    pred, test_labels = ErnieModel.get_predictions(model, test_dataloader, device, embed)

    print(classification_report(test_labels, pred, target_names=class_names, digits=4))
    with open('record_.txt', 'wb') as f:
        pickle.dump(pred, f)
        pickle.dump(test_labels, f)


if __name__ == "__main__":
    main()
