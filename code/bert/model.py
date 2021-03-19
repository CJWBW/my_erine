from collections import defaultdict
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm
from knowledge_bert.optimization import BertAdam


class ErnieModel:

    @staticmethod
    def warmup_linear(x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0

    @staticmethod
    def train_epoch(model, optimizer, train_dataloader, device, embed, t_total):

        model.train()

        global_step = 0

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.to(device) if i != 3 else t for i, t in enumerate(batch))
            input_ids, input_mask, segment_ids, input_ent, ent_mask, labels = batch
            input_ent = embed(input_ent + 1).to(device)  # -1 -> 0

            loss = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                input_ent=input_ent.half(),
                ent_mask=ent_mask,
                labels=labels
            )
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1

            if (step + 1) % 1 == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = 2e-5 * ErnieModel.warmup_linear(global_step / t_total, 0.1)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

    @staticmethod
    def train_model(model, train_dataloader, validation_dataloader, epochs, device, loss_fn,
                    embed):

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())
        no_grad = ['bert.encoder.layer.11.output.dense_ent', 'bert.encoder.layer.11.output.LayerNorm_ent']
        param_optimizer = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_grad)]
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=2e-5,
                             warmup=0.1,
                             t_total=total_steps)

        # Measure the total training time for the whole run.
        total_t0 = time.time()

        history = defaultdict(list)
        best_accuracy = 0

        for epoch in range(epochs):

            # ========================================
            #               Training
            # ========================================

            print('')
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
            # print(f'======== Epoch {epoch + 1} / {epochs} ========')
            print('Training...')

            # Measure how long the training epoch takes.
            t0 = time.time()

            ErnieModel.train_epoch(model, optimizer, train_dataloader, device, embed, total_steps)

            print('Epoch {:} took {:} minutes'.format(epoch + 1, (time.time() - t0) / 60))

            # ========================================
            #               Validation
            # ========================================

            print('')
            print("Running Validation...")

            val_acc, val_loss = ErnieModel.eval_model(model, validation_dataloader, device, embed)
            print('Validation loss: {:}, accuracy: {:}'.format(val_loss, val_acc))
            print('')

            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)

            if val_acc > best_accuracy:
                torch.save(model.state_dict(), 'best_model_state.bin')
                best_accuracy = val_acc

        print('')
        print('Total Training took: {:} minutes'.format((time.time() - total_t0) / 60))
        print('Best validation accuracy: {:}'.format(best_accuracy))
        return history

    @staticmethod
    def accuracy(out, label):
        outputs = np.argmax(out, axis=1)
        return np.sum(outputs == label), outputs

    @staticmethod
    def eval_model(model, validation_dataloader, device, embed):
        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        for batch in validation_dataloader:

            batch = tuple(t.to(device) if i != 3 else t for i, t in enumerate(batch))
            input_ids, input_mask, segment_ids, input_ent, ent_mask, labels = batch
            input_ent = embed(input_ent + 1).to(device)  # -1 -> 0

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, input_ent, ent_mask, labels)
                logits = model(input_ids, segment_ids, input_mask, input_ent, ent_mask)

            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            tmp_eval_accuracy, pred = ErnieModel.accuracy(logits, labels)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        return eval_accuracy, eval_loss

    @staticmethod
    def get_predictions(model, test_dataloader, device, embed):
        model.eval()
        predictions, true_labels = [], []

        for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):

            batch = tuple(t.to(device) if i != 3 else t for i, t in enumerate(batch))
            input_ids, input_mask, segment_ids, input_ent, ent_mask, labels = batch
            input_ent = embed(input_ent + 1).to(device)  # -1 -> 0

            with torch.no_grad():
                logits = model(input_ids, segment_ids, input_mask, input_ent, ent_mask)

            logits = logits.detach().cpu().numpy()
            labels = labels.to('cpu').numpy()
            _, pred = ErnieModel.accuracy(logits, labels)
            predictions.extend(pred)
            true_labels.extend(labels)

        return predictions, true_labels


