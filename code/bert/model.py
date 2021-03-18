from collections import defaultdict
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import torch.nn as nn
import numpy as np
import time
from tqdm import tqdm


class ErnieModel:

    @staticmethod
    def train_epoch(model, optimizer, scheduler, train_dataloader, n_examples, device, loss_fn, embed):

        model.train()

        # Store the average loss after each epoch so we can plot them.
        losses = []
        correct_predictions = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

            batch = tuple(t.to(device) if i != 3 else t for i, t in enumerate(batch))
            input_ids, input_mask, segment_ids, input_ent, ent_mask, labels = batch
            input_ent = embed(input_ent + 1).to(device)  # -1 -> 0

            outputs = model(
                input_ids=input_ids,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                input_ent=input_ent.half(),
                ent_mask=ent_mask,
                labels=labels
            )

            _, pred = torch.max(outputs, dim=1)
            loss = loss_fn(outputs.view(-1, 2), labels.view(-1))
            # loss = loss_fn(outputs, labels)
            correct_predictions += torch.sum(pred == labels)
            losses.append(loss.item())

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

        return correct_predictions.double() / n_examples, np.mean(losses)

    @staticmethod
    def train_model(model, train_dataloader, validation_dataloader, train_len, validation_len, epochs, device, loss_fn,
                    embed):
        optimizer = AdamW(model.parameters(), lr=1e-5, correct_bias=False)

        # Total number of training steps is number of batches * number of epochs.
        total_steps = len(train_dataloader) * epochs

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

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

            train_acc, train_loss = ErnieModel.train_epoch(model, optimizer, scheduler, train_dataloader, train_len,
                                                           device, loss_fn, embed)

            print('Train loss: {:}, accuracy: {:}'.format(train_loss, train_acc))
            print('Epoch {:} took {:} minutes'.format(epoch + 1, (time.time() - t0) / 60))

            # ========================================
            #               Validation
            # ========================================

            print('')
            print("Running Validation...")

            val_acc, val_loss = ErnieModel.eval_model(model, validation_dataloader, validation_len, device, loss_fn,
                                                      embed)
            print('Validation loss: {:}, accuracy: {:}'.format(val_loss, val_acc))
            print('')

            history['train_acc'].append(train_acc)
            history['train_loss'].append(train_loss)
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
    def eval_model(model, validation_dataloader, n_examples, device, loss_fn, embed):
        model.eval()

        losses = []
        correct_predictions = 0

        with torch.no_grad():

            for step, batch in enumerate(tqdm(validation_dataloader, desc="Iteration")):

                batch = tuple(t.to(device) if i != 3 else t for i, t in enumerate(batch))
                input_ids, input_mask, segment_ids, input_ent, ent_mask, labels = batch
                input_ent = embed(input_ent + 1).to(device)  # -1 -> 0

                outputs = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    input_ent=input_ent.half(),
                    ent_mask=ent_mask,
                    labels=labels
                )

                _, pred = torch.max(outputs, dim=1)
                loss = loss_fn(outputs, labels)

                correct_predictions += torch.sum(pred == labels)
                losses.append(loss.item())

        return correct_predictions.double() / n_examples, np.mean(losses)

    @staticmethod
    def get_predictions(model, test_dataloader, device, embed):
        model.eval()

        # Tracking variables
        predictions, true_labels = [], []
        with torch.no_grad():
            # Predict
            for step, batch in enumerate(tqdm(test_dataloader, desc="Iteration")):

                batch = tuple(t.to(device) if i != 3 else t for i, t in enumerate(batch))
                input_ids, input_mask, segment_ids, input_ent, ent_mask, labels = batch
                input_ent = embed(input_ent + 1).to(device)  # -1 -> 0

                outputs = model(
                    input_ids=input_ids,
                    token_type_ids=segment_ids,
                    attention_mask=input_mask,
                    input_ent=input_ent.half(),
                    ent_mask=ent_mask
                )

                _, pred = torch.max(outputs, dim=1)

                # Store predictions and true labels
                predictions.extend(pred)
                true_labels.extend(labels)

        predictions = torch.stack(predictions).cpu()
        true_labels = torch.stack(true_labels).cpu()
        return predictions, true_labels
