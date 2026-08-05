import os

from tqdm.notebook import tqdm
import torch
from sklearn.metrics import classification_report, accuracy_score


def train(model,
          dataloader,
          loss_function,
          lr,
          num_epochs,
          valid_loader=None,
          test_loader=None,
          train_skip_gram=False,
          base_path=''):
    os.makedirs(f'{base_path}/GeneModels', exist_ok=True)
    # pytorch training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {'train': [], 'valid': [], 'test': []}
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader)

        all_preds = []
        all_labels = []
        current_loss = 0
        for batch in pbar:

            batch_sequences, y = batch
            x = batch_sequences.to('cuda')
            if train_skip_gram:
                y = x.clone()
            else:
                y = y.to('cuda')

            h = model(x)  # [B, C] or [B, L, V]
            if train_skip_gram:
                h = h.permute(0, 2, 1)  # [B, V, L]
            j = loss_function(h, y)

            # do gradient descent
            optimizer.zero_grad()  # remove junk from last step
            j.backward()  # calculate gradient from current batch outputs
            optimizer.step()  # update the weights using the gradients

            current_loss += j.item()

            if not train_skip_gram:
                all_preds.append(h.argmax(-1).detach().cpu())
                all_labels.append(y.cpu())

        if train_skip_gram:
            train_score = current_loss / len(dataloader)
        else:
            all_preds = torch.cat(all_preds).numpy()
            all_labels = torch.cat(all_labels).numpy()

            print(classification_report(all_labels, all_preds, digits=4))
            train_score = accuracy_score(all_labels, all_preds)

        if valid_loader is not None:
            val_score = evaluate(model,
                                 valid_loader,
                                 train_skip_gram=train_skip_gram,
                                 loss_function=loss_function)
            history['valid'].append(val_score)
        if test_loader is not None:
            test_score = evaluate(model,
                                  test_loader,
                                  train_skip_gram=train_skip_gram,
                                  loss_function=loss_function)
            history['test'].append(test_score)

        history['train'].append(train_score)

        if train_skip_gram:
            model_name = f'sg{epoch}.pth'
        else:
            model_name = f'rnn{epoch}.pth'
        torch.save(model.state_dict(), f'{base_path}/GeneModels/{model_name}')

    return history


def evaluate(model, valid_loader, loss_function=None, train_skip_gram=False):
    valid_preds = []
    valid_labels = []
    current_loss = 0
    for batch in valid_loader:
        batch_sequences, y = batch
        x = batch_sequences.to('cuda')
        if train_skip_gram:
            y = x.clone()
        else:
            y = y.to('cuda')
        h = model(x)  # [B, C]
        if train_skip_gram:
            h = h.permute(0, 2, 1)

        if train_skip_gram:
            assert loss_function is not None, "loss function is None"
            j = loss_function(h, y)
            current_loss += j.item()
        else:
            valid_preds.append(h.argmax(-1).detach().cpu())
            valid_labels.append(y.cpu())

    if train_skip_gram:
        loss = current_loss / len(valid_loader)
        return loss

    valid_preds = torch.cat(valid_preds).numpy()
    valid_labels = torch.cat(valid_labels).numpy()

    print(classification_report(valid_labels, valid_preds, digits=4))
    accuracy = accuracy_score(valid_labels, valid_preds)
    return accuracy