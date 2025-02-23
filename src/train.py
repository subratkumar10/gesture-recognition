import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from CustomDataset import CustomRawDataset
from model_dispatcher import dispatch_model
# from model_dispatcher_cnn import dispatch_model
import config
from torch import nn
import os
from glob import glob
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import date, timedelta, datetime
import time
import random
from torch.utils.tensorboard import SummaryWriter
from CustomDataset import custom_collate_fn
random.seed(42)
def return_dataloader(type_of_data):
    return DataLoader(dataset=CustomRawDataset(type_of_data=type_of_data), shuffle=True, batch_size=config.BATCH_SIZE, collate_fn=custom_collate_fn)

def num_correct_classifications(Y_prob, Y_true):
    Y_pred = torch.argmax(Y_prob, dim = 1)
    return (Y_pred == Y_true).sum().float()

def loss_for_batch(model, X, Y_true, criterion, optimizer = None):
    Y_pred = model(X)
    loss = criterion(Y_pred, Y_true)
    Y_prob = nn.Softmax(dim = 1)(Y_pred)

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), num_correct_classifications(Y_prob, Y_true)

def save_best_model(path_of_saved_model, epoch, current_loss, previous_loss, name_of_model, model, face_model = False):
    if current_loss < previous_loss:
        try:
            os.makedirs(path_of_saved_model)
        except (OSError, FileExistsError) as e:
            pass
        finally:
            if epoch == 1:
                saved_models = glob(os.path.join(path_of_saved_model, "*.pt"))
                if len(saved_models) > 0:
                    print("Cleaning the Directory as Models of Previous Runs are Present")
                    for model_path in saved_models:
                        os.remove(model_path)
                        print("Model File: {} Removed".format(model_path))
                    print("Directory Cleaned, Resuming with the Training")
            torch.save(model, os.path.join(path_of_saved_model, str(epoch) + '.pt'))
            print("Validation Loss decreased from {} to {}, Saving the Updated Model".format(previous_loss,current_loss))
            return current_loss
    else:
        return previous_loss

def fit(name_of_model, path_of_saved_model, logs_path):
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    print("GPU Name: {}".format(torch.cuda.get_device_name(device)))

    train_dl = return_dataloader('train')
    val_dl = return_dataloader('val')
    previous_val_loss = float('inf')
    train_loss = []
    val_loss = []

    model = dispatch_model(name_of_model).to(device)
    # model = dispatch_model(name_of_model)

    optimizer = Adam(model.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(logs_path)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, threshold=1e-5, patience=3)

    print("length train dl",len(train_dl))
    for epoch in range(1, config.NUM_EPOCHS + 1):
        total_train_loss = 0
        total_train_correct = 0
        total_val_loss = 0
        total_val_correct = 0
        num_batches_completed = 0
        num_samples_completed = 0
        optimizer.zero_grad()
        model.train()

        for (X_train, Y_train, X_lens) in train_dl:
            X_train = pack_padded_sequence(X_train, X_lens, batch_first=True, enforce_sorted=True)
            X_train, Y_train = X_train.to(device), Y_train.to(device)
            
            train_loss_batch, train_correct_batch = loss_for_batch(model, X_train, Y_train, criterion, optimizer)
            total_train_loss += train_loss_batch * (Y_train.shape[0])
            total_train_correct += train_correct_batch
            num_batches_completed += 1
            num_samples_completed += Y_train.shape[0]
            print(
                "\rBatch: {}/{} Train Accuracy: {:.2f} Train Loss: {:.4f}".format(
                    num_batches_completed,
                    len(train_dl),
                    total_train_correct / num_samples_completed,
                    train_loss_batch,
                ),
                end='',
            )

        train_accuracy = total_train_correct / len(train_dl.dataset)
        total_train_loss = total_train_loss / len(train_dl.dataset)
        model.eval()

        with torch.no_grad():
            for (X_val, Y_val, X_lens) in val_dl:
                X_val = pack_padded_sequence(X_val, X_lens, batch_first=True)
                X_val, Y_val = X_val.to(device), Y_val.to(device)
            
                val_loss_batch, val_correct_batch = loss_for_batch(model, X_val, Y_val, criterion)
                total_val_loss += val_loss_batch * (Y_val.shape[0])
                total_val_correct += val_correct_batch

            val_accuracy = total_val_correct / len(val_dl.dataset)
            total_val_loss = total_val_loss / len(val_dl.dataset)

            train_loss.append(total_train_loss)
            val_loss.append(total_val_loss)
            writer.add_scalars(
                "Loss",
                {
                    "Train Loss": total_train_loss,
                    "Validation Loss": total_val_loss
                },
                epoch
            )
            writer.add_scalars(
                "Accuracy",
                {
                    "Train Accuracy": train_accuracy,
                    "Validation Accuracy": val_accuracy
                },
                epoch
            )

            print(
                "\nEpoch({}/{}) Train Loss: {:.4f} Validation Loss: {:.4f} Train Accuracy: {:.2f} Validation "
                "Accuracy: {:.2f}".format(
                    epoch,
                    config.NUM_EPOCHS,
                    total_train_loss,
                    total_val_loss,
                    train_accuracy,
                    val_accuracy,
                )
            )
        previous_val_loss = save_best_model(path_of_saved_model, epoch, total_val_loss, previous_val_loss,
                                            name_of_model, model)
        scheduler.step(previous_val_loss)
    writer.add_graph(model, torch.rand(config.BATCH_SIZE, config.NUM_FRAMES, config.EXTRACTED_FEATURES, device=device))
    writer.flush()
    writer.close()
    return

def main():
    start_time = time.monotonic()
    name_of_model = 'CustomMotionModel'
    path_of_saved_model = os.path.join(
        config.MODELS_PATH, name_of_model,
        date.today().strftime("%d-%m-%Y"),
        datetime.now().strftime("%H-%M-%S")
    )
    logs_path = os.path.join(
        config.LOGS_PATH,
        date.today().strftime("%d-%m-%Y"),
        datetime.now().strftime("%H-%M-%S")
    )
    fit(name_of_model, path_of_saved_model, logs_path)
    end_time = time.monotonic()
    print("-" * 20)
    print("Time Elapsed {}".format(timedelta(seconds=end_time - start_time)))

if __name__ == "__main__":
    main()



