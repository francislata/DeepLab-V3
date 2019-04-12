from encoder import Encoder
from dataset import load_cityscapes_datasets
from utils import save_model_checkpoint, load_model_checkpoint, plot_save_losses
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import PIL

# Constants
CITYSCAPES_ROOT_FILEPATH = "./cityscapes-dataset"
CITYSCAPES_RESULTS_FILEPATH = "./cityscapes-results/{}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
EVAL_SAVE_BATCH_SIZE = 1
NUM_CLASSES = 19
NUM_EPOCHS = 20
LR = 1e-2
MOMENTUM = 0.9
IS_TRAINING_MODEL = False

def setup_model(ignore_index):
    """Sets up a model, optimizer, and the loss function to use"""
    encoder_model = Encoder(NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(encoder_model.parameters(), lr=LR, momentum=MOMENTUM, nesterov=True)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index).to(DEVICE)

    return encoder_model, optimizer, loss_fn

def train(model, optimizer, loss_fn, train_ds, valid_ds, start_epoch=1, num_epochs=NUM_EPOCHS):
    """Trains the model"""
    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
    train_losses = []
    valid_losses = []

    for epoch in range(start_epoch, num_epochs + 1):
        print("[Epoch {}] Training and evaluation starts...".format(epoch))

        torch.cuda.empty_cache()
        
        model.train()
        total_loss = []

        for imgs, anns in tqdm(train_dl, desc="Training progress"):
            optimizer.zero_grad()

            imgs, anns = imgs.to(DEVICE), anns.squeeze().long().to(DEVICE)
            outputs = model(imgs)

            loss = loss_fn(outputs, anns)
            total_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        epoch_loss = sum(total_loss) / len(total_loss)
        train_losses.append(epoch_loss)

        print("[Epoch {}] Training loss is {:.2f}".format(epoch, epoch_loss))

        valid_losses.append(evaluate(model, loss_fn, valid_ds))

        print("[Epoch {}] Training and evaluation complete!\n".format(epoch))

    return train_losses, valid_losses

def evaluate(model, loss_fn, dataset):
    """Evaluates the model"""
    model.eval()

    dl = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, num_workers=4)

    with torch.no_grad():
        torch.cuda.empty_cache()

        total_loss = []

        for imgs, anns in tqdm(dl, desc="Validation progress"):
            imgs, anns = imgs.to(DEVICE), anns.long().to(DEVICE)
            outputs = model(imgs)
            total_loss.append(loss_fn(outputs, anns).item())

        loss = sum(total_loss) / len(total_loss)
        print("Evaluation loss is {:.2f}".format(loss))

        return loss

def evaluate_save_predictions(model, dataset, filepath):
    """Evaluates the model with the given dataset and saves the entries in the given filepath"""
    model.eval()

    dl = DataLoader(dataset, batch_size=EVAL_SAVE_BATCH_SIZE)
    img_transform = transforms.ToPILImage()

    with torch.no_grad():
        torch.cuda.empty_cache()

        for idx, (imgs, _) in enumerate(tqdm(dl, desc="Evaluation progress")):
            imgs = imgs.to(DEVICE)
            output = model(imgs)
            output = dataset.convert_train_id_to_id(torch.argmax(output, dim=1)).cpu().int()
            output = img_transform(output)

            filename = os.path.basename(dataset.images[idx]).replace("_leftImg8bit", "*")
            output.save(CITYSCAPES_RESULTS_FILEPATH.format(filename))

if __name__ == "__main__":
    # Setting seed number for reproducibility
    torch.manual_seed(42)

    print("Loading datasets...")
    train_ds, valid_ds, ignore_index = load_cityscapes_datasets(CITYSCAPES_ROOT_FILEPATH)
    print("Done!\n")

    print("Setting up model, optimizer, and loss function...")
    model, optimizer, loss_fn = setup_model(ignore_index)
    print("Done!\n")

    if IS_TRAINING_MODEL:
        print("Training model...\n")
        # start_epoch, train_losses, valid_losses = load_model_checkpoint(model, optimizer, "deeplabv3_epoch25_lr{}".format(1e-1))
        start_epoch = 0
        train_losses, valid_losses = [], []
        num_epochs = start_epoch + NUM_EPOCHS
        start_epoch += 1
        current_train_losses, current_valid_losses = train(model, optimizer, loss_fn, train_ds, valid_ds, start_epoch=start_epoch, num_epochs=num_epochs)
        train_losses.extend(current_train_losses)
        valid_losses.extend(current_valid_losses)
        plot_save_losses(train_losses, valid_losses, "Training and Validation Losses", "losses_epoch{}_lr{}".format(num_epochs, LR))
        save_model_checkpoint(model, optimizer, train_losses, valid_losses, num_epochs, "deeplabv3_epoch{}_lr{}".format(num_epochs, LR))
        print("Done!")
    else:
        print("Loading pretrained model...")
        load_model_checkpoint(model, optimizer, "deeplabv3_epoch20_lr{}".format(1e-2))
        print("Done!\n")

        evaluate_save_predictions(model, valid_ds, CITYSCAPES_RESULTS_FILEPATH)
