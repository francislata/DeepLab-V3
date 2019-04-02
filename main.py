from encoder import Encoder
from dataset import ImageSegmentationDataset
from utils import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_CLASSES = 182
NUM_EPOCHS = 25
LR = 1e-2
MOMENTUM = 0.8
WD = 1e-3
TRAIN_IMG_FILEPATH = "../cocostuff/dataset/images/train2017/"
VALID_IMG_FILEPATH = "../cocostuff/dataset/images/val2017/"
TRAIN_ANN_FILEPATH = "../cocostuff/dataset/annotations/train2017/"
VALID_ANN_FILEPATH = "../cocostuff/dataset/annotations/val2017/"
IS_TRAINING_MODE = True

def setup_model():
    """Sets up a model, optimizer, and the loss function to use"""
    encoder_model = Encoder(NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(encoder_model.parameters(), lr=LR, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss(ignore_index=255).to(DEVICE)

    return encoder_model, optimizer, loss_fn

def load_datasets(max_sample_size=None):
    """Loads the training and validation datasets"""
    train_ds = ImageSegmentationDataset(NUM_CLASSES, TRAIN_IMG_FILEPATH, anns_filepath=TRAIN_ANN_FILEPATH, max_sample_size=max_sample_size)
    valid_ds = ImageSegmentationDataset(NUM_CLASSES, VALID_IMG_FILEPATH, anns_filepath=VALID_ANN_FILEPATH, max_sample_size=max_sample_size)

    return train_ds, valid_ds

def train(model, optimizer, loss_fn, train_ds, valid_ds, num_epochs=NUM_EPOCHS):
    """Trains the model"""
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    train_losses = []
    valid_losses = []

    for epoch in range(1, num_epochs + 1):
        print("[Epoch {}] Training and evaluation starts...".format(epoch))

        model.train()
        total_loss = []

        for imgs, anns in tqdm(train_dl, desc="Progress"):
            optimizer.zero_grad()

            imgs, anns = imgs.to(DEVICE), anns.long().to(DEVICE)
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

    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    with torch.no_grad():
        total_loss = []

        for imgs, anns in tqdm(dl, desc="Progress"):
            imgs, anns = imgs.to(DEVICE), anns.long().to(DEVICE)
            outputs = model(imgs)
            total_loss.append(loss_fn(outputs, anns).item())

        loss = sum(total_loss) / len(total_loss)
        print("Evaluation loss is {:.2f}".format(loss))

        return loss


if __name__ == "__main__":
    # Setting seed number for reproducibility
    torch.manual_seed(42)
    torch.cuda.empty_cache()

    print("Setting up model, optimizer, and loss function...")
    model, optimizer, loss_fn = setup_model()
    print("Done!\n")

    print("Loading datasets...")
    train_ds, valid_ds = load_datasets(max_sample_size=100)
    print("Done!\n")

    if IS_TRAINING_MODE:
        print("Training model...\n")
        # losses = load_model_checkpoint(model, optimizer, "deeplabv3_epoch20_lr{}".format(1e-3))
        train_losses, valid_losses = train(model, optimizer, loss_fn, train_ds, valid_ds)
        plot_save_losses(train_losses, valid_losses, "Training and Validation Losses", "losses_epoch25_lr{}".format(LR))
        save_model_checkpoint(model, optimizer, train_losses, valid_losses, "deeplabv3_epoch25_lr{}".format(LR))
    else:
        print("Evaluating model on validation set...\n")
        load_model_checkpoint(model, optimizer, "deeplabv3_epoch50_lr{}_test".format(LR))
        evaluate(model, loss_fn, valid_ds)

    print("Done!")
