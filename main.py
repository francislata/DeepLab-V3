from encoder import Encoder
from dataset import load_cityscapes_datasets
from utils import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
import PIL

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
NUM_CLASSES = 35
NUM_EPOCHS = 5
LR = 1e-3
MOMENTUM = 0.9
TRAIN_IMG_FILEPATH = "../cocostuff/dataset/images/train2017/"
VALID_IMG_FILEPATH = "../cocostuff/dataset/images/val2017/"
TRAIN_ANN_FILEPATH = "../cocostuff/dataset/annotations/train2017/"
VALID_ANN_FILEPATH = "../cocostuff/dataset/annotations/val2017/"
IS_TRAINING_MODE = True

def setup_model(ignore_index):
    """Sets up a model, optimizer, and the loss function to use"""
    encoder_model = Encoder(NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(encoder_model.parameters(), lr=LR, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index).to(DEVICE)

    return encoder_model, optimizer, loss_fn

def train(model, optimizer, loss_fn, train_ds, valid_ds, num_epochs=NUM_EPOCHS):
    """Trains the model"""
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    train_losses = []
    valid_losses = []

    for epoch in range(1, num_epochs + 1):
        print("[Epoch {}] Training and evaluation starts...".format(epoch))

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

    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    with torch.no_grad():
        total_loss = []

        for imgs, anns in tqdm(dl, desc="Validation progress"):
            imgs, anns = imgs.to(DEVICE), anns.long().to(DEVICE)
            outputs = model(imgs)
            total_loss.append(loss_fn(outputs, anns).item())

        loss = sum(total_loss) / len(total_loss)
        print("Evaluation loss is {:.2f}".format(loss))

        return loss

def visualize_prediction(model, img, ann):
    model.eval()
    img_transforms = transforms.ToPILImage()

    with torch.no_grad():
        img, ann = img.unsqueeze(0).to(DEVICE), ann.float()
        pred = torch.argmax(model(img), dim=1)
        pred_img = img_transforms(pred.cpu().float())
        ann_img = img_transforms(ann)
        pred_img.save("prediction.png")
        ann_img.save("annotation.png")


if __name__ == "__main__":
    # Setting seed number for reproducibility
    torch.manual_seed(42)

    print("Loading datasets...")
    train_ds, valid_ds, ignore_index = load_cityscapes_datasets()
    print("Done!\n")

    print("Setting up model, optimizer, and loss function...")
    model, optimizer, loss_fn = setup_model(ignore_index)
    print("Done!\n")

    # load_model_checkpoint(model, optimizer, "deeplabv3_epoch15_lr{}".format(LR))
    # img, ann = valid_ds[0]
    # visualize_prediction(model, img, ann)

    print("Training model...\n")
    train_losses, valid_losses = load_model_checkpoint(model, optimizer, "deeplabv3_epoch15_lr{}".format(LR))
    current_train_losses, current_valid_losses = train(model, optimizer, loss_fn, train_ds, valid_ds)
    train_losses.extend(current_train_losses)
    valid_losses.extend(current_valid_losses)
    plot_save_losses(train_losses, valid_losses, "Training and Validation Losses", "losses_epoch20_lr{}".format(LR))
    save_model_checkpoint(model, optimizer, train_losses, valid_losses, "deeplabv3_epoch20_lr{}".format(LR))
    print("Done!")
