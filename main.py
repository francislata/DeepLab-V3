import torch
import torch.optim as optim
import torch.nn as nn
from encoder import Encoder
from dataset import ImageSegmentationDataset
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_CLASSES = 172
NUM_EPOCHS = 3
LR = 1e-3
TRAIN_IMG_FILEPATH = "../cocostuff/dataset/images/train2017/"
VALID_IMG_FILEPATH = "../cocostuff/dataset/images/val2017/"
TRAIN_ANN_FILEPATH = "../cocostuff/dataset/annotations/train2017/"
VALID_ANN_FILEPATH = "../cocostuff/dataset/annotations/val2017/"
MODEL_CHECKPOINT_FILEPATH = "checkpoints/"

def setup_model():
    """Sets up a model, optimizer, and the loss function to use"""
    encoder_model = Encoder(NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(encoder_model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    return encoder_model, optimizer, loss_fn

def load_datasets():
    """Loads the training and validation datasets"""
    train_ds = ImageSegmentationDataset(TRAIN_IMG_FILEPATH, annotations_filepath=TRAIN_ANN_FILEPATH)
    valid_ds = ImageSegmentationDataset(VALID_IMG_FILEPATH, annotations_filepath=VALID_ANN_FILEPATH)

    return train_ds, valid_ds

def train(model, optimizer, loss_fn, train_ds, valid_ds, num_epochs=NUM_EPOCHS):
    """Trains the model"""
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, num_workers=4)

    for epoch in range(1, num_epochs + 1):
        print("[Epoch {}] Training and evaluation starts...".format(epoch))

        model.train()
        total_loss = []

        for imgs, anns in tqdm(train_dl, desc="Progress"):
            imgs, anns = imgs.to(DEVICE), anns.squeeze().to(DEVICE)
            output = model(imgs)

            loss = loss_fn(output, anns)
            total_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        print("[Epoch {}] Training loss is {:.2f}".format(epoch, sum(total_loss) / len(total_loss)))

        # model.eval()
        # total_loss = []

        # with torch.no_grad():
        #     for imgs, anns in tqdm(valid_dl, desc="Progress"):
        #         imgs, anns = imgs.to(DEVICE), anns.squeeze().to(DEVICE)
        #         output = model(imgs)

        #         loss = loss_fn(output, anns)
        #         total_loss.append(loss.item())

        # print("[Epoch {}] Validation loss is {:.2f}".format(epoch, sum(total_loss) / len(total_loss)))
        print("[Epoch {}] Training and evaluation complete!\n".format(epoch))

        model_checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        torch.save(model_checkpoint, "{}deeplabv3_epoch{}_sgd_lr{}.pth".format(MODEL_CHECKPOINT_FILEPATH, epoch, LR))

def evaluate(model, optimizer, filepath):
    model_checkpoint = torch.load(filepath)
    model.load_state_dict(model_checkpoint["model"])
    optimizer.load_state_dict(model_checkpoint["optimizer"])

    valid_dl = DataLoader(valid_ds, batch_size=BATCH_SIZE, num_workers=4)

    model.eval()
    total_loss = []

    with torch.no_grad():
        for imgs, anns in tqdm(valid_dl, desc="Progress"):
            imgs, anns = imgs.to(DEVICE), anns.squeeze().to(DEVICE)
            output = model(imgs)

            loss = loss_fn(output, anns)
            total_loss.append(loss.item())

        print("Loss is {:.2f}".format(sum(total_loss) / len(total_loss)))
            

if __name__ == "__main__":
    print("Setting up model, optimizer, and loss function...")
    model, optimizer, loss_fn = setup_model()
    print("Done!\n")

    print("Loading datasets...")
    train_ds, valid_ds = load_datasets()
    print("Done!\n")

    print("Training and evaluating model...\n")
    # train(model, optimizer, loss_fn, train_ds, valid_ds)
    evaluate(model, optimizer, "{}deeplabv3_epoch3_sgd_lr{}.pth".format(MODEL_CHECKPOINT_FILEPATH, LR))
    print("Done!")
