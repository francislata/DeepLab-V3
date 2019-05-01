from encoder import Encoder
from dataset import load_cityscapes_datasets
from utils import save_model_checkpoint, load_model_checkpoint, plot_save_losses, create_lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from cityscapesScripts.cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling import main
from lr_scheduler import PolynomialLR
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
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
EVAL_SAVE_BATCH_SIZE = 1
NUM_CLASSES = 19
NUM_EPOCHS = 5
NUM_WARMUP_EPOCHS = 3
LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
IS_TRAINING_MODEL = True

def setup_model(ignore_index):
    """Sets up the model"""
    encoder_model = Encoder(NUM_CLASSES).to(DEVICE)
    optimizer = optim.SGD(encoder_model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
#     lr_scheduler = PolynomialLR(optimizer, NUM_EPOCHS, NUM_WARMUP_EPOCHS, LR, WARMUP_LR)
    lr_scheduler = None
    loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index).to(DEVICE)

    return encoder_model, optimizer, lr_scheduler, loss_fn

def train(model, optimizer, loss_fn, train_ds, valid_ds, start_epoch=1, num_epochs=NUM_EPOCHS, lr_scheduler=None):
    """Trains the model"""
    train_dl = DataLoader(train_ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
    train_losses = []
    valid_losses = []

    for epoch in range(start_epoch, num_epochs + 1):
        print("[Epoch {}] Training and evaluation starts...".format(epoch))

        torch.cuda.empty_cache()

        if lr_scheduler is not None:
            lr_scheduler.step()
        
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

        print("[Epoch {}] Training loss is {:.2f}\n".format(epoch, epoch_loss))

        valid_losses.append(evaluate(model, loss_fn, valid_ds))

        if epoch % 5 == 0:
            print("[Epoch {}] Calculating mIoU...".format(epoch))
            evaluate_save_predictions(model, valid_ds, CITYSCAPES_RESULTS_FILEPATH)
            mIoU = main()
            print("[Epoch {}] mIoU is {:.3f}\n".format(epoch, mIoU))
            save_model_checkpoint(model, optimizer, train_losses, valid_losses, num_epochs, "deeplabv3_epoch{}_lr{}_mIoU{:.3f}".format(epoch, LR, mIoU), lr_scheduler=lr_scheduler)

        print("[Epoch {}] Training and evaluation complete!\n".format(epoch))

    return train_losses, valid_losses

def evaluate(model, loss_fn, dataset):
    """Evaluates the model"""
    model.eval()

    dl = DataLoader(dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=4)

    with torch.no_grad():
        torch.cuda.empty_cache()

        total_loss = []

        for imgs, anns in tqdm(dl, desc="Validation progress"):
            imgs, anns = imgs.to(DEVICE), anns.long().to(DEVICE)
            outputs = model(imgs)
            total_loss.append(loss_fn(outputs, anns).item())

        print("Evaluation loss is {:.2f}\n".format(sum(total_loss) / len(total_loss)))

        return sum(total_loss) / len(total_loss)

def evaluate_save_predictions(model, dataset, filepath):
    """Evaluates the model with the given dataset and saves the entries in the given filepath"""
    model.eval()

    dl = DataLoader(dataset, batch_size=EVAL_SAVE_BATCH_SIZE, num_workers=4)
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
    
#     img, lbl = train_ds[50]
#     pil_img_trans = transforms.ToPILImage()
#     img, lbl = pil_img_trans(img), pil_img_trans(lbl)
#     img.save("input.png")
#     lbl.save("lbl.png")

    print("Setting up model, optimizer, learning rate scheduler, and loss function...")
    model, optimizer, lr_scheduler, loss_fn = setup_model(ignore_index)
    print("Done!\n")

    if IS_TRAINING_MODEL:
        print("Training model...\n")
        start_epoch, train_losses, valid_losses = load_model_checkpoint(model, optimizer, "deeplabv3_epoch10_lr0.001", lr_scheduler)
#         start_epoch = 0
#         train_losses, valid_losses = [], []
        num_epochs = start_epoch + NUM_EPOCHS
        start_epoch += 1
        current_train_losses, current_valid_losses = train(model, optimizer, loss_fn, train_ds, valid_ds, start_epoch=start_epoch, num_epochs=num_epochs, lr_scheduler=lr_scheduler)
        train_losses.extend(current_train_losses)
        valid_losses.extend(current_valid_losses)
        plot_save_losses(train_losses, valid_losses, "Training and Validation Losses", "losses_epoch{}_lr{}".format(num_epochs, LR))
        save_model_checkpoint(model, optimizer, train_losses, valid_losses, num_epochs, "deeplabv3_epoch{}_lr{}".format(num_epochs, LR), lr_scheduler=lr_scheduler)
        print("Done!")
    else:
        print("Loading pretrained model...")
        load_model_checkpoint(model, optimizer, "deeplabv3_epoch10_lr{}".format(LR))
        print("Done!\n")

        evaluate_save_predictions(model, valid_ds, CITYSCAPES_RESULTS_FILEPATH)
        mIoU = main()
        print("mIoU is {:.3f}\n".format(mIoU))
