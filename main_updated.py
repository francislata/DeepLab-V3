from scheduler import PolynomialLRScheduler
from encoder import Encoder
from dataset import load_cityscapes_datasets
from utils import save_model_checkpoint, load_model_checkpoint, plot_save_losses
from tqdm import tqdm
from torch.utils.data import DataLoader
# from cityscapesScripts.cityscapesscripts.evaluation import evalPixelLevelSemanticLabeling
import torchvision.transforms as transforms
import argparse
import torch.optim as optim
import torch.nn as nn
import torch
import os

# Constants
LEARNING_RATE = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SAVE_CHECKPOINT_INDEX = 5
TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, EVAL_METRIC_BATCH_SIZE = 8, 8, 1
NUM_EPOCHS = 30
IGNORE_INDEX = 255
NUM_CLASSES = 19
RESULTS_PATH = "./cityscapes-results/{}"
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_model(args):
    """Sets up the model"""
    model = Encoder(args.num_classes).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=args.ignore_index).to(DEVICE)

    return model, optimizer, loss_fn

def train(model, optimizer, loss_fn, train_ds, valid_ds, start_epoch, num_epochs, lr_scheduler, args, train_losses=[], valid_losses=[]):
    """Trains the model"""
    train_dl = DataLoader(train_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=NUM_WORKERS)

    for epoch in range(start_epoch, num_epochs):
        print("[Epoch {}] Training and evaluation starts...".format(epoch))

        torch.cuda.empty_cache()
        
        model.train()
        total_loss = []

        for idx, (imgs, anns) in enumerate(tqdm(train_dl, desc="Training progress")):
            lr_scheduler.step(epoch - 1, idx)
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

        valid_losses.append(evaluate(model, loss_fn, valid_ds, args))

        if epoch % args.save_checkpoint_index == 0:
            print("[Epoch {}] Calculating mIoU...".format(epoch))
            # TODO: Fix this issue
#             evaluate_save_predictions(model, valid_ds, args)
#             mIoU = evalPixelLevelSemanticLabeling.main()
            mIoU = 0.0
            save_model_checkpoint(model, optimizer, train_losses, valid_losses, num_epochs, "deeplabv3_epoch{}_lr{}_wd{}_mIoU{:.3f}".format(epoch, args.learning_rate, args.weight_decay, mIoU))
            plot_save_losses(train_losses, valid_losses, "Training and Validation Losses", "deeplabv3_epoch{}_lr{}_wd{}_mIoU{:.3f}".format(epoch, args.learning_rate, args.weight_decay, mIoU))
            print("[Epoch {}] mIoU is {:.3f}\n".format(epoch, mIoU))

        print("[Epoch {}] Training and evaluation complete!\n".format(epoch))

def evaluate(model, loss_fn, dataset, args):
    """Evaluates the model"""
    model.eval()

    dl = DataLoader(dataset, batch_size=args.eval_batch_size, num_workers=NUM_WORKERS)

    with torch.no_grad():
        torch.cuda.empty_cache()

        total_loss = []

        for imgs, anns in tqdm(dl, desc="Validation progress"):
            imgs, anns = imgs.to(DEVICE), anns.long().to(DEVICE)
            outputs = model(imgs)
            total_loss.append(loss_fn(outputs, anns).item())

        loss = sum(total_loss) / len(total_loss)
        print("Evaluation loss is {:.2f}\n".format(loss))

        return loss

def evaluate_save_predictions(model, dataset, args):
    """Evaluates the model with the given dataset and saves the entries in the given filepath"""
    model.eval()

    dl = DataLoader(dataset, batch_size=EVAL_METRIC_BATCH_SIZE, num_workers=NUM_WORKERS)
    img_transform = transforms.ToPILImage()

    with torch.no_grad():
        torch.cuda.empty_cache()

        for idx, (imgs, _) in enumerate(tqdm(dl, desc="Evaluation progress")):
            imgs = imgs.to(DEVICE)
            output = model(imgs)
            output = dataset.convert_train_id_to_id(torch.argmax(output, dim=1)).cpu().int()
            output = img_transform(output)

            filename = os.path.basename(dataset.images[idx]).replace("_leftImg8bit", "*")
            output.save(args.results_path.format(filename))

def parse_arguments():
    """Handles the command line arguments and proceeds with training or evaluation of the model"""
    args_parser = argparse.ArgumentParser(description="DeepLab V3 Training and Evaluation")

    # Required arguments
    args_parser.add_argument("dataset_folder", type=str, help="the root folder of the dataset")
    args_parser.add_argument("model_checkpoint_folder", type=str, help="the checkpoint folder for the model")
    args_parser.add_argument("losses_folder", type=str, help="the folder to store the losses graphs")

    # Optional arguments
    args_parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="learning rate to use for SGD optimizer")
    args_parser.add_argument("--momentum", type=float, default=MOMENTUM, help="momentum to use for SGD optimizer")
    args_parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="L2 regularization to use for SGD optimizer")
    args_parser.add_argument("--train-batch-size", type=int, default=TRAIN_BATCH_SIZE, help="the batch size to use when training the model")
    args_parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE, help="the batch size to use when evaluating the model")
    args_parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS, help="the total number of epochs to train the model for")
    args_parser.add_argument("--save-checkpoint-index", type=int, default=SAVE_CHECKPOINT_INDEX, help="the number when to make a checkpoint of the model")
    args_parser.add_argument("--is-training-model", action="store_true", help="indicates whether to train a model")
    args_parser.add_argument("--is-evaluating-model", action="store_true", help="indicates whether to evaluate a model")
    args_parser.add_argument("--ignore-index", type=int, default=IGNORE_INDEX, help="the ignore index to use in the dataset")
    args_parser.add_argument("--num-classes", type=int, default=NUM_CLASSES, help="the number of classes in the dataset")
    args_parser.add_argument("--results-path", type=str, default=RESULTS_PATH, help="the path to put the model's predictions at")
    args_parser.add_argument("--checkpoint-path", type=str, help="the path to a model checkpoint")

    return args_parser.parse_args()

def main(args):
    """Performs the necessary operations once the arguments have been parsed"""
    # For reproducibility
    torch.manual_seed(42)

    print("Loading datasets...")
    train_ds, valid_ds = load_cityscapes_datasets(args.dataset_folder)
    print("Done!\n")

    print("Setting up model, optimizer, and loss function...")
    model, optimizer, loss_fn = setup_model(args)
    print("Done!\n")

    if args.checkpoint_path:
        print("Loading checkpoint...")
        start_epoch, train_losses, valid_losses = load_model_checkpoint(model, optimizer, args.checkpoint_path)
        print("Done!\n")

    if args.is_training_model:
        if args.checkpoint_path is None:
            start_epoch, train_losses, valid_losses = 1, [], []

        num_epochs = start_epoch + args.num_epochs

        print("Setting up learning rate scheduler...")
        lr_scheduler = PolynomialLRScheduler(optimizer, args.learning_rate, len(train_ds), num_epochs - 1)
        print("Done!\n")

        print("Training model...\n")
        train(model, optimizer, loss_fn, train_ds, valid_ds, start_epoch, num_epochs, lr_scheduler, args, train_losses=train_losses, valid_losses=valid_losses)
        print("Done!")
    elif args.is_evaluating_model:
        print("Evaluating model on the validation set...")
        evaluate_save_predictions(model, valid_ds, args)
        print("Done!\n")

        print("Calculating mIoU...")
        # mIoU = evalPixelLevelSemanticLabeling.main()
        mIoU = 0.0
        print("mIoU is {:.3f}".format(mIoU))

if __name__ == "__main__":
    main(parse_arguments())
