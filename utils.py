import torch
import matplotlib; matplotlib.use("Agg")
from matplotlib import pyplot as plt
from PIL import Image
from torch.optim.lr_scheduler import StepLR

# Constants
MODEL_KEY = "model"
OPTIMIZER_KEY = "optimizer"
TRAIN_LOSSES_KEY = "train_losses"
VALID_LOSSES_KEY = "valid_losses"
NUM_EPOCHS_KEY = "num_epochs"
LR_SCHEDULER_KEY = "lr_scheduler"
MODEL_CHECKPOINT_FILEPATH = "checkpoints/"
LOSSES_FILEPATH = "losses/"
MODEL_CHECKPOINT_EXT = ".pth"
LOSSES_EXT = ".png"

def save_model_checkpoint(model, optimizer, train_losses, valid_losses, num_epochs, filename, lr_scheduler=None):
    """Saves a checkpoint of the model"""
    model_checkpoint = {
        MODEL_KEY: model.state_dict(), 
        OPTIMIZER_KEY: optimizer.state_dict(), 
        TRAIN_LOSSES_KEY: train_losses,
        VALID_LOSSES_KEY: valid_losses,
        NUM_EPOCHS_KEY: num_epochs
    }

    if lr_scheduler is not None:
        model_checkpoint[LR_SCHEDULER_KEY] = lr_scheduler.state_dict()

    torch.save(model_checkpoint, MODEL_CHECKPOINT_FILEPATH + filename + MODEL_CHECKPOINT_EXT)

def load_model_checkpoint(model, optimizer, filename, lr_scheduler=None):
    """Loads a checkpoint of the model"""
    model_checkpoint = torch.load(MODEL_CHECKPOINT_FILEPATH + filename + MODEL_CHECKPOINT_EXT)
    model.load_state_dict(model_checkpoint[MODEL_KEY])
    optimizer.load_state_dict(model_checkpoint[OPTIMIZER_KEY])

    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(model_checkpoint[LR_SCHEDULER_KEY])

    return model_checkpoint[NUM_EPOCHS_KEY], model_checkpoint[TRAIN_LOSSES_KEY], model_checkpoint[VALID_LOSSES_KEY]

def plot_save_losses(train_losses, valid_losses, title, filename):
    """Plots the losses and saves the figure"""
    plt.clf()
    plt.plot(train_losses, label="Training")
    plt.plot(valid_losses, label="Validation")
    plt.gca().set_xlabel("Iteration")
    plt.gca().set_ylabel("Loss")
    plt.gca().set_title(title)
    plt.gca().legend()
    plt.savefig(LOSSES_FILEPATH + filename + LOSSES_EXT)

def create_lr_scheduler(optimizer, step_size, gamma):
    """Creates a learning rate scheduler"""
    lr_scheduler = StepLR(optimizer, step_size, gamma=gamma)
    return lr_scheduler
