import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np
import random
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from typing import Tuple
from torch.utils.data import DataLoader
from models.resnet import ResNet
from utils._model import SaveBestModel, EarlyStopping
from utils._dataset import read_data, create_dataloader
from torch.optim.lr_scheduler import StepLR

warnings.filterwarnings("ignore")
# Making sure the experiments are reproducible
seed = 2109
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def train(
    model: nn.Module,
    loss: nn.CrossEntropyLoss,
    device: torch.device,
    dataloader: DataLoader,
    optimizer: torch.optim,
) -> Tuple[float, float]:
    """
    Function responsible for training the model.

    Args:
        model (nn.Module): the model architecture.
        loss (nn.CrossEntropyLoss): the loss function which is being used.
        device (torch.device): the device that is being used (cpu or cuda).
        dataloader (DataLoader): the train dataloader.
        optimizer (torch.optim): the optimizer which is being used.

    Returns:
        Tuple[float, float]: the train loss and f1 score.
    """
    model.train()
    train_loss = 0.0
    train_f1 = 0.0

    for batch in dataloader:
        data, target = batch[0], batch[1]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        l = loss(output, target)
        l.backward()
        optimizer.step()

        train_loss += l.item()
        prediction = F.softmax(output, dim=-1)
        prediction = prediction.argmax(dim=1, keepdim=True).squeeze(-1)

        train_f1 += f1_score(
            y_true=target.detach().cpu().numpy(),
            y_pred=prediction.detach().cpu().numpy(),
            average="weighted",
        )

    train_loss /= len(dataloader)
    train_f1 /= len(dataloader)
    return train_loss, train_f1


def test(
    model: nn.Module,
    loss: nn.CrossEntropyLoss,
    device: torch.device,
    dataloader: DataLoader,
) -> Tuple[float, float]:
    """
    Function responsible for validating/testing the model whilst
    is being trained.

    Args:
        model (nn.Module): the model architecture.
        loss (nn.CrossEntropyLoss): the loss function which is being used.
        device (torch.device): the device that is being used (cpu or cuda).
        dataloader (DataLoader): the test dataloader.

    Returns:
        Tuple[float, float]: the test loss and f1 score.
    """
    model.eval()
    test_loss = 0.0
    test_f1 = 0.0

    with torch.inference_mode():
        for batch in dataloader:
            data, target = batch[0], batch[1]
            data, target = data.to(device), target.to(device)

            output = model(data)
            l = loss(output, target)
            test_loss += l.item()
            prediction = F.softmax(output, dim=-1)
            prediction = prediction.argmax(dim=1, keepdim=True).squeeze(-1)

            test_f1 += f1_score(
                y_true=target.detach().cpu().numpy(),
                y_pred=prediction.detach().cpu().numpy(),
                average="weighted",
            )

    test_loss /= len(dataloader)
    test_f1 /= len(dataloader)
    return test_loss, test_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str.lower, required=True)
    parser.add_argument("--scheduler_gamma", type=float)
    parser.add_argument("--tolerance", type=int)
    parser.add_argument("--scheduler_step", type=int)
    parser.set_defaults(epochs=50, lr=1e-03, batch_size=32, scheduler_gamma=0.001, tolerance=5, scheduler_step=5)
    args = parser.parse_args()
    
    # Validating the passed arguments
    assert os.path.exists(args.input_dir), "Please enter a valid input directory."
    assert args.epochs > 0, "Please enter a valid value for epochs."
    assert args.lr > 0, "Please enter a valid value for learning rate."
    assert args.batch_size > 0, "Please enter a valid value for batch size."
    assert args.scheduler_gamma > 0, "Please enter a valid value for scheduler gamma."
    assert args.scheduler_step > 0, "Please enter a valid value for scheduler step."
    assert (
        args.tolerance > 0
    ), "Please enter a valid value for early stopping tolerance."
    assert (
        args.model_name == "resnet"
    ), f"Please enter a valid model name."
    os.makedirs(args.output_dir, exist_ok=True)

    # Reading the data
    df = read_data(
        input_dir=f"{args.input_dir}/train"
    )
    
    X = df["data"].values.tolist()
    y = df["label"].values.tolist()
    
    # Splitting the data into training and validation
    X_train, X_validation, y_train, y_validation = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        shuffle=True,
        random_state=seed
    )

    # Creating the model
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")

    # Creating the data loaders
    train_dataloader = create_dataloader(
        X=X_train,
        y=y_train,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False
    )
    
    test_dataloader = create_dataloader(
        X=X_validation,
        y=y_validation,
        batch_size=args.batch_size,
        num_workers=0,
        shuffle=False
    )
    
    model = ResNet().to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, eps=1e-07)
    loss = nn.CrossEntropyLoss()
    sbm = SaveBestModel(output_dir=args.output_dir, model_name=args.model_name)
    es = EarlyStopping(tolerance=args.tolerance)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)

    # Creating the log path
    log_path = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_path, exist_ok=True)
    training_log = pd.DataFrame()

    # Training step
    for epoch in range(1, args.epochs + 1):
        print(f"\nTraining epoch {epoch}/{args.epochs}")

        train_loss, train_f1 = train(
            model=model,
            loss=loss,
            device=device,
            dataloader=train_dataloader,
            optimizer=optimizer,
        )

        test_loss, test_f1 = test(
            model=model, loss=loss, device=device, dataloader=test_dataloader
        )

        scheduler.step()

        # Saving the best model
        sbm(
            current_valid_f1=test_f1,
            current_valid_loss=test_loss,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
        )

        log = pd.DataFrame(
            {
                "train_f1": [train_f1],
                "train_loss": [train_loss],
                "test_f1": [test_f1],
                "test_loss": [test_loss],
                "epoch": [epoch],
            }
        )

        # Checking early stopping criterion
        es(validation_loss=test_loss)

        if es.early_stop:
            break

        training_log = pd.concat([training_log, log], axis=0)

    training_log.to_csv(f"{log_path}/log_{args.model_name}.csv", index=False, sep=";")
