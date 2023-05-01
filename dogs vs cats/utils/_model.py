import torch
import os
import numpy as np


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(self, output_dir: str, model_name: str) -> None:
        self.best_valid_loss = float(np.Inf)
        self.best_valid_f1 = float(np.NINF)
        self.output_dir = output_dir
        self.model_name = model_name

    def __call__(
        self,
        current_valid_loss: float,
        current_valid_f1: float,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim,
    ) -> None:
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            self.best_valid_f1 = current_valid_f1
            print("\nSaving model...")
            print(f"Epoch: {epoch}")
            print(f"Validation F1-Score: {current_valid_f1:1.6f}")
            print(f"Validation Loss: {current_valid_loss:1.6f}\n")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                os.path.join(self.output_dir, f"{self.model_name}.pth"),
            )


class EarlyStopping:
    """
    The Early Stopping class (used to avoid overfitting during training).
    """

    def __init__(self, tolerance: int) -> None:
        self.tolerance = tolerance
        self.early_stop = False
        self.counter = 0
        self.best_loss = None

    def __call__(self, validation_loss) -> None:
        if self.best_loss is None:
            self.best_loss = validation_loss
        elif validation_loss >= self.best_loss:
            self.counter += 1

            if self.counter >= self.tolerance:
                self.early_stop = True
        else:
            self.best_loss = validation_loss
            self.counter = 0
