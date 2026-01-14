import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from mlops_mnist.model import Model  # Adjust import based on your folder structure


def test_model():
    """
    Runs a single batch of training, validation, and testing 
    to ensure all model methods (steps, logging, optimizers) are reachable.
    """
    model = Model()

    data = torch.randn(10, 1, 28, 28)
    targets = torch.randint(0, 10, (10,))
    dataset = TensorDataset(data, targets)
    dataloader = DataLoader(dataset, batch_size=2)

    trainer = Trainer(
        fast_dev_run=True,  
        accelerator="cpu",  
        devices=1,
        logger=False        
    )

    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)
    trainer.test(model, dataloaders=dataloader)