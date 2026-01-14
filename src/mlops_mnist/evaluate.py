import torch
import typer
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from mlops_mnist.data import corrupt_mnist
from mlops_mnist.model import Model


app = typer.Typer()

@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    model = Model.load_from_checkpoint(model_checkpoint)

    _, test_set = corrupt_mnist()
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    trainer = Trainer(accelerator="auto", devices=1, logger=False) 
    trainer.test(model, dataloaders=test_loader)

if __name__ == "__main__":
    app()