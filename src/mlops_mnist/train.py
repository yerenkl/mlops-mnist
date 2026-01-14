import torch
import typer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from mlops_mnist.data import corrupt_mnist
from mlops_mnist.model import Model

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

app = typer.Typer()

@app.command()
def train(lr: float = 1e-3, batch_size: int = 64, epochs: int = 5) -> None:
    """Train a model on MNIST."""
    print("Training day and night and noon")
    wandb_logger = WandbLogger(project="corrupt_mnist", log_model=True)

    model = Model().to(DEVICE)
    train_set, test_set = corrupt_mnist()

    # split train_set into training and validation sets
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min", filename="best"
    )

    trainer = Trainer(max_epochs=epochs, accelerator="auto", 
                      devices=1 if torch.cuda.is_available() else None, 
                      logger=wandb_logger, 
                      callbacks=[checkpoint_callback])
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(dataloaders=test_loader, ckpt_path="best")
    trainer.save_checkpoint("best_model.pth", weights_only=True)

if __name__ == "__main__":
    typer.run(train)
