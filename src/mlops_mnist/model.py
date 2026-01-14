from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.nn import functional as F
from torchmetrics.functional import accuracy


class Model(LightningModule):
    def __init__(self, lr: float = 1e-3, num_classes: int = 10) -> None:
        super().__init__()
        self.save_hyperparameters()

        conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        self.convs = nn.Sequential(
            conv1,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            conv2,
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = accuracy(preds, target, task='multiclass', num_classes=self.hparams.num_classes)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch) -> None:
        loss, acc = self._shared_eval_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch):
        loss, acc = self._shared_eval_step(batch)
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return loss
    
    def _shared_eval_step(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y, task='multiclass', num_classes=self.hparams.num_classes)
        return loss, acc
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)