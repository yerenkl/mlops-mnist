import torch
import typer
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from torchvision.utils import make_grid

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
    print("Training day and night")
    wandb.init(
    project="corrupt_mnist",
    config={"lr": lr, "batch_size": batch_size, "epochs": epochs},)

    # TODO: Implement training loop here
    model = Model().to(DEVICE)
    train_set, _ = corrupt_mnist()

    # split train_set into training and validation sets
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = random_split(train_set, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    results = {"acc_tr": [], "acc_val": [], "loss_tr": [], "loss_val": []}
    num_epochs = 5

    for epoch in tqdm(range(num_epochs), total=num_epochs):  # number of epochs
        train_loss_sum = 0.0
        train_correct = 0
        total_train = 0
        total_val = 0
        val_loss_sum = 0.0
        val_correct = 0

        # training
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss_sum += loss.item() * data.size(0)
            pred = output.argmax(1)
            train_correct += (pred == target.view_as(pred)).sum().item()
            total_train += target.size(0)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, iter {batch_idx}, loss: {loss.item()}")
                
                # 1. Get batch (Keep as [5, 1, 28, 28])
                imgs = data[:5].detach().cpu() 
                
                # 2. Stitch into one image using make_grid
                grid = make_grid(imgs, normalize=True, scale_each=True)
                
                # 3. Log the single grid image
                images = wandb.Image(grid, caption="Input images")
                wandb.log({"images": images})
        avg_train_loss = train_loss_sum / len(train_dataset)
        train_acc = train_correct / total_train

        results["loss_tr"].append(avg_train_loss)
        results["acc_tr"].append(train_acc)

        print(f"Epoch {epoch + 1}, Avg Train Loss: {avg_train_loss:.4f}")
        # validation
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss_sum += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1)
                val_correct += (pred == target.view_as(pred)).sum().item()
                total_val += target.size(0)

        avg_val_loss = val_loss_sum / len(val_dataset)
        val_accuracy = val_correct / len(val_loader.dataset)

        results["loss_val"].append(avg_val_loss)
        results["acc_val"].append(val_accuracy)
        print(
            "Epoch [{}/{}] | Train Loss: {:.4f} | Val Loss: {:.4f} | Train Acc: {:.2f}% | Val Acc: {:.2f}%".format(
                epoch + 1,
                num_epochs,
                avg_train_loss,
                avg_train_loss,
                100 * train_acc,
                100 * val_accuracy,
            )
        )
        wandb.log({"train_loss": avg_train_loss, "train_accuracy": train_acc, "val_loss": avg_val_loss, "val_accuracy": val_accuracy})

    torch.save(model.state_dict(), "./models/model.pth")
    print("Model saved to ./models/model.pth")


if __name__ == "__main__":
    typer.run(train)
