import torch
import typer
from torch.utils.data import DataLoader

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
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    model = Model().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.eval()
    _, test_set = corrupt_mnist()
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to("cuda"), target.to("cuda")
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += (pred == target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_set)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {100 * test_accuracy:.2f}%")


if __name__ == "__main__":
    app()
