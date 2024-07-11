import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from data import corrupt_mnist
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=32, help="batch size to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")
@click.option("--bucket_name", required=True, help="Google Cloud Storage bucket name to save the files")
def train(lr, batch_size, epochs, bucket_name) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # 使用挂载的GCS路径
    gcs_path = f"/gcs/{bucket_name}/data/corruptmnist"
    train_set, _ = corrupt_mnist(gcs_path)

    model = MyAwesomeModel().to(DEVICE)

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    print("Training complete")
    model_save_path = f"/gcs/{bucket_name}/trained_model.pt"
    torch.save(model.state_dict(), model_save_path)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    
    fig_save_path = f"/gcs/{bucket_name}/training_statistics.png"
    fig.savefig(fig_save_path)

@click.command()
@click.argument("model_checkpoint")
@click.option("--bucket_name", required=True, help="Google Cloud Storage bucket name to load the files")
def evaluate(model_checkpoint, bucket_name) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    # 从GCS加载模型
    model_load_path = f"/gcs/{bucket_name}/{model_checkpoint}"
    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_load_path))

    # 使用挂载的GCS路径
    gcs_path = f"/gcs/{bucket_name}/data/corruptmnist"
    _, test_set = corrupt_mnist(gcs_path)
    
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")

cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
