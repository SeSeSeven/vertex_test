import click
import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from data import corrupt_mnist
from google.cloud import storage

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

@click.group()
def cli():
    """Command line interface."""
    pass

@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
@click.option("--batch_size", default=32, help="batch size to use for training")
@click.option("--epochs", default=10, help="number of epochs to train for")
@click.option("--bucket_name", help="Google Cloud Storage bucket name to save the files")
def train(lr, batch_size, epochs, bucket_name) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    # Uncomment the lines below to use DVC for data versioning
    # import dvc.api
    # data_path = dvc.api.get_url("data/corruptmnist", repo="path/to/repo")

    # Download data from GCS to local filesystem
    for i in range(5):  # assuming there are 5 training files
        download_from_gcs(bucket_name, f"data/corruptmnist/train_{i}.pt", f"data/corruptmnist/train_{i}.pt")
    download_from_gcs(bucket_name, "data/corruptmnist/test.pt", "data/corruptmnist/test.pt")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

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
    torch.save(model.state_dict(), "trained_model.pt")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("training_statistics.png")

    # Upload files to GCS
    if bucket_name:
        upload_to_gcs(bucket_name, "trained_model.pt", "trained_model.pt")
        upload_to_gcs(bucket_name, "training_statistics.png", "training_statistics.png")

@click.command()
@click.argument("model_checkpoint")
@click.option("--bucket_name", help="Google Cloud Storage bucket name to load the files")
def evaluate(model_checkpoint, bucket_name) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depended on it")
    print(model_checkpoint)

    # Download model from GCS to local filesystem
    if bucket_name:
        download_from_gcs(bucket_name, model_checkpoint, model_checkpoint)

    model = MyAwesomeModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupt_mnist()
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
