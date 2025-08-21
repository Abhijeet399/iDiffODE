import argparse
import torch
from torch.utils.data import random_split, DataLoader
from models import AccidentPredictionPipeline
from dataset import AccidentPreprocessor, AccidentDataset
from utils import composite_loss, save_checkpoint, ensure_dir
import torch.optim as optim
import os

def train(csv_path, model_type='mlp', epochs=50, batch_size=32, lr=1e-3, weight_decay=1e-5,
          hidden_dim=128, latent_dim=64, save_dir='checkpoints', gpu=True, transformer_cfg=None):
    device = torch.device('cuda' if (gpu and torch.cuda.is_available()) else 'cpu')

    preproc = AccidentPreprocessor()
    dataset = AccidentDataset(csv_path, preproc, fit=True)

    # split
    N = len(dataset)
    train_n = int(0.8 * N)
    val_n = N - train_n
    train_ds, val_ds = random_split(dataset, [train_n, val_n])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = dataset.X.shape[1]
    model = AccidentPredictionPipeline(input_dim, hidden_dim, latent_dim, model_type, transformer_cfg).to(device)

    if torch.cuda.device_count() > 1:
        print("Using DataParallel on", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    time_points = torch.linspace(0, 1, 2).to(device)

    ensure_dir(save_dir)
    ckpt_path = os.path.join(save_dir, f"{model_type}_checkpoint.pth")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch, time_points)
            loss = composite_loss(outputs, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch, time_points)
                val_loss += composite_loss(outputs, batch).item()
        val_loss /= len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {train_loss:.6f} Val Loss: {val_loss:.6f}")

        # quick checkpoint every 10 epochs
        if (epoch+1) % 10 == 0 or (epoch+1) == epochs:
            save_checkpoint(ckpt_path, model, optimizer,
                            meta={'input_dim': input_dim, 'hidden_dim': hidden_dim,
                                  'latent_dim': latent_dim, 'epochs_trained': epoch+1})
    print("Training finished. Final checkpoint:", ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train iDiffODE variants on Traffic Accident dataset")
    parser.add_argument("--csv", type=str, required=True, help="Path to raw_combined_accidents.csv")
    parser.add_argument("--model", type=str, default="mlp", choices=['mlp','rnn','transformer'])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--no_gpu", dest='no_gpu', action='store_true')
    parser.add_argument("--nhead", type=int, default=4, help="transformer nhead")
    parser.add_argument("--nlayers", type=int, default=2, help="transformer num layers")
    args = parser.parse_args()

    transformer_cfg = {'nhead': args.nhead, 'num_layers': args.nlayers} if args.model == 'transformer' else None
    train(args.csv, model_type=args.model, epochs=args.epochs, batch_size=args.bs, lr=args.lr,
          weight_decay=args.wd, hidden_dim=args.hidden, latent_dim=args.latent,
          save_dir=args.save_dir, gpu=not args.no_gpu, transformer_cfg=transformer_cfg)
