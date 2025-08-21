
import argparse
import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

from dataset import AccidentDataset, AccidentPreprocessor
from models import AccidentPredictionPipeline
from utils import composite_loss, save_checkpoint

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True, help='path to CSV (traffic accidents) file')
    p.add_argument('--model', choices=['mlp','rnn','transformer'], default='mlp')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--latent', type=int, default=64)
    p.add_argument('--nhead', type=int, default=4, help='transformer nhead (only used if transformer)')
    p.add_argument('--nlayers', type=int, default=2, help='transformer nlayers (only used if transformer)')
    p.add_argument('--rnn_type', choices=['gru','rnn'], default='gru')
    p.add_argument('--save', default='checkpoints/last.pth')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def main(argv=None):
    args = parse_args() if argv is None else argv

    # Dataset + splits
    preprocessor = AccidentPreprocessor()
    dataset = AccidentDataset(args.csv, preprocessor=preprocessor)
    N = len(dataset)
    train_size = int(0.8 * N)
    val_size = N - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.bs, shuffle=False)

    # Build model
    encoder_kwargs = {}
    if args.model == 'transformer':
        encoder_kwargs = dict(nhead=args.nhead, num_layers=args.nlayers)
    elif args.model == 'rnn':
        encoder_kwargs = dict(rnn_type=args.rnn_type)

    input_dim = dataset.X.shape[1]
    model = AccidentPredictionPipeline(input_dim, hidden_dim=args.hidden, latent_dim=args.latent,
                                       encoder_type=args.model, **encoder_kwargs)
    device = torch.device(args.device)
    model.to(device)

    # multi-gpu (optional)
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    time_points = torch.linspace(0, 1, 2).to(device)

    best_val = float('inf')
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x, time_points)
            loss = composite_loss(outputs, batch_x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x, time_points)
                val_loss += composite_loss(outputs, batch_x).item()
        val_loss /= max(1, len(val_loader))

        print(f"Epoch {epoch+1}/{args.epochs}  Train: {train_loss:.6f}  Val: {val_loss:.6f}")

        # checkpoint
        ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'input_dim': input_dim,
            'hidden_dim': args.hidden,
            'latent_dim': args.latent,
            'encoder_type': args.model,
            'epochs_trained': epoch+1,
        }
        save_checkpoint(ckpt, args.save)

    return model

if __name__ == '__main__':
    main()
