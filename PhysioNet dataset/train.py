# train.py
import argparse
import torch
import torch.optim as optim
from dataset import get_dataloaders
from models import build_model
from utils import composite_loss, save_checkpoint
import torch.nn as nn
from datetime import datetime
import os

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x: x

def run_training(args):
    train_loader, val_loader, input_dim, scaler = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        random_seed=args.seed
    )

    model = build_model(model_type=args.model_type,
                        input_dim=input_dim,
                        hidden_dim=args.hidden_dim,
                        latent_dim=args.latent_dim,
                        timesteps=args.timesteps,
                        nhead=args.nhead,
                        num_layers=args.num_layers)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.force_cpu else 'cpu')
    model.to(device)

    if torch.cuda.device_count() > 1 and not args.force_cpu:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_points = torch.linspace(0, 1, args.ode_steps).to(device)

    best_val = float('inf')
    os.makedirs(args.ckpt_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch, time_points)
            loss = composite_loss(outputs, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch, time_points)
                loss = composite_loss(outputs, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{args.epochs}  Train: {train_loss:.6f}  Val: {val_loss:.6f}")

        # save best
        if val_loss < best_val or (epoch + 1) % args.save_every == 0:
            best_val = min(val_loss, best_val)
            ckpt_name = os.path.join(args.ckpt_dir,
                                     f"{args.model_type}_ckpt_epoch{epoch+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_dim': input_dim,
                'hidden_dim': args.hidden_dim,
                'latent_dim': args.latent_dim,
                'model_type': args.model_type
            }, ckpt_name)

def get_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_type', choices=['mlp', 'rnn', 'transformer'], required=True)
    p.add_argument('--data_dir', type=str, default='merged_data')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--val_split', type=float, default=0.2)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--latent_dim', type=int, default=64)
    p.add_argument('--ode_steps', type=int, default=2)
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--nhead', type=int, default=4)
    p.add_argument('--num_layers', type=int, default=2)
    p.add_argument('--ode_solver', type=str, default='dopri5')
    p.add_argument('--ode_timepoints', type=int, default=2)
    p.add_argument('--save_every', type=int, default=25)
    p.add_argument('--ckpt_dir', type=str, default='checkpoints')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force_cpu', action='store_true', help="Force CPU even if CUDA available")
    return p

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    run_training(args)
