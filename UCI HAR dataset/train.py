import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from models import AccidentPredictionPipeline
from dataset import get_dataloaders
from utils import composite_loss, set_seed, save_checkpoint, get_device, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Train iDiffODE variants on UCI HAR")
    p.add_argument('--data_dir', type=str, required=True, help="Path to UCI HAR extracted folder")
    p.add_argument('--model_type', type=str, choices=['mlp', 'rnn', 'transformer'], default='mlp')
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--latent_dim', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--timesteps', type=int, default=1000)
    p.add_argument('--device', type=str, default=None, help="Device (cpu or cuda). If omitted, auto-selects.")
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--resume', type=str, default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()
    train_loader, val_loader, input_dim = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers)

    model = AccidentPredictionPipeline(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        model_type=args.model_type,
        diffusion_timesteps=args.timesteps
    )
    model.to(device)

    # If multiple GPUs, wrap with DataParallel
    if torch.cuda.device_count() > 1 and device.type == 'cuda':
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(model, optimizer, args.resume, map_location=device)
        start_epoch = ckpt.get('epochs_trained', 0) + 1
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    time_points = torch.linspace(0, 1, 2).to(device)

    best_val = float('inf')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_train_loss = 0.0
        it = 0
        for batch_x in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x, time_points)
            loss = composite_loss(outputs, batch_x)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            it += 1
        avg_train = total_train_loss / max(1, it)

        # Validation
        model.eval()
        total_val_loss = 0.0
        vit = 0
        with torch.no_grad():
            for batch_x in val_loader:
                batch_x = batch_x.to(device)
                outputs = model(batch_x, time_points)
                total_val_loss += composite_loss(outputs, batch_x).item()
                vit += 1
        avg_val = total_val_loss / max(1, vit)

        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch+1}/{args.epochs} | Train: {avg_train:.6f} | Val: {avg_val:.6f}")

        ckpt_file = save_dir / f"{args.model_type}_epoch{epoch+1}.pth"
        save_checkpoint({
            'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'latent_dim': args.latent_dim,
            'epochs_trained': epoch,
            'args': vars(args)
        }, str(ckpt_file))

        if avg_val < best_val:
            best_val = avg_val
            best_file = save_dir / f"{args.model_type}_best.pth"
            save_checkpoint({
                'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_dim': input_dim,
                'hidden_dim': args.hidden_dim,
                'latent_dim': args.latent_dim,
                'epochs_trained': epoch,
                'args': vars(args)
            }, str(best_file))
            print(f"  -> New best val {best_val:.6f} saved to {best_file}")

    print("Training finished.")


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
