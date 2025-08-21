# run_transformer.py
import argparse
from train import run_training

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='merged_data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--ckpt_dir', default='checkpoints/transformer')
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    args, unknown = parser.parse_known_args()

    import argparse as _arg
    train_args = _arg.Namespace(
        model_type='transformer',
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.2,
        epochs=args.epochs,
        lr=1e-3,
        weight_decay=1e-5,
        hidden_dim=128,
        latent_dim=64,
        ode_steps=2,
        timesteps=1000,
        nhead=args.nhead,
        num_layers=args.num_layers,
        save_every=25,
        ckpt_dir=args.ckpt_dir,
        num_workers=0,
        seed=42,
        force_cpu=False
    )
    run_training(train_args)

if __name__ == "__main__":
    main()
