import argparse
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--nlayers", type=int, default=2)
    args = parser.parse_args()
    transformer_cfg = {'nhead': args.nhead, 'num_layers': args.nlayers}
    train(args.csv, model_type='transformer', epochs=args.epochs, batch_size=args.bs, transformer_cfg=transformer_cfg)
