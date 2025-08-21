import argparse
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--bs", type=int, default=32)
    args = parser.parse_args()
    train(args.csv, model_type='rnn', epochs=args.epochs, batch_size=args.bs)
