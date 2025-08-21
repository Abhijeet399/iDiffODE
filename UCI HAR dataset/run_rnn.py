import sys
from train import parse_args, train

if __name__ == "__main__":
    sys.argv += ["--model_type", "rnn"]
    args = parse_args()
    train(args)
