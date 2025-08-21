
from argparse import Namespace
import train

if __name__ == '__main__':
    args = Namespace(csv="data/raw_combined_accidents.csv",
                     model="rnn",
                     epochs=200,
                     bs=32,
                     lr=1e-3,
                     wd=1e-5,
                     hidden=128,
                     latent=64,
                     rnn_type="gru",
                     save="checkpoints/idiffode_rnn.pth",
                     device="cuda" if __import__('torch').cuda.is_available() else "cpu")
    train.main(args)
