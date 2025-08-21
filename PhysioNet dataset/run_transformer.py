from argparse import Namespace
import train

if __name__ == '__main__':
    args = Namespace(csv="data/raw_combined_accidents.csv",
                     model="transformer",
                     epochs=200,
                     bs=16,
                     lr=1e-3,
                     wd=1e-5,
                     hidden=256,
                     latent=64,
                     nhead=4,
                     nlayers=2,
                     save="checkpoints/idiffode_trans.pth",
                     device="cuda" if __import__('torch').cuda.is_available() else "cpu")
    train.main(args)
