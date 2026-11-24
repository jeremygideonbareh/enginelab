"""Train the part classifier: python -m enginelab.train [n_engines]"""
import sys
import time
import warnings

from .classify import train

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 240
    t = time.time()
    train(n_engines=n)
    print(f"trained on {n} engines in {time.time() - t:.0f}s")
