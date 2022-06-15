# create a two moons dataset.
# this script takes two arguments, the number of samples and the noise
from sklearn.datasets import make_moons
import sys
import numpy as np
if __name__ == "__main__":
    nsamples = 100
    noise = 0.1
    test_perc = 0.2
    # read args.
    if len(sys.argv) > 1:
        nsamples = int(sys.argv[1])

    if len(sys.argv) > 2:
        noise = float(sys.argv[2])
    print(f"Sampling moons with {nsamples} samples and noise={noise}...")

    for mode, use_samples in zip(["train", "test"],[nsamples, int(test_perc*nsamples)]):

        X, y = make_moons(use_samples, noise=noise)
        # export to csv.
        data_matrix = np.concatenate([X, y.reshape(-1,1)], axis=1)
        np.savetxt(f"twomoons-{mode}.csv", data_matrix, "%f,%f,%d",header="x1,x2,label", delimiter=",")

