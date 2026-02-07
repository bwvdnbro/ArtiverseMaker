#!/usr/bin/env python3

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    argparser.add_argument("output")
    args = argparser.parse_args()

    data = np.load(args.input)

    img = data["img"]
    #    img += np.random.rand(*img.shape) * 1.e-1

    print(img.min(), img.max())

    #    img = np.log(img)

    pl.imshow(img)
    pl.gca().axis("off")
    pl.savefig(args.output, dpi=300, bbox_inches="tight")
