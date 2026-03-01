#!/usr/bin/env python3

import numpy as np
import imageio.v3 as iio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl


def create_mask(imgfile):
    data = iio.imread(imgfile)
    nx = data.shape[0]
    ny = data.shape[1]
    ngrid = max(nx, ny)
    dx = (ngrid - nx) // 2
    dy = (ngrid - ny) // 2
    mask = np.zeros((ngrid, ngrid), dtype=bool)
    if len(data.shape) > 2:
        bw = np.sum(data, axis=2)
    else:
        bw = data
    imgmask = bw < 500
    mask[dx : nx + dx, dy : ny + dy] = imgmask
    return mask[::-1, :]


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    argparser.add_argument("output")
    args = argparser.parse_args()

    mask = create_mask(args.input)
    pl.imshow(mask)
    pl.gca().axis("off")
    pl.tight_layout()
    pl.savefig(args.output, dpi=300, bbox_inches="tight")
