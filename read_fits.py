#!/usr/bin/env python3

import numpy as np
from astropy import wcs
from astropy.io import fits
import astropy.units as u

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl


def plot_fits(input_name, output_name):
    hdul = fits.open(input_name)
    w = wcs.WCS(hdul[0].header)
    distance = hdul[0].header["DISTANCE"] * u.Mpc

    p = w.pixel_to_world([[0, 0], [1, 1]], [0, 0])
    dangle = p[1][0].ra - p[0][0].ra
    dx = ((dangle / u.radian) * distance).to(u.kpc)

    imgcoord = np.linspace(0.0, dx * w.array_shape[0], w.array_shape[0])
    imgcoord -= 0.5 * dx * w.array_shape[0]
    imgx, imgy = np.meshgrid(imgcoord, imgcoord)

    pl.contourf(imgx, imgy, hdul[0].data, levels=100)
    pl.xlabel("x [kpc]")
    pl.ylabel("y [kpc]")
    pl.tight_layout()
    pl.savefig(output_name, dpi=300, bbox_inches="tight")
    pl.close()


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("input")
    argparser.add_argument("output")
    args = argparser.parse_args()

    plot_fits(args.input, args.output)
