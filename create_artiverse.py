#!/usr/bin/env python3

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pl

import astropy.units as u
import astropy.coordinates as c
from astropy import wcs
from astropy.io import fits
from image_to_mask import create_mask
import multiprocessing as mp


def create_fits(name, x, y, I, D, cx, cy):
    dx = (x[0, 1] - x[0, 0]) * u.kpc
    dy = (y[1, 0] - y[0, 0]) * u.kpc
    dax = (dx / D).to(u.arcsec, equivalencies=u.dimensionless_angles())
    day = (dy / D).to(u.arcsec, equivalencies=u.dimensionless_angles())
    centre = c.SkyCoord(ra=cx, dec=cy, frame="icrs")
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [x.shape[0] // 2, x.shape[1] // 2]
    w.wcs.cdelt = [dax / u.degree, day / u.degree]
    w.wcs.crval = [centre.ra.degree, centre.dec.degree]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    header = w.to_header()
    header.update(DISTANCE=(D.to(u.Mpc).value, "[Mpc] Distance to object"))

    hdu = fits.PrimaryHDU(data=I, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto(name, overwrite=True)


def sersic(r, I0, rs, n):
    return I0 * np.exp(-((r / rs) ** (1.0 / n)))


def create_artificial_galaxy(parameters):
    i, (dpix, sizefac, I0, (n, rs), ra, dec, D) = parameters

    size = sizefac * rs
    imgcoord = np.arange(0.0, size, dpix)
    imgcoord = np.append(-imgcoord[1:][::-1], imgcoord)
    imgx, imgy = np.meshgrid(imgcoord, imgcoord)
    r = np.sqrt(imgx**2 + imgy**2)
    I = sersic(r, I0, rs, n)
    I += np.random.rand(*I.shape) * 1.0e-3 * I0
    print(i, dpix, sizefac, I.shape)
    np.savez_compressed(f"img_{i:03d}.npz", I0=I0, n=n, rs=rs, img=I, x=imgx, y=imgy)
    create_fits(f"img_{i:03d}.fits", imgx, imgy, I, D, ra, dec)


if __name__ == "__main__":
    np.random.seed(42)

    mask = create_mask("Smiley.png")
    x = np.linspace(0.0, 10.0, mask.shape[0])
    y = np.linspace(0.0, 10.0, mask.shape[1])
    xs, ys = np.meshgrid(x, y)

    rate = mask.sum() / mask.size

    target = 1000
    sample_size = int(target / rate)

    params = np.random.rand(sample_size, 2) * 10.0
    xbin = np.digitize(params[:, 0], x)
    ybin = np.digitize(params[:, 1], y)
    params = params[mask[ybin, xbin]]
    while len(params) < target:
        xy = np.random.rand(int(0.1 * target / rate), 2) * 10.0
        xbin = np.digitize(xy[:, 0], x)
        ybin = np.digitize(xy[:, 1], y)
        params = np.append(params, xy[mask[ybin, xbin]], axis=0)
    params = params[:target]

    params[:, 0] = 1.0 + 0.5 * params[:, 0]
    params[:, 1] = 1.0 + 0.01 * params[:, 1]

    pl.semilogy(params[:, 0], params[:, 1], ".")
    pl.savefig("params.png", dpi=300)
    pl.close()

    maxx = 1000.0
    coords = np.random.rand(target, 2) * maxx
    I0s = 1.0 + 10.0 * np.random.rand(target)

    long = (2.0 * np.pi * np.random.random(target) * u.radian).to(u.deg)
    lat = (
        (np.arccos(2.0 * np.random.random(target) - 1.0) - 0.5 * np.pi) * u.radian
    ).to(u.deg)
    Ds = (300.0 + 200.0 * np.random.random(target)) * u.Mpc

    dpixs = 0.01 + 0.02 * np.random.rand(target)
    sizefacs = 2.0 + 3.0 * np.random.rand(target)
    with mp.Pool(16) as pool:
        pool.map(
            create_artificial_galaxy,
            enumerate(zip(dpixs, sizefacs, I0s, params, long, lat, Ds)),
        )
