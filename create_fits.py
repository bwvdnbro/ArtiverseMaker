#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as pl
import astropy.units as u
import astropy.coordinates as c
from astropy import wcs
from astropy.io import fits


def create_fits(x, y, I, D, cx, cy):
    dx = (x[0, 1] - x[0, 0]) * u.kpc
    dy = (y[1, 0] - y[0, 0]) * u.kpc
    D *= u.kpc
    dax = (dx / D).to(u.arcsec, equivalencies=u.dimensionless_angles())
    day = (dy / D).to(u.arcsec, equivalencies=u.dimensionless_angles())
    centre = c.SkyCoord(ra=cx * u.degree, dec=cy * u.degree, frame="icrs")
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [x.shape[0] // 2, x.shape[1] // 2]
    w.wcs.cdelt = [dax / u.degree, day / u.degree]
    w.wcs.crval = [centre.ra.degree, centre.dec.degree]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    header = w.to_header()
    header.update(DISTANCE=(D.to(u.Mpc).value, "[Mpc] Distance to object"))
    print(header)

    hdu = fits.PrimaryHDU(data=I, header=header)
    hdul = fits.HDUList([hdu])
    hdul.writeto("test.fits", overwrite=True)


def sersic(r, I0, rs, n):
    return I0 * np.exp(-((r / rs) ** (1.0 / n)))


if __name__ == "__main__":
    cx = 253.0
    cy = 60.0
    D = 342500.0
    I0 = 4.32
    rs = 1.1
    n = 3.4
    dpix = 0.01

    size = 3.0 * rs
    imgcoord = np.arange(0.0, size, dpix)
    imgcoord = np.append(-imgcoord[1:][::-1], imgcoord)

    imgx, imgy = np.meshgrid(imgcoord, imgcoord)
    r = np.sqrt(imgx**2 + imgy**2)
    I = sersic(r, I0, rs, n)
    I += np.random.rand(*I.shape) * 1.0e-3 * I0

    create_fits(imgx, imgy, I, D, cx, cy)
