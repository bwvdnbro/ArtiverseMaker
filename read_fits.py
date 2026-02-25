#!/usr/bin/env python3

from astropy import wcs
from astropy.io import fits
import astropy.units as u

if __name__ == "__main__":
  hdul = fits.open("test.fits")
  w = wcs.WCS(hdul[0].header)
  distance = hdul[0].header["DISTANCE"] * u.Mpc

  xmin = w.wcs_pix2world([0,0],0)
  xmax = w.wcs_pix2world([x.shape[0]-1,0],0)
  print(xmin, xmax)
