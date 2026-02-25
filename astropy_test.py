from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
fn = get_pkg_data_filename('data/j94f05bgq_flt.fits', package='astropy.wcs.tests')
f = fits.open(fn)
w = WCS(f[1].header)
print(f[1].header)
sky = w.pixel_to_world(30, 40)
print(sky)
