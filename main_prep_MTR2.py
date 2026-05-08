import numpy as np
from astropy.io import fits
import astropy.units as u #Operations between units and constants
import astropy.coordinates #SkyCoord, SpectralCoord, StokesCoord
import matplotlib.pyplot as plt
import glob
import sys
from astropy.wcs import WCS
import os

def flat_dark_corr(img, array_flat, array_dark):
    master_dark = np.mean(array_dark, axis=0)
    master_flat = np.mean(array_flat, axis=0)
    master_flat_avg = np.mean(master_flat)
    pure_master_flat = master_flat - master_dark
    img_dark_corr = img - master_dark
    img_dark_flat_corr = img_dark_corr / pure_master_flat
    return img_dark_flat_corr * master_flat_avg

#-------------------------INFORMATION---------------------------------
#Multi-ray spectrograph MTR2
# MTR2 is made of a predispersor (“SP1” first one on the left part), 
# forming a low resolution spectrum on the wavelength selector 
# (so called the “mask plate”). From there, only the selected bandwiths (masks)
# travel to the high-resolution echelle spectrograph (right part), 
# ending over the MTR cameras at the SP2 output (one bandwith per camera).

# Info of the dataset
# t001_b0303_sp_20250206_100733_b3.fts
# sequence+wl(303:6299 to 6305 AA)+sp+time+b3(sci data)
# 606(Halpha), b3(sci data) y3(flats) x3(darks)

# Initial construction of the ND object
# Active region test
# Spectra - 6301 spectral range


#-------------------------LOAD DATA---------------------------------
raw_file_sci = "250206_AR13981flaring/t001_b0303_sp_20250206_100733_b3.fts"
raw_file_flats = "250206_AR13981flaring/t013_b0303_sp_20250206_112630_y3.fts"
raw_file_darks = "250206_AR13981flaring/t058_b0303_sp_20250206_153655_x3.fts"

raw_hdu_sci = fits.open(raw_file_sci)
raw_hdu_flats = fits.open(raw_file_flats)
raw_hdu_darks = fits.open(raw_file_darks)

header = raw_hdu_sci[0].header

#----------------------CAMERA CORRECTIONS----------------------
raw_cube_sci = raw_hdu_sci[0].data
raw_cube_flats = raw_hdu_flats[0].data
raw_cube_darks = raw_hdu_darks[0].data

raw_cube_sci_l1 = np.zeros_like(raw_cube_sci)
raw_cube_sci_l1.astype(np.float32)
for i, img in enumerate(raw_cube_sci[:12]):
    print(f"\rCorrection Progress: {i}", end="", flush=True)
    corr_img = flat_dark_corr(img, raw_cube_flats, raw_cube_darks)
    raw_cube_sci_l1[i] = corr_img

#----------------------STOKES PARAMETERS----------------------
seq_stk = header["SEQ_STOK"].split(" ")
nsew_stk = len(seq_stk)
scans = header["NAXIS3"]//nsew_stk
_, H, W =  raw_cube_sci_l1.shape
raw_cube_sci_l1_split = np.reshape(raw_cube_sci_l1, (scans, nsew_stk, H, W))
p_stkQ, p_stkU, p_stkV =[], [], []
for s in range(scans): 
    for i, stk in enumerate(seq_stk):
        if 'Q' in stk:
            p_stkQ.append(raw_cube_sci_l1_split[s, i, : ,:])
        elif 'U' in stk:
            p_stkU.append(raw_cube_sci_l1_split[s, i, : ,:])
        elif 'V' in stk:
            p_stkV.append(raw_cube_sci_l1_split[s, i, : ,:])
        else:
            raise TypeError("No Stokes parameters recognized")
         
    I = [(p_stkQ[0][:header["NAXIS2"]//2, :] + p_stkQ[0][header["NAXIS2"]//2:, :])/2,
         (p_stkQ[1][:header["NAXIS2"]//2, :] + p_stkQ[1][header["NAXIS2"]//2:, :])/2,
         (p_stkU[0][:header["NAXIS2"]//2, :] + p_stkU[0][header["NAXIS2"]//2:, :])/2,
         (p_stkU[1][:header["NAXIS2"]//2, :] + p_stkU[1][header["NAXIS2"]//2:, :])/2,
         (p_stkV[0][:header["NAXIS2"]//2, :] + p_stkV[0][header["NAXIS2"]//2:, :])/2,
         (p_stkV[1][:header["NAXIS2"]//2, :] + p_stkV[1][header["NAXIS2"]//2:, :])/2]

    fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
    ax_p = ax.flatten()
    for i, a in enumerate(ax_p):
        a.plot(I[i][200,:])
    plt.show()

    Q = [np.abs((p_stkQ[0][:header["NAXIS2"]//2, :] - p_stkQ[0][header["NAXIS2"]//2:, :])/2),
         np.abs((p_stkQ[1][:header["NAXIS2"]//2, :] - p_stkQ[1][header["NAXIS2"]//2:, :])/2)]
    
    U = [np.abs((p_stkU[0][:header["NAXIS2"]//2, :] - p_stkU[0][header["NAXIS2"]//2:, :])/2),
         np.abs((p_stkU[1][:header["NAXIS2"]//2, :] - p_stkU[1][header["NAXIS2"]//2:, :])/2)]
    
    V = [np.abs((p_stkV[0][:header["NAXIS2"]//2, :] - p_stkV[0][header["NAXIS2"]//2:, :])/2),
         np.abs((p_stkV[1][:header["NAXIS2"]//2, :] - p_stkV[1][header["NAXIS2"]//2:, :])/2)]

    sys.exit()


#ToDos - scatter plots Intensity cutting edges - ask Bernard for the matlab code


#----------------------HEADER-----------------------------

#spec_scale = header.get('SPECSCAL', 1.0) / 1000.0  # Converting mAngstrom to Angstrom if needed
#wave_ref = header.get('WAVELNTH', 6562.8)         # Central wavelength
#
## Spatial Axis (Axis 2)
#spat_scale = header.get('SPATSCAL', 0.234) / 3600.0 # Converting arcsec to degrees
#
## Time/Scan Axis (Axis 3)
## In SCAN mode, each step is often a shift in solar X
#step_x = header.get('STEP_X', 0.5) / 3600.0        # Step size in degrees
#
#wcs_dict = {
#    'BITPIX': header['BITPIX'],
#    'NAXIS': header['NAXIS'],
#    'NAXIS1': header['NAXIS1'],
#    'NAXIS2': header['NAXIS2'],
#    'NAXIS3': header['NAXIS3'],
#
#    'DATE-OBS': header["DATE-OBS"],
#    'DATE-BEG': header["DATE-BEG"],
#    'DATE-END': header["DATE-END"],
#    'EXPTIME': header["EXPTIME"],
#
#    'CTYPE1': 'WAVELNTH', 
#    'CUNIT1': 'Angstrom', 
#    'CDELT1': header['SPECSCAL'] * 1000, #verificar con la calibración 
#    'CRPIX1': header['NAXIS1'], 
#    'CRVAL1': np.nan, #verificar con la calibración
#
#    'CTYPE2': 'HPLT-TAN', # Helioprojective Latitude (Spatial along slit)
#    'CUNIT2': 'deg',
#    'CDELT2': spat_scale,
#    'CRPIX2': header['NAXIS2'],
#    'CRVAL2': header.get('DIST_NS', 0.0) / 3600.0,
#
#    'CTYPE3': 'HPLN-TAN', # Helioprojective Longitude (The scan direction)
#    'CUNIT3': 'deg',
#    'CDELT3': step_x,
#    'CRPIX3': 1,
#    'CRVAL3': header.get('DIST_EW', 0.0) / 3600.0,
#}




