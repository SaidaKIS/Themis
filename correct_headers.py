import numpy as np
from astropy.io import fits
import astropy.units as u #Operations between units and constants
import astropy.coordinates #SkyCoord, SpectralCoord, StokesCoord
import matplotlib.pyplot as plt
import glob
import sys
from astropy.wcs import WCS
import os

#Keyword to modify
raw_file_sci_cam6 = "260522_observation_test/t014_b0606_sp_20260522_080309_b3.fts"
data_hdu_cam6 = fits.open(raw_file_sci_cam6)
data_hdr_cam6 = data_hdu_cam6[0].header
stk_seq = data_hdr_cam6["SEQ_STOK"]

# modification of a header camera 505 
raw_file_sci_cam5 = "20290901_Bommier/t001_b0606_sp_20260901_073054_b3.fts"

#Checking and rewriting the headers

with fits.open(raw_file_sci_cam5, mode="update") as hdul:
    # Access the primary header (indexing at 0)
    header = hdul[0].header

    # 1. Update an existing keyword (or add it if it doesn't exist)
    header["SEQ_STOK"] = stk_seq

    # 5. Add HISTORY or COMMENT lines
    header.add_history("Modified header using Astropy on 2026-08-26.")
    header.add_comment("Correcting stokes sequence")

data_hdu_cam5 = fits.open(raw_file_sci_cam5)
data_hdr_cam5 = data_hdu_cam5[0].header
print(data_hdr_cam5["SEQ_STOK"])