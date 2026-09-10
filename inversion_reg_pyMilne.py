import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import glob
import sys
import os
sys.path.append("../pyMilne")
import MilneEddington as MEy
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_fits_cube(fits_path, out_path, wavelength, lines, 
                      N_WORKERS=1, initial_guess=None, psf=None):
    
    with fits.open(fits_path, memmap=True) as hdul:
        data_cube = hdul[0].data  # Shape: (4, Ny, Nx, N_wvl)
        
        if data_cube.ndim == 4:
            stokes_dim, ny, nx, n_wvl = data_cube.shape
            n_pixels = ny * nx
            reshaped_stokes = np.transpose(data_cube, (1, 2, 0, 3)).astype(np.float32)
        else:
            raise ValueError(f"Expected 4D array (Stokes, Y, X, Lambda), got {data_cube.ndim}D")

        if initial_guess is None:
            initial_guess = np.array([500., 1.0, 0.39, 0.1, 0.02, 30., 0.1, 0.2, 0.8], dtype=np.float32)
            initial_guess = np.tile(initial_guess, (ny, nx, 1))
        else:
            initial_guess = np.array(initial_guess, dtype=np.float32)
            initial_guess = np.tile(initial_guess, (ny, nx, 1))
            print(f"Initial guess shape: {initial_guess.shape}")

        regions_config = [[wavelength.astype(np.float32), psf]]

        me_inverter = ME.MilneEddington(
                regions=regions_config,
                lines=lines,
                nthreads=N_WORKERS,          # ProcessPoolExecutor handles multi-core batching
                precision='float32'
            )

        inverted_model, syn , chi2 = me_inverter.invert_spatially_regularized(initial_guess, 
                                                                         obs=reshaped_stokes, 
                                                                         sig=3E-3, 
                                                                         mu=1.0, 
                                                                         nIter=20, 
                                                                         chi2_thres=1.0)

        # Check output of the inversion
        if inverted_model is None:
            raise RuntimeError("Inversion failed, no output generated.")
        else:
            print(f"Inverted model shape: {inverted_model.shape}")
            print(f"Synthetic Stokes profiles shape: {syn.shape}")
            print(f"Chi-squared: {chi2}")

        # Ask the user if they want to save the results
        save_results = input("Do you want to save the inversion results to a FITS file? (y/n): ").strip().lower()
        if save_results != 'y':
            return

        # Save out to standard FITS format the inverted model and the synthetic Stokes profiles in a same file
        hdu = fits.PrimaryHDU(data=np.array(inverted_model))
        hdu.header['COMMENT'] = "pyMilne Milne-Eddington inversion parameters"
        hdu_syn = fits.ImageHDU(data=np.array(syn))
        hdu_syn.header['COMMENT'] = "Synthetic Stokes profiles from pyMilne inversion"
        hdul = fits.HDUList([hdu, hdu_syn])
        hdul.writeto(out_path, overwrite=True)

        print("Inversion finished successfully for all pixels.")


if __name__ == "__main__":
    INPUT_FITS = input("Enter path to input FITS file: ")
    OUTPUT_FITS = input("Enter path to output FITS file: ")
    LINES_IN = input("Enter spectral line labels (e.g., 6301, 6302, 5247, 5250): ")
    LINES = [int(line.strip()) for line in LINES_IN.split(',')]

    # Calibrated wavelength axis in Ångströms
    WVL_nm = fits.open(INPUT_FITS)[1].data
    WVL_A = (WVL_nm * 10).astype(np.float32)
    
    N_WORKERS = max(1, os.cpu_count() - 2)
    BATCH_SIZE = 500

    m_in = np.array([500., 1.0, 0.39, 0.1, 0.02, 30., 0.1, 0.2, 0.8], dtype=np.float32)

    process_fits_cube(
        INPUT_FITS, 
        OUTPUT_FITS, 
        WVL_A, 
        LINES, 
        N_WORKERS=N_WORKERS,
        initial_guess=m_in
    )
