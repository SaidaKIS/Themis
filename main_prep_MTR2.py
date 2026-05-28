import numpy as np
from astropy.io import fits
import astropy.units as u #Operations between units and constants
import astropy.coordinates #SkyCoord, SpectralCoord, StokesCoord
import matplotlib.pyplot as plt
import glob
import sys
from astropy.wcs import WCS
import os

from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform
from scipy.optimize import minimize_scalar
from scipy.ndimage import map_coordinates
from interative_get_roi import get_roi

#------Basic procedure: Raw Data -> Dark Subtraction -> De-curving (Straightening) NOT CONSIDERED FOR NOW-> Flat-Field Division -> Affine Beam Alignment -> Stokes Demodulation


def maping_coord(image, poly_coefficients, fixed_x_target, order=1, mode='nearest'):

    height, width = image.shape
    straightened_image = np.zeros_like(image)

    for y in range(height):
        # Find out where the curve thinks the line core is at this specific row
        current_line_x = np.polyval(poly_coefficients, y)
        
        # Calculate the sub-pixel shift required to pull it to the target column
        dx = current_line_x - fixed_x_target
        
        # Create an interpolated coordinate map for just this row
        # (Original column coordinates + the horizontal shift delta)
        shifted_cols = np.arange(width) + dx
        
        # We clamp the coordinates to ensure they don't break the boundaries
        shifted_cols = np.clip(shifted_cols, 0, width - 1)
        
        # Build coordinate grid for this specific row line
        coords = np.vstack((np.full(width, y), shifted_cols))
        
        # Re-sample the image line smoothly. 
        # FIX: Changed mode='edge' to mode='nearest' for scipy compatibility
        straightened_image[y, :] = map_coordinates(
            image, 
            coords, 
            order=order, 
            mode=mode
        )
    return straightened_image

def straighten_spectral_lines(image, line_center_col, ymin, ymax, search_window=4):
    """
    Tracks a curved spectral line along the row axis (Y), fits a 2nd-order 
    polynomial, and straightens the entire image based on that curve.
    """
    height, width = image.shape
    y_indices = np.arange(ymin, ymax)
    detected_x = []
    
    # Step 1: Trace the center (minima/core) of the spectral line for each row
    for y in range(ymin, ymax):
        # Isolate a small horizontal window around the expected line position
        start_win = max(0, line_center_col - search_window)
        end_win = min(width, line_center_col + search_window)
        row_segment = image[y, start_win:end_win]
        
        # Find the pixel column where the absorption line is deepest (minimum intensity)
        local_min_idx = np.argmin(row_segment)
        actual_x = start_win + local_min_idx
        detected_x.append(actual_x)
        
    # Step 2: Fit a 2nd-degree polynomial (x = a*y^2 + b*y + c) to the curve
    poly_coefficients = np.polyfit(y_indices, detected_x, deg=1)

    # Step 3: Create a destination coordinate map for the un-warping process
    # We want to shift every pixel horizontally so the line becomes perfectly vertical

    straightened_image = maping_coord(image, poly_coefficients, line_center_col, order=1, mode='nearest')
    
    return straightened_image, poly_coefficients

def compute_calibration(flat_filepath, dark_filepath=None):
    """
    Reads flat-field, applies dark subtraction if available, splits the dual 
    beams, computes sub-pixel alignment, and finds the intensity scale factor 's'.
    """
    # Load the flat-field FITS image
    with fits.open(flat_filepath) as hdul:
        flat_data = hdul[0].data.astype(float)
        flat_header = hdul[0].header
    if flat_data.ndim == 3:
        flat_data = np.mean(flat_data, axis=0) # Average frames if it's a cube
        
    # --- CRITICAL STEP: DARK CORRECTION (Performed first) ---

    if dark_filepath:
        print(f"Applying dark frame correction to flat-field using: {dark_filepath}")
        with fits.open(dark_filepath) as hdul:
            dark_data = hdul[0].data.astype(float)
            dark_header = hdul[0].header
        if dark_data.ndim == 3:
            dark_data = np.mean(dark_data, axis=0)
        
        # Additive dark noise must be removed before dealing with multiplicative flats
        flat_data = flat_data - dark_data

    #Get the config file of the parameters interativelly
    config_roi = get_roi(flat_data)    

    height, width = flat_data.shape
    half_y = height // 2
    
    # Split into Top Beam (b1) and Bottom Beam (b2)
    b1_flat = flat_data[0:half_y, :]
    b2_flat = flat_data[half_y:, :]

    # Compute the the polynomial De-curving NOT CONSIDERED FOR NOW
    #b1_flat_straight, poly_coeffs_b1 = straighten_spectral_lines(b1_flat, line_center_col=config_roi['line_center_b1'], ymin=config_roi['lrg1'][0], ymax=config_roi['lrg1'][1])   
    #b2_flat_straight, poly_coeffs_b2 = straighten_spectral_lines(b2_flat, line_center_col=config_roi['line_center_b2'], ymin=config_roi['lrg2'][0], ymax=config_roi['lrg2'][1]) 

    #This measures the exact shift needed to register b2 (moving) onto b1 (reference).
    # upsample_factor=100 enables ultra-precise 0.01 sub-pixel registration.
    shift, error, diffphase = phase_cross_correlation(
        b2_flat, 
        b1_flat, 
        upsample_factor=100
    )
    shift_y, shift_x = shift[0], shift[1]

    tform = AffineTransform(translation=(shift_x, shift_y))

    b2_flat_reg = warp(b2_flat, tform, order=1)
    
    # Find scale factor 's' by minimizing intensity variance between channels
    def loss_function(s):
        diff = b1_flat - s * b2_flat_reg
        return np.std(diff)

    # from scal=fminbnd(@(s) std(mean(f1R(:,c1wrg)-s*f2R(:,c1wrg),1)),0.5,1.5); in get_tform.m
    res = minimize_scalar(loss_function, bounds=(0.5, 1.5), method='bounded')
    s_factor = res.x
    
    print(f"--- Calibration Computed ---")
    print(f"Measured Alignment Shift -> X: {shift_x:.3f}px, Y: {shift_y:.3f}px")
    print(f"Calculated Gain Scale Factor (s): {s_factor:.4f}\n")

    return tform, s_factor, config_roi

def process_science_data(science_filepath, tform, s_factor, poly_coeffs_b1=None, 
                        poly_coeffs_b2=None, config_roi=None, dark_filepath=None):
    """
    Applies initial dark subtraction, maps alignment, normalizes gain, 
    and resolves the 4 Stokes components from the modulation sequence.
    """
    with fits.open(science_filepath) as hdul:
        header = hdul[0].header
        raw_data = hdul[0].data.astype(float) # Shape: (frames, height, width)
        
    # --- CRITICAL STEP: DARK CORRECTION (Performed first) ---
    if dark_filepath:
        print(f"Applying dark frame correction to science data...")
        with fits.open(dark_filepath) as hdul:
            dark_data = hdul[0].data.astype(float)
        if dark_data.ndim == 3:
            dark_data = np.mean(dark_data, axis=0)
            
        # Subtract the static sensor thermal floor from every raw temporal frame
        # NumPy automatically broadcasts the 2D dark across the 3D science cube
        raw_data = raw_data - dark_data
        
    frames, height, width = raw_data.shape
    half_y = height // 2

    seq_stk = header["SEQ_STOK"].split(" ")
    nsew_stk = len(seq_stk)
    scans = header["NAXIS3"]//nsew_stk

    raw_data_split = np.reshape(raw_data, (scans, nsew_stk, height, width))

    scan_stokes = np.zeros((scans, 4, half_y, width))

    # Slice arrays and apply the calculated geometric shifts
    for s in range(scans):
        
        b1_cleaned = np.zeros((nsew_stk, half_y, width))
        b2_cleaned = np.zeros((nsew_stk, half_y, width))

        for k in range(nsew_stk):
            b1_frame = raw_data_split[s, k, 0:half_y, :]
            b2_frame = raw_data_split[s, k, half_y:, :]

            # Apply the same polynomial straightening to the science frames NOT CONSIDERED
            if poly_coeffs_b1 is not None and poly_coeffs_b2 is not None and config_roi is not None:
                b1_frame = maping_coord(b1_frame, poly_coeffs_b1, config_roi['line_center_b1'], order=1, mode='nearest')
                b2_frame = maping_coord(b2_frame, poly_coeffs_b2, config_roi['line_center_b2'], order=1, mode='nearest')
        
            # Warp beam 2 coordinates onto beam 1 coordinates
            b2_frame_reg = warp(b2_frame, tform, order=1)
        
            b1_cleaned[k] = b1_frame
            b2_cleaned[k] = b2_frame_reg

        # Compute demodulated differences (bs) and sums (is) incorporating gain 's'
        bs = (b1_cleaned - s_factor * b2_cleaned) / 2.0
        is_map = (b1_cleaned + b2_cleaned) / 2.0
    
        # Demodulate sequences into final Stokes Vector components

        stokes = np.zeros((4, half_y, width))
        stokes[0] = is_map[0]                 # Stokes I (Total Brightness)
        stokes[1] = (bs[0] - bs[1]) / 2.0     # Stokes Q (Linear 0°/90°)
        stokes[2] = (bs[2] - bs[3]) / 2.0     # Stokes U (Linear 45°/135°)
        stokes[3] = (bs[4] - bs[5]) / 2.0     # Stokes V (Circular Right/Left)

        scan_stokes[s] = stokes

        plt.ioff()
        fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        stokes_labels = ['I', 'Q', 'U', 'V']
        for i in range(4):
            ax[i//2, i%2].imshow(stokes[i,:, config_roi['srg'][0]:config_roi['srg'][1]], cmap='gray', origin='lower')
            ax[i//2, i%2].set_title(f"Stokes {stokes_labels[i]}")
        plt.tight_layout()
        plt.show()
        sys.exit()

    scan_stokes_crop = scan_stokes

    return scan_stokes_crop, header


if __name__ == "__main__":
    # Define your paths (Optional: Set dark_file to None if no dark frame exists)
    raw_file_sci = "250206_AR13981flaring/t001_b0303_sp_20250206_100733_b3.fts"
    raw_file_flats = "250206_AR13981flaring/t013_b0303_sp_20250206_112630_y3.fts"
    raw_file_darks = "250206_AR13981flaring/t058_b0303_sp_20250206_153655_x3.fts"
    
    # Step 1: Compute alignment and scale factor (handling darks first if present)
    tform_matrix, s, config_roi = compute_calibration(raw_file_flats, dark_filepath=raw_file_darks)
    
    # Step 2: Extract final Stokes matrix (handling darks first if present)
    stokes_cube, fits_header = process_science_data(raw_file_sci, tform_matrix, s, poly_coeffs_b1=None, poly_coeffs_b2=None, config_roi=config_roi, dark_filepath=raw_file_darks)
    print(f"Final Stokes Cube Shape: {stokes_cube.shape}")
    plt.ioff()
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    stokes_labels = ['I', 'Q', 'U', 'V']
    for i in range(4):
        ax[i//2, i%2].imshow(stokes_cube[0, i, config_roi['srg'][0]:config_roi['srg'][1]], cmap='gray', origin='lower')
        ax[i//2, i%2].set_title(f"Stokes {stokes_labels[i]}")
    plt.tight_layout()
    plt.show()
    ## Save output
    #hdu = fits.PrimaryHDU(stokes_cube, header=fits_header)
    #hdu.writeto("stokes_output_s4.fits", overwrite=True)
    #print("Successfully saved clean Stokes parameters to 'stokes_output_s4.fits'")



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

# NOTES 28 MAY 26: The residuals of the dual-beam processing in the flat images reaches around -250 to 250
# counts in the image intensity so it could give an idea of some cross-talk. Intensity levels are around 1300 to 1400 and the
# stokes V signals are reaching 100 to 200 counts.

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




