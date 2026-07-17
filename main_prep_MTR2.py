import numpy as np
from astropy.io import fits
import astropy.units as u #Operations between units and constants
import astropy.coordinates #SkyCoord, SpectralCoord, StokesCoord
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import glob
import sys
from astropy.wcs import WCS
import os

from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform
from scipy.optimize import minimize_scalar
from scipy.ndimage import map_coordinates
from interative_get_roi import get_roi
from scipy.optimize import curve_fit
from tqdm import tqdm

#------Basic procedure: Raw Data -> Dark Subtraction -> 
#      De-curving (Straightening) -> Flat-Field Division -> Affine Beam Alignment 
#      -> Stokes Demodulation

# Conclusion based on the analysis of the flat-field images: 
# The dual-beam MTR2 system has a slight misalignment between the two beams, 
# which can be corrected using an affine transformation. Additionally, 
# the spectral lines exhibit curvature that can be straightened using polynomial fitting. 
# measuring the diference between the flat images in both beams the distribution of the residuals is centered around zero (62) 
# with a standard deviation of ~700 counts,
# The gain factor 's' between the two beams is also determined to ensure accurate Stokes parameter extraction.

def remove_spectral_lines(straightened_flat):
    """
    Removes the vertical spectral line profiles to isolate pure pixel-to-pixel spatial gain variations.
    """
    # Collapse along the Y-axis (spatial) to find the average spectral line profile
    spectral_profile = np.mean(straightened_flat, axis=0, keepdims=True)
    
    # Avoid division by zero
    spectral_profile[spectral_profile == 0] = 1.0
    
    # Divide out the spectral profile to leave ONLY the spatial sensor artifacts
    pure_spatial_flat = straightened_flat / spectral_profile
    
    return pure_spatial_flat

def maping_coord(image, poly_coefficients, fixed_x_target, order=1, mode='nearest'):

    height, width = image.shape
    straightened_image = np.zeros_like(image)
    line_x = []

    for y in range(height):
        # Find out where the curve thinks the line core is at this specific row
        current_line_x = np.polyval(poly_coefficients, y)
        line_x.append(current_line_x)
        
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


    return straightened_image, np.array(line_x)

def straighten_spectral_lines(image, line_center_col, ymin, ymax, search_window=20):
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
    poly_coefficients = np.polyfit(y_indices, detected_x, deg=2)

    # Step 3: Create a destination coordinate map for the un-warping process
    # We want to shift every pixel horizontally so the line becomes perfectly vertical

    straightened_image, fitted_line = maping_coord(image, poly_coefficients, line_center_col, order=1, mode='nearest')
    
    return straightened_image, poly_coefficients, fitted_line

def process_flats_eval(flat_filepath, tform, s_factor, poly_coeffs_b1=None, 
                        poly_coeffs_b2=None, config_roi=None, dark_filepath=None):
    """
    Evaluates the flat-field calibration by applying dark subtraction, 
    mapping alignment, normalizing gain, and computing the residuals.
    """
    with fits.open(flat_filepath) as hdul:
        data_raw = hdul[0].data
        flat_header = hdul[0].header
        # Convert raw data to signed integers and then to float for processing - Correct way for reading MATLAB fits
        data_sign = np.asanyarray(data_raw).view(np.int16).astype(np.float64)
        flat_data = data_sign + flat_header.get('BZERO', 0)  # Apply BZERO offset if present
        
    if flat_data.ndim == 3:
        flat_data = np.mean(flat_data, axis=0) # Average frames if it's a cube
        
    # --- CRITICAL STEP: DARK CORRECTION (Performed first) ---

    if dark_filepath:
        print(f"Applying dark frame correction to flat-field using: {dark_filepath}")
        with fits.open(dark_filepath) as hdul:
            data_raw = hdul[0].data
            dark_header = hdul[0].header
            # Convert raw data to signed integers and then to float for processing - Correct way for reading MATLAB fits
            data_sign = np.asanyarray(data_raw).view(np.int16).astype(np.float64)
            dark_data = data_sign + dark_header.get('BZERO', 0)  # Apply BZERO offset if present
        if dark_data.ndim == 3:
            dark_data = np.mean(dark_data, axis=0)
            
        flat_data = flat_data - dark_data
        
    height, width = flat_data.shape
    half_y = height // 2

    b1_flat = flat_data[0:half_y, :]
    b2_flat = flat_data[half_y:, :]

    # Apply polynomial straightening if coefficients are provided
    if poly_coeffs_b1 is not None and poly_coeffs_b2 is not None and config_roi is not None:
        b1_flat,_ = maping_coord(b1_flat, poly_coeffs_b1, config_roi['line_center_b1'], order=1, mode='nearest')
        b2_flat,_ = maping_coord(b2_flat, poly_coeffs_b2, config_roi['line_center_b2'], order=1, mode='nearest')

    # Warp beam 2 coordinates onto beam 1 coordinates
    b2_flat_reg = warp(b2_flat, tform, order=1)

    # Compute the residual difference after alignment and scaling
    residual_diff = b1_flat - s_factor * b2_flat_reg

    plt.ioff()
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    ax[0].imshow(b1_flat, cmap='gray', origin='lower')
    ax[0].set_title("Beam 1 Flat")
    
    ax[1].imshow(b2_flat_reg, cmap='gray', origin='lower')
    ax[1].set_title("Beam 2 Flat (Registered)")
    
    im = ax[2].imshow(residual_diff, cmap='bwr', origin='lower', vmin=-10000, vmax=10000)
    ax[2].set_title("Residual Difference (b1 - s*b2)")
    plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()    

    n, bins, patches = plt.hist(residual_diff.flatten(), bins=200, color='gray', alpha=0.7)

    bin_centers = (bins[:-1] + bins[1:]) / 2

    def funcion_gaussiana(x, amp, mean, sigma):
        return amp * np.exp(-(x - mean)**2 / (2 * sigma**2))

    p0 = [max(n), np.mean(residual_diff.flatten()), np.std(residual_diff.flatten())]

    popt, pcov = curve_fit(funcion_gaussiana, bin_centers, n, p0=p0)

    x_curva = np.linspace(min(bins), max(bins), 100)
    y_curva = funcion_gaussiana(x_curva, *popt)

    # Print gaussian fit parameters
    print(f"Gaussian Fit Parameters:")
    print(f"Amplitude: {popt[0]:.2f}")
    print(f"Mean: {popt[1]:.2f}")
    print(f"Sigma: {popt[2]:.2f}") 

    plt.plot(x_curva, y_curva, 'r-', linewidth=2, label=f'Ajuste Gaussiano\n$mu={popt[1]:.2f}, sigma={popt[2]:.2f}$')

    plt.title('Histogram of Residual Differences and Gaussian Fit')
    plt.title("")
    plt.xlabel("Residual Intensity")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()  

def compute_calibration(flat_filepath, dark_filepath=None, plot_check=False):
    """
    Reads flat-field, applies dark subtraction if available, splits the dual 
    beams, computes sub-pixel alignment, and finds the intensity scale factor 's'.
    """
    # Load the flat-field FITS image
    with fits.open(flat_filepath) as hdul:
        data_raw = hdul[0].data
        flat_header = hdul[0].header
        # Convert raw data to signed integers and then to float for processing - Correct way for reading MATLAB fits
        data_sign = np.asanyarray(data_raw).view(np.int16).astype(np.float64)
        flat_data = data_sign + flat_header.get('BZERO', 0)  # Apply BZERO offset if present
    
    if flat_data.ndim == 3:
        flat_data = np.mean(flat_data, axis=0) # Average frames if it's a cube
        
    # --- CRITICAL STEP: DARK CORRECTION (Performed first) ---

    if dark_filepath:
        print(f"Applying dark frame correction to flat-field using: {dark_filepath}")
        with fits.open(dark_filepath) as hdul:
            data_raw = hdul[0].data
            dark_header = hdul[0].header
            # Convert raw data to signed integers and then to float for processing - Correct way for reading MATLAB fits
            data_sign = np.asanyarray(data_raw).view(np.int16).astype(np.float64)
            dark_data = data_sign + dark_header.get('BZERO', 0)  # Apply BZERO offset if present
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

    # Compute the the polynomial De-curving debugging
    b1_flat_straight, poly_coeffs_b1, fitted_line_b1 = straighten_spectral_lines(b1_flat, line_center_col=config_roi['line_center_b1'], ymin=config_roi['lrg1'][0], ymax=config_roi['lrg1'][1])   
    b2_flat_straight, poly_coeffs_b2, fitted_line_b2 = straighten_spectral_lines(b2_flat, line_center_col=config_roi['line_center_b2'], ymin=config_roi['lrg2'][0], ymax=config_roi['lrg2'][1])   

    b1_spatial_flat = remove_spectral_lines(b1_flat_straight)
    b2_spatial_flat = remove_spectral_lines(b2_flat_straight)

    if plot_check == True:

        # Checking the differnce between b1_flat and b1_flat_straight and plot it 
        diff_b1 = b1_flat - b1_flat_straight
        diff_b2 = b2_flat - b2_flat_straight
    
        plt.ioff()
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 5), sharex=True, sharey=True)
        ax[0][0].tick_params(bottom=True, top=True, left=True, right=False)
        ax[0][1].tick_params(bottom=True, top=True, left=True, right=False)
        ax[0][2].tick_params(bottom=True, top=True, left=True, right=False)
        ax[0][0].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
        ax[0][1].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
        ax[0][2].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
        ax[1][0].tick_params(bottom=True, top=True, left=True, right=False)
        ax[1][1].tick_params(bottom=True, top=True, left=True, right=False)
        ax[1][2].tick_params(bottom=True, top=True, left=True, right=False)
        ax[1][0].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
        ax[1][1].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
        ax[1][2].tick_params(labelbottom=True, labeltop=True, labelleft=True, labelright=False)
        ax[0][0].imshow(b1_flat, cmap='gray', origin='lower')
        ax[0][0].plot(fitted_line_b1, np.arange(half_y), color='red', linewidth=1.5, label='Fitted Curve')
        ax[0][0].set_title("Beam 1 Flat")
        ax[0][1].imshow(b1_flat_straight, cmap='gray', origin='lower')
        ax[0][1].set_title("Beam 1 Straightened")
        im = ax[0][2].imshow(diff_b1, cmap='bwr', origin='lower')
        ax[0][2].set_title("Residual Difference (b1 - b1_straight)")
        plt.colorbar(im, ax=ax[0][2], fraction=0.046, pad=0.04)

        ax[1][0].imshow(b2_flat, cmap='gray', origin='lower')
        ax[1][0].plot(fitted_line_b2, np.arange(half_y), color='red', linewidth=1.5, label='Fitted Curve')
        ax[1][0].set_title("Beam 2 Flat")
        ax[1][1].imshow(b2_flat_straight, cmap='gray', origin='lower')
        ax[1][1].set_title("Beam 2 Straightened")
        im = ax[1][2].imshow(diff_b2, cmap='bwr', origin='lower')
        ax[1][2].set_title("Residual Difference (b2 - b2_straight)")
        plt.colorbar(im, ax=ax[1][2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show() 

    print(f"--- Beam 1 Straightening ---")
    print(f"Polynomial Coefficients for Beam 1: {poly_coeffs_b1}\n")    
    print(f"--- Beam 2 Straightening ---")
    print(f"Polynomial Coefficients for Beam 2: {poly_coeffs_b2}\n")    

    #This measures the exact shift needed to register b2 (moving) onto b1 (reference).
    # upsample_factor=100 enables ultra-precise 0.01 sub-pixel registration.
    shift, error, diffphase = phase_cross_correlation(
        b2_flat_straight, 
        b1_flat_straight, 
        upsample_factor=100
    )
    shift_y, shift_x = shift[0], shift[1]

    tform = AffineTransform(translation=(shift_x, shift_y))

    b2_flat_reg = warp(b2_flat_straight, tform, order=1)
    
    # Find scale factor 's' by minimizing intensity variance between channels
    def loss_function(s):
        diff = b1_flat_straight - s * b2_flat_reg
        return np.std(diff)

    # from scal=fminbnd(@(s) std(mean(f1R(:,c1wrg)-s*f2R(:,c1wrg),1)),0.5,1.5); in get_tform.m
    res = minimize_scalar(loss_function, bounds=(0.5, 1.5), method='bounded')
    s_factor = res.x
    
    print(f"--- Calibration Computed ---")
    print(f"Measured Alignment Shift -> X: {shift_x:.3f}px, Y: {shift_y:.3f}px")
    print(f"Calculated Gain Scale Factor (s): {s_factor:.4f}\n")

    # Normalize flats by their mean values so they only correct spatial response, not global intensity
    b1_flat_norm = b1_spatial_flat / np.mean(b1_spatial_flat)
    b2_flat_norm = b2_spatial_flat / np.mean(b2_spatial_flat)

    #Check the b1_flat_norm and b2_flat_norm after dark subtraction, straightening, and flat-fielding
    if plot_check == True:
        plt.ioff()
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharex=True, sharey=True)
        ax[0].imshow(b1_flat_norm, cmap='gray', origin='lower')
        ax[0].set_title(f"Beam 1 Normalized Flat")
        ax[1].imshow(b2_flat_norm, cmap='gray', origin='lower')
        ax[1].set_title(f"Beam 2 Normalized Flat")
        plt.tight_layout()
        plt.show()

    return tform, s_factor, config_roi, poly_coeffs_b1, poly_coeffs_b2, b1_flat_norm, b2_flat_norm

def process_science_data(science_filepath, tform, s_factor, poly_coeffs_b1=None, 
                        poly_coeffs_b2=None, config_roi=None, dark_filepath=None, 
                        b1_flat=None, b2_flat=None, plot_check=False):
    """
    Applies initial dark subtraction, maps alignment, normalizes gain, 
    and resolves the 4 Stokes components from the modulation sequence.
    """
    # 1. Open with memmap=True (Do not use 'with' so the file stays mapped)
    hdul = fits.open(science_filepath)
    header = hdul[0].header

    # Keep it as an int16 view to save massive RAM
    raw_data_mapped = np.asanyarray(hdul[0].data).view(np.int16)
    bzero = header.get('BZERO', 0)

    # For validation of the procedure
    input_local='no'
    
    print("Dark correction and Stokes demodulation will be applied to the science data...")
        
    # --- CRITICAL STEP: DARK CORRECTION (Performed first) ---
    if dark_filepath:
        print(f"Applying dark frame correction to science data...")
        with fits.open(dark_filepath) as hdul:
            data_raw_dark = hdul[0].data
            dark_header = hdul[0].header
            dark_raw = np.asanyarray(data_raw_dark).view(np.int16).astype(np.float32)
            bzero = dark_header.get('BZERO', 0)
            if bzero != 0:
                dark_raw += bzero

        dark_data = np.mean(dark_raw, axis=0) if dark_raw.ndim == 3 else dark_raw
        del dark_raw # Immediately free memory
        dark_data = dark_data.astype(np.float32)

    frames, height, width = raw_data_mapped.shape
    half_y = height // 2

    seq_stk = header["SEQ_STOK"].split(" ")
    nsew_stk = len(seq_stk)
    scans = header["NAXIS3"]//nsew_stk

    x_start, x_end = config_roi.get('crg1', [0, width]) if config_roi else [0, width]
    y_start, y_end = config_roi.get('lrg1', [0, half_y]) if config_roi else [0, half_y]
    
    print("Analyzing frame intensities to calculate scaling profiles within clean ROI...")
    avb = np.zeros(frames, dtype=np.float32)
    avb1 = np.zeros(frames, dtype=np.float32)
    avb2 = np.zeros(frames, dtype=np.float32)

    for idx in tqdm(range(frames), desc="Profiling Frames", unit="frame"):
        # 1. Get raw frame stats
        raw_b1 = raw_data_mapped[idx, 0:half_y, :].astype(np.float32) + bzero
        raw_b2 = raw_data_mapped[idx, half_y:, :].astype(np.float32) + bzero

        avb[idx] = np.mean(raw_b1[y_start:y_end, x_start:x_end] + 
                           raw_b2[y_start:y_end, x_start:x_end]) / 2.0 # Combined global mean equivalent

        # 2. Process frame just enough to grab the mean, then discard it from RAM
        if dark_data is not None:
            raw_b1 -= dark_data[0:half_y, :]
            raw_b2 -= dark_data[half_y:, :]

        if poly_coeffs_b1 is not None and poly_coeffs_b2 is not None and config_roi is not None:
            raw_b1, _ = maping_coord(raw_b1, poly_coeffs_b1, config_roi['line_center_b1'], order=1, mode='nearest')
            raw_b2, _ = maping_coord(raw_b2, poly_coeffs_b2, config_roi['line_center_b2'], order=1, mode='nearest')

        if b1_flat is not None:
            raw_b1 /= b1_flat
        if b2_flat is not None:
            raw_b2 /= b2_flat

        raw_b2_reg = warp(raw_b2, tform, order=1)

        # Store only the 1D scalar vector tracks[cite: 5]
        avb1[idx] = np.mean(raw_b1[y_start:y_end, x_start:x_end])
        avb2[idx] = np.mean(raw_b2_reg[y_start:y_end, x_start:x_end])

    # Compute global scalars
    min_avb, max_avb = np.min(avb), np.max(avb)
    min_avb1, max_avb1 = np.min(avb1), np.max(avb1)
    min_avb2, max_avb2 = np.min(avb2), np.max(avb2)

    # Pre-allocate ONLY the final output array
    scan_stokes = np.zeros((scans, 4, half_y, width), dtype=np.float32)

    print("Processing and demodulating Stokes parameters...")
    
    for s in tqdm(range(scans), desc="Processing Scans", unit="scan"):
        # Temporary small storage just for the current scan's modulation frames
        b1_chunk = np.zeros((nsew_stk, half_y, width), dtype=np.float32)
        b2_chunk = np.zeros((nsew_stk, half_y, width), dtype=np.float32)

        for k in range(nsew_stk):
            frame_idx = s * nsew_stk + k

            # Load individual 2D frame slice
            b1_frame = raw_data_mapped[frame_idx, 0:half_y, :].astype(np.float32) + bzero
            b2_frame = raw_data_mapped[frame_idx, half_y:, :].astype(np.float32) + bzero

            if dark_data is not None:
                b1_frame -= dark_data[0:half_y, :]
                b2_frame -= dark_data[half_y:, :]

            if poly_coeffs_b1 is not None and poly_coeffs_b2 is not None and config_roi is not None:
                b1_frame, _ = maping_coord(b1_frame, poly_coeffs_b1, config_roi['line_center_b1'], order=1, mode='nearest')
                b2_frame, _ = maping_coord(b2_frame, poly_coeffs_b2, config_roi['line_center_b2'], order=1, mode='nearest')

            if b1_flat is not None:
                b1_frame /= b1_flat
            if b2_flat is not None:
                b2_frame /= b2_flat

            b2_frame_reg = warp(b2_frame, tform, order=1)

            b1_chunk[k] = min_avb + ((b1_frame - min_avb1) * (max_avb - min_avb)) / (max_avb1 - min_avb1)
            b2_chunk[k] = min_avb + ((b2_frame_reg - min_avb2) * (max_avb - min_avb)) / (max_avb2 - min_avb2)

        # Calculate differences and sums on the chunk
        bs = (b1_chunk - s_factor * b2_chunk) / 2.0
        is_map = (b1_chunk + b2_chunk) / 2.0
    
        # Save straight into the final pre-allocated matrix
        scan_stokes[s, 0] = np.mean(is_map, axis=0)            
        scan_stokes[s, 1] = (bs[0] - bs[1]) / 2.0     
        scan_stokes[s, 2] = (bs[2] - bs[3]) / 2.0     
        scan_stokes[s, 3] = (bs[4] - bs[5]) / 2.0

        if input_local == 'no':
            plt.ioff()
            fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
            stokes_labels = ['I', 'Q', 'U', 'V']
            for i in range(4):
                ax[i//2, i%2].imshow(scan_stokes[s, i, :, config_roi['srg'][0]:config_roi['srg'][1]], cmap='gray', origin='lower')
                ax[i//2, i%2].set_title(f"Stokes {stokes_labels[i]}")
            plt.tight_layout()
            plt.show()

            input_local = input("  Validate the calculation? [yes/no]  ")
            plt.close(fig) 

    hdul.close()
    return scan_stokes, header


if __name__ == "__main__":
    # Define your paths (Optional: Set dark_file to None if no dark frame exists)
    #raw_file_sci = "250206_AR13981flaring/t001_b0303_sp_20250206_100733_b3.fts"
    #raw_file_flats = "250206_AR13981flaring/t013_b0303_sp_20250206_112630_y3.fts"
    #raw_file_darks = "250206_AR13981flaring/t058_b0303_sp_20250206_153655_x3.fts"

    raw_file_flats = "260522_observation_test/t012_b0606_sp_20260522_073915_y3.fts"
    raw_file_darks = "260522_observation_test/t013_b0606_sp_20260522_073955_x3.fts"
    raw_file_sci = "260522_observation_test/t014_b0606_sp_20260522_080309_b3.fts"

    # Step 1: Compute alignment and scale factor (handling darks first if present)

    tform_matrix, s, config_roi, poly_coeffs_b1, poly_coeffs_b2, b1_flat_norm, b2_flat_norm = compute_calibration(raw_file_flats, dark_filepath=raw_file_darks)

    # Step 1.1 (Optional) Evaluate the calibration visually
    
    #process_flats_eval(raw_file_flats, tform_matrix, s, poly_coeffs_b1=poly_coeffs_b1, poly_coeffs_b2=poly_coeffs_b2, config_roi=config_roi, dark_filepath=raw_file_darks)

    # Step 2: Extract final Stokes matrix (handling darks first if present)
    stokes_cube, fits_header = process_science_data(raw_file_sci, tform_matrix, s, poly_coeffs_b1=poly_coeffs_b1, poly_coeffs_b2=poly_coeffs_b2, 
                                                    config_roi=config_roi, dark_filepath=raw_file_darks, b1_flat=b1_flat_norm, b2_flat=b2_flat_norm)
    print(f"Final Stokes Cube Shape: {stokes_cube.shape}")
    plt.ioff()
    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    stokes_labels = ['I', 'Q', 'U', 'V']
    for i in range(4):
        ax[i//2, i%2].imshow(stokes_cube[0, i, :,:], cmap='gray', origin='lower')
        ax[i//2, i%2].set_title(f"Stokes {stokes_labels[i]}")
    plt.tight_layout()
    plt.show()
    ## Save output
    hdu = fits.PrimaryHDU(stokes_cube, header=fits_header)
    hdu.writeto("stokes_output_s4.fits", overwrite=True)
    print("Successfully saved clean Stokes parameters to 'stokes_output_s4.fits'")



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




